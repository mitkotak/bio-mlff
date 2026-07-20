from __future__ import annotations

import json
import math
from os import PathLike
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax_md import partition, space

jax.config.update("jax_default_matmul_precision", "highest")

DEFAULT_ENSEMBLE_MODEL_PATH = Path(__file__).resolve().with_name("ani2x_ensemble.eqx")
DEFAULT_SINGLE_MODEL_PATH = Path(__file__).resolve().with_name("ani2x_model0.eqx")
ANI2X_MODEL_PATHS = {
    "ani2x-jax-ensemble": DEFAULT_ENSEMBLE_MODEL_PATH,
    "ani2x-jax-model0": DEFAULT_SINGLE_MODEL_PATH,
}
ANI2X_MODEL_NAMES = tuple(ANI2X_MODEL_PATHS)


def dense_neighbor_edges(
    positions,
    neighbor_idx,
    *,
    box_vectors=None,
):
    num_atoms = positions.shape[0]
    atom_ids = jnp.arange(num_atoms, dtype=jnp.int32)
    neighbor_idx = jnp.asarray(neighbor_idx, dtype=jnp.int32)
    edge_mask = (neighbor_idx >= 0) & (neighbor_idx < num_atoms)
    safe_neighbor_idx = jnp.where(edge_mask, neighbor_idx, atom_ids[:, None])

    neighbor_positions = positions[safe_neighbor_idx]
    if box_vectors is None:
        edge_vectors = neighbor_positions - positions[:, None, :]
    else:
        displacement, _ = space.periodic_general(
            jnp.swapaxes(jnp.asarray(box_vectors, dtype=positions.dtype), -1, -2),
            fractional_coordinates=False,
        )
        edge_vectors = space.map_neighbor(displacement)(positions, neighbor_positions)
    edge_vectors = jnp.where(edge_mask[..., None], edge_vectors, 0.0)
    return edge_vectors, safe_neighbor_idx, edge_mask


def get_neighbors(
    positions,
    box=None,
    *,
    cell_atom_threshold: int,
    cutoff: float,
    cell_capacity_multiplier: float,
    neighbors=None,
    periodic: bool = False,
):
    num_atoms = int(positions.shape[0])
    use_cell_list = periodic and num_atoms >= cell_atom_threshold
    if periodic:
        if box is None:
            raise ValueError("periodic neighbor lists require OpenMM box vectors.")
        jax_box = jnp.swapaxes(jnp.asarray(box, dtype=positions.dtype), -1, -2)
        displacement, _ = space.periodic_general(
            jax_box,
            fractional_coordinates=False,
        )
        neighbor_kwargs = {"box": jax_box}
    else:
        displacement, _ = space.free()
        neighbor_kwargs = {}

    if neighbors is not None:
        return neighbors.update(positions)

    neighbor_fn = partition.neighbor_list(
        displacement,
        jnp.asarray(1.0, dtype=positions.dtype),
        float(cutoff),
        dr_threshold=0.0,
        capacity_multiplier=float(cell_capacity_multiplier),
        disable_cell_list=not use_cell_list,
        mask_self=True,
        format=partition.NeighborListFormat.Dense,
    )
    return neighbor_fn.allocate(
        positions,
        **neighbor_kwargs,
    )


def piecewise_cutoff(distance, cutoff: float):
    """ANI cosine cutoff."""
    return 0.5 * jnp.cos(distance * (math.pi / cutoff)) + 0.5


class ANI2xCheckpoint(eqx.Module):
    """Full on-disk ANI2x checkpoint leaves before active-species pruning."""

    atom_energies: jnp.ndarray
    layer_weights: list
    layer_biases: list

    def __init__(
        self,
        config: dict,
        *,
        dtype: Any = jnp.float32,
        atom_energy_dtype: Any | None = None,
    ):
        num_species = len(config["species_order"])
        num_models = config["num_models"]
        network_sizes = tuple(config["network_sizes"])

        self.atom_energies = jnp.zeros(
            num_species,
            dtype=dtype if atom_energy_dtype is None else atom_energy_dtype,
        )
        self.layer_weights = []
        self.layer_biases = []
        for layer_index in range(len(network_sizes) - 1):
            d_in = network_sizes[layer_index]
            d_out = network_sizes[layer_index + 1]
            self.layer_weights.append(
                jnp.zeros((num_models, num_species, d_in, d_out), dtype=dtype)
            )
            self.layer_biases.append(jnp.zeros((num_models, num_species, d_out), dtype=dtype))


def active_pair_ids(
    active_species: tuple[int, ...],
    pair_to_index: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    active_species_np = np.asarray(active_species, dtype=np.int32)
    pair_to_index_np = np.asarray(pair_to_index, dtype=np.int32)
    return tuple(
        int(pair_id)
        for pair_id in np.unique(pair_to_index_np[np.ix_(active_species_np, active_species_np)])
    )


def lookup_table(active_ids: tuple[int, ...], full_size: int) -> tuple[int, ...]:
    active_ids_np = np.asarray(active_ids, dtype=np.int32)
    lookup = np.full(full_size, -1, dtype=np.int32)
    lookup[active_ids_np] = np.arange(len(active_ids_np), dtype=np.int32)
    return tuple(int(x) for x in lookup.tolist())


def pair_index_table(num_species: int) -> tuple[tuple[int, ...], ...]:
    table = np.zeros((num_species, num_species), dtype=np.int32)
    pair_index = 0
    for species_i in range(num_species):
        for species_j in range(species_i, num_species):
            table[species_i, species_j] = pair_index
            table[species_j, species_i] = pair_index
            pair_index += 1
    return tuple(tuple(int(index) for index in row) for row in table)


def species_index_table(species_order: tuple[int, ...]) -> tuple[int, ...]:
    table = [-1] * (max(species_order) + 1)
    for species_index, atomic_number in enumerate(species_order):
        table[atomic_number] = species_index
    return tuple(table)


def basis_block_columns(
    active_ids: tuple[int, ...],
    block_width: int,
    *,
    offset: int = 0,
) -> np.ndarray:
    active_ids_np = np.asarray(active_ids, dtype=np.int32)
    block_offsets = active_ids_np[:, None] * block_width
    block_columns = np.arange(block_width, dtype=np.int32)
    return (offset + block_offsets + block_columns).reshape(-1)


def first_layer_columns(
    active_species: tuple[int, ...],
    active_pairs: tuple[int, ...],
    *,
    num_species: int,
    radial_divisions: int,
    angular_basis_width: int,
) -> np.ndarray:
    radial_cols = basis_block_columns(active_species, radial_divisions)
    angular_cols = basis_block_columns(
        active_pairs,
        angular_basis_width,
        offset=num_species * radial_divisions,
    )
    return np.concatenate((radial_cols, angular_cols))


class ANI2x(eqx.Module):
    # Runtime leaves pruned from the full checkpoint.
    atom_energies: jnp.ndarray
    layer_weights: list
    layer_biases: list

    neighbor_cell_atom_threshold: int = eqx.field(static=True)
    neighbor_cell_capacity_multiplier: float = eqx.field(static=True)
    radial_eta: float = eqx.field(static=True)
    angular_eta: float = eqx.field(static=True)
    zeta: float = eqx.field(static=True)
    radial_cutoff: float = eqx.field(static=True)
    angular_cutoff: float = eqx.field(static=True)
    celu_alpha: float = eqx.field(static=True)
    radial_shifts: tuple[float, ...] = eqx.field(static=True)
    angular_shifts: tuple[float, ...] = eqx.field(static=True)
    angular_radial_shifts: tuple[float, ...] = eqx.field(static=True)
    species_to_index: tuple[int, ...] = eqx.field(static=True)
    pair_to_index: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    species_lookup: tuple[int, ...] = eqx.field(static=True)
    pair_lookup: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        config: dict,
        checkpoint: ANI2xCheckpoint,
        active_species: tuple[int, ...] | None = None,
    ):
        self.neighbor_cell_atom_threshold = config["neighbor_cell_atom_threshold"]
        self.neighbor_cell_capacity_multiplier = config["neighbor_cell_capacity_multiplier"]
        self.radial_eta = config["radial_eta"]
        self.angular_eta = config["angular_eta"]
        self.zeta = config["zeta"]
        self.radial_cutoff = config["radial_cutoff"]
        self.angular_cutoff = config["angular_cutoff"]
        self.celu_alpha = config["celu_alpha"]
        self.radial_shifts = tuple(config["radial_shifts"])
        self.angular_shifts = tuple(config["angular_shifts"])
        self.angular_radial_shifts = tuple(config["angular_radial_shifts"])
        species_order = tuple(int(z) for z in config["species_order"])
        self.species_to_index = species_index_table(species_order)
        num_species = len(species_order)
        self.pair_to_index = pair_index_table(num_species)
        num_species_pairs = num_species * (num_species + 1) // 2
        radial_divisions = len(self.radial_shifts)
        angular_basis_width = len(self.angular_shifts) * len(self.angular_radial_shifts)

        if active_species is None:
            active_species = tuple(range(num_species))
        else:
            if not active_species:
                raise ValueError("ANI active species cannot be empty.")
            invalid = [x for x in active_species if x < 0 or x >= num_species]
            if invalid:
                raise ValueError(
                    "ANI active species must be ANI species indices in "
                    f"[0, {num_species}); got {invalid}."
                )

        active_pairs = active_pair_ids(active_species, self.pair_to_index)

        self.species_lookup = lookup_table(active_species, num_species)
        self.pair_lookup = lookup_table(active_pairs, num_species_pairs)
        first_layer_cols = first_layer_columns(
            active_species,
            active_pairs,
            num_species=num_species,
            radial_divisions=radial_divisions,
            angular_basis_width=angular_basis_width,
        )

        active_species_idx = jnp.asarray(active_species, dtype=jnp.int32)
        first_layer_cols = jnp.asarray(first_layer_cols, dtype=jnp.int32)
        self.atom_energies = checkpoint.atom_energies[active_species_idx]
        layer_weights = []
        layer_biases = []
        for layer_index, checkpoint_weights in enumerate(checkpoint.layer_weights):
            weights = checkpoint_weights[:, active_species_idx]
            if layer_index == 0:
                weights = weights[:, :, first_layer_cols, :]
            layer_weights.append(weights)
            checkpoint_biases = checkpoint.layer_biases[layer_index]
            layer_biases.append(checkpoint_biases[:, active_species_idx])
        self.layer_weights = layer_weights
        self.layer_biases = layer_biases

    def species_indices(self, atomic_numbers) -> jnp.ndarray:
        atomic_numbers_array = np.asarray(jax.device_get(atomic_numbers)).reshape(-1)
        species_to_index = np.asarray(self.species_to_index, dtype=np.int32)
        unsupported = sorted(
            z
            for z in set(atomic_numbers_array.tolist())
            if z < 0 or z >= len(species_to_index) or species_to_index[z] < 0
        )
        if unsupported:
            raise ValueError(f"ANI2x does not support atomic numbers {unsupported}.")
        return jnp.asarray(self.species_to_index, dtype=jnp.int32)[
            jnp.asarray(atomic_numbers, dtype=jnp.int32)
        ]

    def local_node_energies(
        self,
        positions,
        species,
        *,
        radial_neighbor_idx,
        angular_neighbor_idx,
        box_vectors,
    ):
        species = jnp.asarray(species, dtype=jnp.int32)
        num_atoms = species.shape[0]
        atom_ids = jnp.arange(num_atoms, dtype=jnp.int32)
        species_lookup = jnp.asarray(self.species_lookup, dtype=jnp.int32)
        pair_lookup = jnp.asarray(self.pair_lookup, dtype=jnp.int32)
        pair_to_index = jnp.asarray(self.pair_to_index, dtype=jnp.int32)
        radial_shifts = jnp.asarray(self.radial_shifts, dtype=positions.dtype)
        angular_shifts = jnp.asarray(self.angular_shifts, dtype=positions.dtype)
        angular_radial_shifts = jnp.asarray(
            self.angular_radial_shifts,
            dtype=positions.dtype,
        )
        num_active_species = self.atom_energies.shape[0]
        num_active_pairs = max(self.pair_lookup) + 1
        radial_divisions = len(self.radial_shifts)
        angular_basis_width = len(self.angular_shifts) * len(self.angular_radial_shifts)

        local_species = species_lookup[species]

        radial_displacements, radial_safe_neighbor_idx, radial_edge_mask = dense_neighbor_edges(
            positions,
            radial_neighbor_idx,
            box_vectors=box_vectors,
        )
        local_radial_neighbor_species = local_species[radial_safe_neighbor_idx]

        # R_ij is the distance between atom i and radial neighbor j.
        radial_distance2 = jnp.sum(radial_displacements**2, axis=-1)
        radial_distance = jnp.sqrt(jnp.clip(radial_distance2, min=1e-5))
        radial_real_neighbor = radial_edge_mask & (radial_safe_neighbor_idx != atom_ids[:, None])

        # Eq. 3: radial symmetry terms, then sum them by species.
        radial_mask = radial_real_neighbor & (radial_distance < self.radial_cutoff)
        radial_switch = piecewise_cutoff(radial_distance, self.radial_cutoff) * radial_mask
        radial_terms = (
            jnp.exp(-self.radial_eta * (radial_distance[..., None] - radial_shifts) ** 2)
            * (0.25 * radial_switch)[..., None]
        )

        radial_active = local_radial_neighbor_species >= 0
        radial_terms = jnp.where(radial_active[..., None], radial_terms, 0.0)
        radial_one_hot = jax.nn.one_hot(
            local_radial_neighbor_species,
            num_active_species,
            dtype=radial_terms.dtype,
        )
        radial_aev = jnp.einsum("nkr,nks->nsr", radial_terms, radial_one_hot).reshape(
            num_atoms,
            num_active_species * radial_divisions,
        )

        angular_displacements, angular_safe_neighbor_idx, angular_edge_mask = dense_neighbor_edges(
            positions,
            angular_neighbor_idx,
            box_vectors=box_vectors,
        )
        angular_neighbor_species = species[angular_safe_neighbor_idx]

        # Eq. 4/5: angular symmetry terms over unique neighbor pairs around atom i.
        angular_distance2 = jnp.sum(angular_displacements**2, axis=-1)
        angular_distance = jnp.sqrt(jnp.clip(angular_distance2, min=1e-5))
        angular_real_neighbor = angular_edge_mask & (
            angular_safe_neighbor_idx != atom_ids[:, None]
        )
        angular_mask = angular_real_neighbor & (angular_distance < self.angular_cutoff)
        angular_switch = piecewise_cutoff(angular_distance, self.angular_cutoff) * angular_mask
        neighbor_i_np, neighbor_j_np = np.triu_indices(int(angular_neighbor_idx.shape[1]), k=1)
        neighbor_i = jnp.asarray(neighbor_i_np, dtype=jnp.int32)
        neighbor_j = jnp.asarray(neighbor_j_np, dtype=jnp.int32)
        pair_mask = angular_mask[:, neighbor_i] & angular_mask[:, neighbor_j]
        vector_i = angular_displacements[:, neighbor_i, :]
        vector_j = angular_displacements[:, neighbor_j, :]
        dot_product = jnp.sum(vector_i * vector_j, axis=-1)
        distance_product = angular_distance[:, neighbor_i] * angular_distance[:, neighbor_j]
        cos_angle = dot_product / jnp.clip(distance_product, min=1.0e-10)
        angle = jnp.arccos(0.95 * cos_angle)

        angular_part = ((1.0 + jnp.cos(angle[..., None] - angular_shifts)) / 2.0) ** self.zeta
        angular_part = (
            2.0
            * angular_part
            * (angular_switch[:, neighbor_i] * angular_switch[:, neighbor_j])[..., None]
        )

        pair_distance = (
            (angular_distance[:, neighbor_i] + angular_distance[:, neighbor_j]) / 2.0
        )[..., None]
        unscaled_radial_shifts = angular_radial_shifts / jnp.sqrt(self.angular_eta)
        angular_radial_part = jnp.exp(
            -self.angular_eta * (pair_distance - unscaled_radial_shifts) ** 2
        )
        angular_terms = (angular_part[..., None, :] * angular_radial_part[..., :, None]).reshape(
            num_atoms, -1, angular_basis_width
        )

        pair_index = pair_to_index[
            angular_neighbor_species[:, neighbor_i],
            angular_neighbor_species[:, neighbor_j],
        ]
        active_pair = pair_lookup[pair_index]
        pair_active = active_pair >= 0
        angular_terms = jnp.where((pair_mask & pair_active)[..., None], angular_terms, 0.0)
        pair_one_hot = jax.nn.one_hot(
            active_pair,
            num_active_pairs,
            dtype=angular_terms.dtype,
        )
        angular_aev = jnp.einsum("npa,nps->nsa", angular_terms, pair_one_hot).reshape(
            num_atoms,
            num_active_pairs * angular_basis_width,
        )

        species_selector = jax.nn.one_hot(
            local_species,
            num_active_species,
            dtype=positions.dtype,
        )

        def select_atom_species(values_by_species):
            return jnp.einsum("ns,nmso->nmo", species_selector, values_by_species)

        radial_width = radial_aev.shape[-1]
        weights = self.layer_weights[0]
        bias = self.layer_biases[0]

        # Evaluate per-species lanes to avoid materializing per-atom weight tensors.
        x = (
            jnp.einsum("ni,msio->nmso", radial_aev, weights[:, :, :radial_width, :])
            + jnp.einsum("ni,msio->nmso", angular_aev, weights[:, :, radial_width:, :])
            + bias[None, :, :, :]
        )
        x = select_atom_species(x)
        if len(self.layer_weights) > 1:
            x = jax.nn.celu(x, alpha=self.celu_alpha)

        for layer_index in range(1, len(self.layer_weights)):
            weights = self.layer_weights[layer_index]
            bias = self.layer_biases[layer_index]
            x = select_atom_species(jnp.einsum("nmi,msio->nmso", x, weights) + bias)
            if layer_index < len(self.layer_weights) - 1:
                x = jax.nn.celu(x, alpha=self.celu_alpha)

        mlp_energies = x.squeeze(-1)

        atom_energies = jax.lax.stop_gradient(self.atom_energies[local_species])
        return jnp.mean(mlp_energies + atom_energies[:, None], axis=1)

    def __call__(
        self,
        positions,
        species,
        *,
        box_vectors=None,
        radial_neighbors=None,
        angular_neighbors=None,
        radial_neighbor_idx=None,
        angular_neighbor_idx=None,
        periodic: bool | None = False,
    ):
        periodic = bool(periodic)
        if radial_neighbor_idx is None:
            radial_neighbors = get_neighbors(
                positions,
                box_vectors,
                cell_atom_threshold=int(self.neighbor_cell_atom_threshold),
                cutoff=float(self.radial_cutoff),
                cell_capacity_multiplier=float(self.neighbor_cell_capacity_multiplier),
                neighbors=radial_neighbors,
                periodic=periodic,
            )
            radial_neighbor_idx = radial_neighbors.idx
        if angular_neighbor_idx is None:
            angular_neighbors = get_neighbors(
                positions,
                box_vectors,
                cell_atom_threshold=int(self.neighbor_cell_atom_threshold),
                cutoff=float(self.angular_cutoff),
                cell_capacity_multiplier=float(self.neighbor_cell_capacity_multiplier),
                neighbors=angular_neighbors,
                periodic=periodic,
            )
            angular_neighbor_idx = angular_neighbors.idx
        node_energies = self.local_node_energies(
            positions,
            species,
            radial_neighbor_idx=radial_neighbor_idx,
            angular_neighbor_idx=angular_neighbor_idx,
            box_vectors=box_vectors if periodic else None,
        )
        return jnp.sum(node_energies)


def load_model(
    model: str | PathLike = "ani2x-jax-ensemble",
    *,
    dtype=jnp.float32,
    atomic_numbers=None,
    neighbor_cell_atom_threshold: int | None = None,
    neighbor_cell_capacity_multiplier: float | None = None,
) -> ANI2x:
    """Load an ANI-2x checkpoint, optionally specialized to a fixed atomic-number set."""
    path = (
        ANI2X_MODEL_PATHS[model]
        if isinstance(model, str) and model in ANI2X_MODEL_PATHS
        else Path(model)
    )

    use_float64 = jnp.dtype(dtype) == jnp.dtype(jnp.float64)
    with jax.enable_x64(use_float64), path.open("rb") as handle:
        config = json.loads(handle.readline().decode("utf-8"))
        if neighbor_cell_atom_threshold is not None:
            config["neighbor_cell_atom_threshold"] = int(neighbor_cell_atom_threshold)
        if neighbor_cell_capacity_multiplier is not None:
            config["neighbor_cell_capacity_multiplier"] = float(neighbor_cell_capacity_multiplier)
        active_species = None
        if atomic_numbers is not None:
            species_order = tuple(int(z) for z in config["species_order"])
            species_to_index = np.asarray(species_index_table(species_order), dtype=np.int32)
            atomic_numbers = np.asarray(jax.device_get(atomic_numbers), dtype=np.int64).reshape(-1)
            unsupported = sorted(set(atomic_numbers.tolist()) - set(species_order))
            if unsupported:
                raise ValueError(f"ANI2x does not support atomic numbers {unsupported}.")
            species = species_to_index[atomic_numbers]
            active_species = tuple(int(x) for x in sorted(set(species.tolist())))
        # Load the full checkpoint before pruning weights for inactive species.
        checkpoint_template = ANI2xCheckpoint(
            config,
            dtype=jnp.float32,
            atom_energy_dtype=jnp.float64 if use_float64 else jnp.float32,
        )
        loaded_model = ANI2x(
            config=config,
            checkpoint=eqx.tree_deserialise_leaves(
                handle,
                checkpoint_template,
            ),
            active_species=active_species,
        )
        return jax.tree_util.tree_map(
            lambda value: value.astype(dtype)
            if eqx.is_array(value) and jnp.issubdtype(value.dtype, jnp.floating)
            else value,
            loaded_model,
        )
