# Credit to https://github.com/orbital-materials/orb-models

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax_md import partition, space

jax.config.update("jax_default_matmul_precision", "highest")

ORB_MODEL_PATHS = {
    "orb-jax-v3-conservative-omol": Path(__file__)
    .resolve()
    .with_name("orb-v3-conservative-omol.eqx"),
}


def get_neighbors(
    positions,
    box=None,
    *,
    cutoff: float,
    cell_atom_threshold: int = 64,
    cell_capacity_multiplier: float = 1.5,
    neighbors=None,
    periodic: bool = False,
    dr_threshold: float = 0.0,
):
    num_atoms = int(positions.shape[0])
    use_cell_list = periodic and num_atoms >= cell_atom_threshold
    if periodic:
        if box is None:
            raise ValueError("periodic neighbor lists require a box.")
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
        dr_threshold=float(dr_threshold),
        capacity_multiplier=float(cell_capacity_multiplier),
        disable_cell_list=not use_cell_list,
        mask_self=True,
        format=partition.NeighborListFormat.Dense,
    )
    return neighbor_fn.allocate(
        positions,
        **neighbor_kwargs,
    )


def dense_neighbor_edges(
    positions,
    neighbor_idx,
    *,
    box_vectors=None,
    cutoff: float | None = None,
    max_num_neighbors: int | None = None,
):
    num_atoms = positions.shape[0]
    atom_ids = jnp.arange(num_atoms, dtype=jnp.int32)
    neighbor_idx = jnp.asarray(neighbor_idx, dtype=jnp.int32)
    if max_num_neighbors is not None:
        neighbor_idx = neighbor_idx[:, : int(max_num_neighbors)]
    edge_mask = (neighbor_idx >= 0) & (neighbor_idx < num_atoms)
    safe_neighbor_idx = jnp.where(edge_mask, neighbor_idx, atom_ids[:, None])

    neighbor_positions = positions[safe_neighbor_idx]
    if box_vectors is None:
        edge_vectors = neighbor_positions - positions[:, None, :]
    else:
        jax_box = jnp.swapaxes(jnp.asarray(box_vectors, dtype=positions.dtype), -1, -2)
        displacement, _ = space.periodic_general(
            jax_box,
            fractional_coordinates=False,
        )
        edge_vectors = space.map_neighbor(displacement)(positions, neighbor_positions)

    distances = jnp.linalg.norm(edge_vectors, axis=-1)
    edge_mask = edge_mask & (safe_neighbor_idx != atom_ids[:, None]) & (distances > 1.0e-8)
    if cutoff is not None:
        edge_mask = edge_mask & (distances < cutoff)

    edge_vectors = jnp.where(edge_mask[..., None], edge_vectors, 0.0)
    senders = jnp.broadcast_to(atom_ids[:, None], safe_neighbor_idx.shape)
    return (
        edge_vectors.reshape(-1, 3),
        senders.reshape(-1),
        safe_neighbor_idx.reshape(-1),
        edge_mask.reshape(-1),
    )


def polynomial_cutoff(r: Array, r_max: float | Array, p: float) -> Array:
    ratio = r / r_max
    envelope = (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * ratio**p
        + p * (p + 2.0) * ratio ** (p + 1.0)
        - (p * (p + 1.0) / 2.0) * ratio ** (p + 2.0)
    )
    return jnp.where(r < r_max, envelope, 0.0)


def bessel_basis(
    r: Array,
    bessel_weights: Array,
    prefactor: Array,
) -> Array:
    safe_r = jnp.maximum(r, 1.0e-7)
    return prefactor * (jnp.sin(bessel_weights[None, :] * safe_r[:, None]) / safe_r[:, None])


def condition_nodes(
    charge_embedding: Array,
    spin_embedding: Array,
    total_charge: Array,
    total_spin: Array,
    num_atoms: int,
) -> Array:
    charge_proj = total_charge[:, None] * charge_embedding[None, :] * (2.0 * jnp.pi)
    spin_proj = total_spin[:, None] * spin_embedding[None, :] * (2.0 * jnp.pi)
    charge_emb = jnp.concatenate([jnp.sin(charge_proj), jnp.cos(charge_proj)], axis=-1)
    spin_emb = jnp.concatenate([jnp.sin(spin_proj), jnp.cos(spin_proj)], axis=-1)
    spin_emb = jnp.where(total_spin[:, None] == 0, 0.0, spin_emb)
    return jnp.repeat(jnp.concatenate([charge_emb, spin_emb], axis=-1), num_atoms, axis=0)


def safe_norm(x: Array, *, axis=-1, keepdims: bool = False, eps: float = 1.0e-24) -> Array:
    return jnp.sqrt(jnp.maximum(jnp.sum(x * x, axis=axis, keepdims=keepdims), eps))


def spherical_harmonics_0_to_3(edge_vectors: Array) -> Array:
    xyz = edge_vectors / safe_norm(edge_vectors, axis=-1, keepdims=True)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    sh_0_0 = jnp.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = jnp.sqrt(3.0) * x * z
    sh_2_1 = jnp.sqrt(3.0) * x * y
    y2 = y**2
    x2z2 = x**2 + z**2
    sh_2_2 = y2 - 0.5 * x2z2
    sh_2_3 = jnp.sqrt(3.0) * y * z
    sh_2_4 = jnp.sqrt(3.0) / 2.0 * (z**2 - x**2)
    sh_3_0 = jnp.sqrt(5.0 / 6.0) * (sh_2_0 * z + sh_2_4 * x)
    sh_3_1 = jnp.sqrt(5.0) * sh_2_0 * y
    sh_3_2 = jnp.sqrt(3.0 / 8.0) * (4.0 * y2 - x2z2) * x
    sh_3_3 = 0.5 * y * (2.0 * y2 - 3.0 * x2z2)
    sh_3_4 = jnp.sqrt(3.0 / 8.0) * z * (4.0 * y2 - x2z2)
    sh_3_5 = jnp.sqrt(5.0) * sh_2_4 * y
    sh_3_6 = jnp.sqrt(5.0 / 6.0) * (sh_2_4 * z - sh_2_0 * x)
    sh = jnp.stack(
        [
            sh_0_0,
            sh_1_0,
            sh_1_1,
            sh_1_2,
            sh_2_0,
            sh_2_1,
            sh_2_2,
            sh_2_3,
            sh_2_4,
            sh_3_0,
            sh_3_1,
            sh_3_2,
            sh_3_3,
            sh_3_4,
            sh_3_5,
            sh_3_6,
        ],
        axis=-1,
    )
    component_scale = jnp.array(
        [1.0] + [jnp.sqrt(3.0)] * 3 + [jnp.sqrt(5.0)] * 5 + [jnp.sqrt(7.0)] * 7,
        dtype=sh.dtype,
    )
    return sh * component_scale


class Linear(eqx.Module):
    weight: Array
    bias: Array

    def __init__(
        self,
        config: dict[str, Any],
        prefix: str,
    ) -> None:
        self.weight = jnp.zeros(
            tuple(config["params"][f"{prefix}.weight"]),
            dtype=np.dtype(config["parameter_dtype"]),
        )
        self.bias = jnp.zeros(
            tuple(config["params"][f"{prefix}.bias"]),
            dtype=np.dtype(config["parameter_dtype"]),
        )

    def __call__(self, x: Array) -> Array:
        return x @ jnp.swapaxes(self.weight, -1, -2) + self.bias


class MLP(eqx.Module):
    layers: tuple[Linear, ...]

    def __init__(
        self,
        config: dict[str, Any],
        prefix: str,
        num_layers: int,
    ) -> None:
        self.layers = tuple(Linear(config, f"{prefix}.NN-{i}") for i in range(num_layers))

    def __call__(self, x: Array) -> Array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = jax.nn.silu(x)
        return x


class MLPNorm(eqx.Module):
    mlp: MLP
    norm_weight: Array

    def __init__(
        self,
        config: dict[str, Any],
        prefix: str,
    ) -> None:
        self.mlp = MLP(
            config,
            f"{prefix}.mlp",
            int(config["mlp_num_layers"]),
        )
        self.norm_weight = jnp.zeros(
            tuple(config["params"][f"{prefix}.layer_norm.weight"]),
            dtype=np.dtype(config["parameter_dtype"]),
        )

    def __call__(self, x: Array) -> Array:
        x = self.mlp(x)
        eps = jnp.asarray(jnp.finfo(x.dtype).eps, dtype=x.dtype)
        scale = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
        return x * scale * self.norm_weight


class AttentionBlock(eqx.Module):
    cond_node_proj: Linear
    receive_attn: Linear
    send_attn: Linear
    edge_mlp: MLPNorm
    node_mlp: MLPNorm

    def __init__(
        self,
        config: dict[str, Any],
        prefix: str,
    ) -> None:
        self.cond_node_proj = Linear(config, f"{prefix}._cond_node_proj")
        self.receive_attn = Linear(config, f"{prefix}._receive_attn")
        self.send_attn = Linear(config, f"{prefix}._send_attn")
        self.edge_mlp = MLPNorm(config, f"{prefix}._edge_mlp")
        self.node_mlp = MLPNorm(config, f"{prefix}._node_mlp")

    def __call__(
        self,
        nodes: Array,
        edges: Array,
        cond_nodes: Array,
        senders: Array,
        receivers: Array,
        cutoff: Array,
    ) -> tuple[Array, Array]:
        nodes_cond = nodes + self.cond_node_proj(cond_nodes)
        receive_attn = jax.nn.sigmoid(self.receive_attn(edges)) * cutoff
        send_attn = jax.nn.sigmoid(self.send_attn(edges)) * cutoff
        edge_features = jnp.concatenate(
            [edges, nodes_cond[senders], nodes_cond[receivers]],
            axis=1,
        )
        updated_edges = self.edge_mlp(edge_features)
        sent = jnp.zeros_like(nodes).at[senders].add(updated_edges * send_attn)
        received = jnp.zeros_like(nodes).at[receivers].add(updated_edges * receive_attn)
        node_features = jnp.concatenate([nodes_cond, received, sent], axis=1)
        updated_nodes = self.node_mlp(node_features)
        return nodes_cond + updated_nodes, edges + updated_edges


class ORBLayer(eqx.Module):
    rbf_bessel_weights: Array
    rbf_prefactor: Array
    atom_embedding: Array
    charge_embedding: Array
    spin_embedding: Array
    encoder_node_fn: MLPNorm
    encoder_edge_fn: MLPNorm
    blocks: tuple[AttentionBlock, ...]
    edge_feature_dim: int = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    cutoff_polynomial_p: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        config: dict[str, Any],
        cutoff: float,
        num_layers: int,
        edge_feature_dim: int,
        cutoff_polynomial_p: float,
    ) -> None:
        self.edge_feature_dim = int(edge_feature_dim)
        self.cutoff = float(cutoff)
        self.cutoff_polynomial_p = float(cutoff_polynomial_p)

        self.rbf_bessel_weights = jnp.zeros(
            tuple(config["params"]["model.rbf_transform.bessel_weights"]),
            dtype=np.dtype(config["parameter_dtype"]),
        )
        self.rbf_prefactor = jnp.zeros(
            tuple(config["params"]["model.rbf_transform.prefactor"]),
            dtype=np.dtype(config["parameter_dtype"]),
        )
        self.atom_embedding = jnp.zeros(
            tuple(config["params"]["model.atom_emb.embeddings.weight"]),
            dtype=np.dtype(config["parameter_dtype"]),
        )
        self.charge_embedding = jnp.zeros(
            tuple(config["params"]["model.conditioner.charge_embedding.W"]),
            dtype=np.dtype(config["parameter_dtype"]),
        )
        self.spin_embedding = jnp.zeros(
            tuple(config["params"]["model.conditioner.spin_embedding.W"]),
            dtype=np.dtype(config["parameter_dtype"]),
        )
        self.encoder_node_fn = MLPNorm(config, "model._encoder._node_fn")
        self.encoder_edge_fn = MLPNorm(config, "model._encoder._edge_fn")
        self.blocks = tuple(
            AttentionBlock(config, f"model.gnn_stacks.{i}") for i in range(num_layers)
        )

    def prepare_static_inputs(
        self,
        species: Array,
        total_charge: Array,
        total_spin: Array,
    ) -> tuple[Array, Array]:
        """Encode position-independent node inputs once for an exported force."""
        node_features = self.encoder_node_fn(self.atom_embedding[species])
        conditioning_features = condition_nodes(
            self.charge_embedding,
            self.spin_embedding,
            total_charge,
            total_spin,
            species.shape[0],
        )
        return node_features, conditioning_features

    def __call__(
        self,
        edge_vectors: Array,
        species: Array,
        senders: Array,
        receivers: Array,
        edge_mask: Array,
        total_charge: Array,
        total_spin: Array,
        *,
        initial_node_features: Array | None = None,
        conditioning_features: Array | None = None,
    ) -> Array:
        distances = safe_norm(edge_vectors, axis=-1)
        rbfs = bessel_basis(distances, self.rbf_bessel_weights, self.rbf_prefactor)
        angular = spherical_harmonics_0_to_3(edge_vectors)
        cutoff = polynomial_cutoff(
            distances,
            self.cutoff,
            self.cutoff_polynomial_p,
        )
        cutoff = cutoff[:, None] * edge_mask[:, None].astype(cutoff.dtype)
        edges_in = (cutoff[:, :, None] * rbfs[:, :, None] * angular[:, None, :]).reshape(
            (senders.shape[0], self.edge_feature_dim)
        )
        if (initial_node_features is None) != (conditioning_features is None):
            raise ValueError(
                "initial_node_features and conditioning_features must be provided together"
            )
        if initial_node_features is None:
            initial_node_features, conditioning_features = self.prepare_static_inputs(
                species,
                total_charge,
                total_spin,
            )
        assert conditioning_features is not None

        nodes = initial_node_features
        edges = self.encoder_edge_fn(edges_in)
        for block in self.blocks:
            nodes, edges = block(
                nodes,
                edges,
                conditioning_features,
                senders,
                receivers,
                cutoff,
            )
        return nodes


class EnergyHead(eqx.Module):
    energy_mlp: MLP
    energy_normalizer_var: Array
    energy_normalizer_mean: Array
    energy_reference_weight: Array

    def __init__(
        self,
        *,
        config: dict[str, Any],
        energy_mlp_num_layers: int,
    ) -> None:
        self.energy_mlp = MLP(
            config,
            "heads.energy.mlp",
            energy_mlp_num_layers,
        )
        self.energy_normalizer_var = jnp.zeros(
            tuple(config["params"]["heads.energy.normalizer.bn.running_var"]),
            dtype=np.dtype(config["parameter_dtype"]),
        )
        self.energy_normalizer_mean = jnp.zeros(
            tuple(config["params"]["heads.energy.normalizer.bn.running_mean"]),
            dtype=np.dtype(config["parameter_dtype"]),
        )
        self.energy_reference_weight = jnp.zeros(
            tuple(config["params"]["heads.energy.reference.linear.weight"]),
            dtype=np.dtype(config["parameter_dtype"]),
        )

    def __call__(self, node_features: Array, species: Array) -> Array:
        graph_features = jnp.mean(node_features, axis=0, keepdims=True)
        x = self.energy_mlp(graph_features).reshape(())
        x = x * jnp.sqrt(self.energy_normalizer_var[0])
        x = x + self.energy_normalizer_mean[0]
        x = x * species.shape[0]
        reference = jnp.sum(self.energy_reference_weight[species])
        return x + reference


class ZBLRepulsion(eqx.Module):
    covalent_radii: Array
    coulomb_ev_angstrom: float = eqx.field(static=True)
    zbl_polynomial_p: float = eqx.field(static=True)
    zbl_atomic_number_exponent: float = eqx.field(static=True)
    zbl_screening_length_scale: float = eqx.field(static=True)
    zbl_screening_weights: tuple[float, ...] = eqx.field(static=True)
    zbl_screening_exponents: tuple[float, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        config: dict[str, Any],
    ) -> None:
        self.coulomb_ev_angstrom = float(config["zbl_coulomb_ev_angstrom"])
        self.zbl_polynomial_p = float(config["zbl_polynomial_p"])
        self.zbl_atomic_number_exponent = float(config["zbl_atomic_number_exponent"])
        self.zbl_screening_length_scale = float(config["zbl_screening_length_scale"])
        self.zbl_screening_weights = tuple(float(x) for x in config["zbl_screening_weights"])
        self.zbl_screening_exponents = tuple(float(x) for x in config["zbl_screening_exponents"])
        self.covalent_radii = jnp.zeros(
            tuple(config["params"]["covalent_radii"]),
            dtype=np.dtype(config["parameter_dtype"]),
        )

    def __call__(
        self,
        species: Array,
        edge_vectors: Array,
        senders: Array,
        receivers: Array,
        edge_mask: Array,
    ) -> Array:
        num_atoms = species.shape[0]
        distances = safe_norm(edge_vectors, axis=-1)
        safe_distances = jnp.maximum(distances, 1.0e-7)
        z_sender = species[senders] + 1
        z_receiver = species[receivers] + 1
        z_sender_f = z_sender.astype(edge_vectors.dtype)
        z_receiver_f = z_receiver.astype(edge_vectors.dtype)
        zbl_exponent = jnp.asarray(self.zbl_atomic_number_exponent, dtype=edge_vectors.dtype)
        screening_length = self.zbl_screening_length_scale / (
            z_sender_f**zbl_exponent + z_receiver_f**zbl_exponent
        )
        scaled_distance = safe_distances / screening_length
        screening_weights = jnp.asarray(
            self.zbl_screening_weights,
            dtype=edge_vectors.dtype,
        )[:, None]
        screening_exponents = jnp.asarray(
            self.zbl_screening_exponents,
            dtype=edge_vectors.dtype,
        )[:, None]
        zbl_screening = jnp.sum(
            screening_weights * jnp.exp(-screening_exponents * scaled_distance[None, :]),
            axis=0,
        )
        bare_nuclear_repulsion = (
            self.coulomb_ev_angstrom * z_sender_f * z_receiver_f / safe_distances
        )
        cutoff_radius = self.covalent_radii[z_sender] + self.covalent_radii[z_receiver]
        orb_envelope = polynomial_cutoff(
            safe_distances,
            cutoff_radius,
            self.zbl_polynomial_p,
        )
        edge_repulsion = 0.5 * bare_nuclear_repulsion * zbl_screening * orb_envelope
        edge_repulsion = edge_repulsion * edge_mask.astype(edge_repulsion.dtype)
        return jnp.sum(edge_repulsion) / num_atoms


class Orb(eqx.Module):
    layer: ORBLayer
    energy_head: EnergyHead
    zbl_repulsion: ZBLRepulsion
    cutoff: float = eqx.field(static=True)
    ev_to_kjmol: float = eqx.field(static=True)
    num_species_embeddings: int = eqx.field(static=True)
    max_num_neighbors: int = eqx.field(static=True)
    neighbor_cell_atom_threshold: int = eqx.field(static=True)
    neighbor_cell_capacity_multiplier: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        config: dict[str, Any],
    ) -> None:
        self.cutoff = float(config["cutoff"])
        self.ev_to_kjmol = float(config["ev_to_kjmol"])
        self.num_species_embeddings = int(config["num_species_embeddings"])
        self.max_num_neighbors = int(config["max_num_neighbors"])
        self.neighbor_cell_atom_threshold = int(config["neighbor_cell_atom_threshold"])
        self.neighbor_cell_capacity_multiplier = float(config["neighbor_cell_capacity_multiplier"])
        num_layers = int(config["num_layers"])
        energy_mlp_num_layers = int(config["energy_mlp_num_layers"])
        edge_feature_dim = int(config["edge_feature_dim"])
        cutoff_polynomial_p = float(config["cutoff_polynomial_p"])

        self.layer = ORBLayer(
            config=config,
            cutoff=self.cutoff,
            num_layers=num_layers,
            edge_feature_dim=edge_feature_dim,
            cutoff_polynomial_p=cutoff_polynomial_p,
        )
        self.energy_head = EnergyHead(
            config=config,
            energy_mlp_num_layers=energy_mlp_num_layers,
        )
        self.zbl_repulsion = ZBLRepulsion(config=config)

    def __call__(
        self,
        positions_angstrom: Array,
        species: Array,
        total_charge: Array,
        total_spin: Array,
        *,
        box_vectors: Array | None = None,
        neighbors=None,
        neighbor_idx: Array | None = None,
        periodic: bool = False,
        initial_node_features: Array | None = None,
        conditioning_features: Array | None = None,
    ) -> Array:
        if neighbor_idx is None:
            neighbors = get_neighbors(
                positions_angstrom,
                box_vectors if periodic else None,
                cutoff=float(self.cutoff),
                cell_atom_threshold=int(self.neighbor_cell_atom_threshold),
                cell_capacity_multiplier=float(self.neighbor_cell_capacity_multiplier),
                neighbors=neighbors,
                periodic=periodic,
            )
            neighbor_idx = neighbors.idx
        box_vectors = box_vectors if periodic else None
        edge_vectors, senders, receivers, edge_mask = dense_neighbor_edges(
            positions_angstrom,
            neighbor_idx,
            box_vectors=box_vectors,
            cutoff=float(self.cutoff),
            max_num_neighbors=self.max_num_neighbors,
        )
        node_features = self.layer(
            edge_vectors,
            species,
            senders,
            receivers,
            edge_mask,
            total_charge,
            total_spin,
            initial_node_features=initial_node_features,
            conditioning_features=conditioning_features,
        )
        graph_energy = self.energy_head(node_features, species)
        zbl_energy = self.zbl_repulsion(
            species,
            edge_vectors,
            senders,
            receivers,
            edge_mask,
        )
        return graph_energy + zbl_energy


def load_model(
    model: str | PathLike = "orb-jax-v3-conservative-omol",
    *,
    dtype=jnp.float32,
    atomic_numbers=None,
    neighbor_cell_atom_threshold: int | None = None,
    neighbor_cell_capacity_multiplier: float | None = None,
) -> Orb:
    path = (
        ORB_MODEL_PATHS[model]
        if isinstance(model, str) and model in ORB_MODEL_PATHS
        else Path(model)
    )

    with jax.enable_x64(jnp.dtype(dtype) == jnp.dtype(jnp.float64)), path.open("rb") as handle:
        config = json.loads(handle.readline().decode("utf-8"))
        if neighbor_cell_atom_threshold is not None:
            config["neighbor_cell_atom_threshold"] = int(neighbor_cell_atom_threshold)
        if neighbor_cell_capacity_multiplier is not None:
            config["neighbor_cell_capacity_multiplier"] = float(neighbor_cell_capacity_multiplier)
        model_template = Orb(config=config)
        loaded_model = eqx.tree_deserialise_leaves(handle, model_template)
        loaded_model = jax.tree_util.tree_map(
            lambda value: value.astype(dtype)
            if eqx.is_array(value) and jnp.issubdtype(value.dtype, jnp.floating)
            else value,
            loaded_model,
        )
        if atomic_numbers is not None:
            species = np.asarray(jax.device_get(atomic_numbers), dtype=np.int64).reshape(-1)
            unsupported = sorted(
                z
                for z in set(species.tolist())
                if z < 0 or z >= loaded_model.num_species_embeddings
            )
            if unsupported:
                raise ValueError(f"ORB does not support atomic numbers {unsupported}.")
        return loaded_model
