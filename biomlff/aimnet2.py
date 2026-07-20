# Credit to https://github.com/isayevlab/aimnetcentral for the docs

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
from jax import Array
from jax.scipy.special import erfc
from jax_md import partition, space

jax.config.update("jax_default_matmul_precision", "highest")

AIMNET2_MODEL_PATHS = {
    "aimnet2-jax": Path(__file__).resolve().with_name("aimnet2.eqx"),
}
AIMNET2_MODEL_NAMES = tuple(AIMNET2_MODEL_PATHS)


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

    distances = safe_norm(edge_vectors, axis=-1)
    edge_mask = edge_mask & (safe_neighbor_idx != atom_ids[:, None]) & (distances > 1.0e-8)
    if cutoff is not None:
        edge_mask = edge_mask & (distances < cutoff)
    edge_vectors = jnp.where(edge_mask[..., None], edge_vectors, 0.0)
    return edge_vectors, safe_neighbor_idx, edge_mask


def safe_norm(x: Array, *, axis=-1, keepdims: bool = False, eps: float = 1.0e-24) -> Array:
    return jnp.sqrt(jnp.maximum(jnp.sum(x * x, axis=axis, keepdims=keepdims), eps))


class MLP(eqx.Module):
    layers: list["Linear"]

    def __init__(self, sizes: tuple[int, ...], *, dtype: Any = jnp.float32, key: Array):
        sizes = tuple(int(size) for size in sizes)
        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = [
            Linear(in_dim, out_dim, dtype=dtype, key=subkey)
            for subkey, in_dim, out_dim in zip(keys, sizes[:-1], sizes[1:])
        ]

    def __call__(self, x: Array, *, last_linear: bool = True) -> Array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1 or not last_linear:
                x = jax.nn.gelu(x, approximate=False)
        return x


class Linear(eqx.Module):
    weight: Array
    bias: Array

    def __init__(self, in_dim: int, out_dim: int, *, dtype: Any = jnp.float32, key: Array):
        weight_key, bias_key = jax.random.split(key)
        initialization_bound = jnp.sqrt(1.0 / in_dim)
        self.weight = jax.random.uniform(
            weight_key,
            (out_dim, in_dim),
            dtype=dtype,
            minval=-initialization_bound,
            maxval=initialization_bound,
        )
        self.bias = jax.random.uniform(
            bias_key,
            (out_dim,),
            dtype=dtype,
            minval=-initialization_bound,
            maxval=initialization_bound,
        )

    def __call__(self, x: Array) -> Array:
        if x.shape[-1] != self.weight.shape[1]:
            raise ValueError(
                f"Expected feature axis of size {self.weight.shape[1]}, got shape {x.shape}."
            )
        x = x.astype(self.weight.dtype)
        return x @ self.weight.T + self.bias


def d3bj_energy_neighbors(
    positions: Array,
    d3_pre: dict[str, Array],
    neighbor_idx: Array,
    box_vectors: Array | None = None,
    *,
    cutoff: float,
) -> Array:
    edge_vectors, safe_neighbor_idx, edge_mask = dense_neighbor_edges(
        positions,
        neighbor_idx,
        box_vectors=box_vectors,
        cutoff=float(cutoff),
    )
    distances = safe_norm(edge_vectors, axis=-1)
    rij = distances / float(d3_pre["bohr_a"])
    species_indices = d3_pre["species_idx"]
    sp_i = species_indices[:, None]
    sp_j = species_indices[safe_neighbor_idx]
    rcov = d3_pre["rcov"]
    r2r4 = d3_pre["r2r4"]

    rr = (rcov[sp_i] + rcov[sp_j]) / jnp.maximum(rij, 1.0e-8)
    damp = 1.0 / (1.0 + jnp.exp(-float(d3_pre["d3_k1"]) * (rr - 1.0)))
    cn = jnp.sum(jnp.where(edge_mask, damp, 0.0), axis=1)

    atom_ids = jnp.arange(positions.shape[0], dtype=jnp.int32)
    pair_mask = edge_mask & (atom_ids[:, None] < safe_neighbor_idx)
    pair_c6ab = d3_pre["c6ab"][sp_i, sp_j]
    e_pair = d3_pair_energy(
        pair_c6ab,
        cn[:, None],
        cn[safe_neighbor_idx],
        rij,
        r2r4[sp_i],
        r2r4[sp_j],
        d3_s6=float(d3_pre["d3_s6"]),
        d3_s8=float(d3_pre["d3_s8"]),
        d3_a1=float(d3_pre["d3_a1"]),
        d3_a2=float(d3_pre["d3_a2"]),
        d3_k3=float(d3_pre["d3_k3"]),
    )
    return jnp.sum(jnp.where(pair_mask, e_pair, 0.0)) * float(d3_pre["hartree_ev"])


def d3_pair_energy(
    pair_c6ab: Array,
    nci: Array,
    ncj: Array,
    rij: Array,
    r2r4_i: Array,
    r2r4_j: Array,
    *,
    d3_s6: float,
    d3_s8: float,
    d3_a1: float,
    d3_a2: float,
    d3_k3: float,
) -> Array:
    reference_c6 = pair_c6ab[..., 0]
    reference_cn_i = pair_c6ab[..., 1]
    reference_cn_j = pair_c6ab[..., 2]
    num_cn_references = reference_c6.shape[-2] * reference_c6.shape[-1]

    cn_distance2 = (reference_cn_i - nci[..., None, None]) ** 2 + (
        reference_cn_j - ncj[..., None, None]
    ) ** 2
    reference_weights = jnp.exp(d3_k3 * cn_distance2).reshape(
        *rij.shape,
        num_cn_references,
    )
    reference_c6 = reference_c6.reshape(*rij.shape, num_cn_references)
    weight_sum = jnp.sum(reference_weights, axis=-1)
    weighted_c6 = jnp.sum(reference_weights * reference_c6, axis=-1)
    weighted_c6 = jnp.where(weight_sum < 1.0e-5, 0.0, weighted_c6)
    c6 = weighted_c6 / jnp.maximum(weight_sum, 1.0e-5)
    c8 = 3.0 * c6 * r2r4_i * r2r4_j

    bj_radius = d3_a1 * jnp.sqrt(jnp.maximum(c8 / jnp.maximum(c6, 1.0e-30), 0.0)) + d3_a2
    bj_radius2 = bj_radius**2
    bj_radius6 = bj_radius2**3
    bj_radius8 = bj_radius6 * bj_radius2

    e6 = -d3_s6 * c6 / (rij**6 + bj_radius6)
    e8 = -d3_s8 * c8 / (rij**8 + bj_radius8)
    return e6 + e8


def radial_symmetry_functions(distance: Array, shifts: Array, eta: Array, cutoff: float) -> Array:
    cutoff_values = cosine_cutoff(distance, cutoff)
    return jnp.exp(-eta * (distance[..., None] - shifts) ** 2) * cutoff_values[..., None]


def cosine_cutoff(distance: Array, cutoff: float) -> Array:
    distance = jnp.clip(distance, 1.0e-6, cutoff)
    return 0.5 * (jnp.cos(distance * jnp.pi / cutoff) + 1.0)


def exponential_cutoff(d: Array, rc: float, exp_minus_1: float) -> Array:
    x = jnp.clip(d / rc, 0.0, 1.0 - 1.0e-6)
    return jnp.exp(-1.0 / (1.0 - x**2)) / exp_minus_1


def short_range_coulomb_dense(
    charges: Array,
    d: Array,
    safe_neighbor_idx: Array,
    edge_mask: Array,
    *,
    coulomb_rc: float,
    coulomb_factor: float,
    exp_minus_1: float,
) -> Array:
    q_ij = charges[:, None] * charges[safe_neighbor_idx]
    inv_d = 1.0 / jnp.maximum(d, 1.0e-8)
    fc = exponential_cutoff(d, coulomb_rc, exp_minus_1)
    e = coulomb_factor * fc * q_ij * inv_d
    return jnp.sum(jnp.where(edge_mask, e, 0.0))


def dsf_coulomb_dense(
    charges: Array,
    positions: Array,
    neighbor_idx: Array,
    *,
    box_vectors: Array,
    cutoff: float,
    alpha: float,
    coulomb_factor: float,
) -> Array:
    edge_vectors, safe_neighbor_idx, edge_mask = dense_neighbor_edges(
        positions,
        neighbor_idx,
        box_vectors=box_vectors,
        cutoff=float(cutoff),
    )
    d = safe_norm(edge_vectors, axis=-1)
    rc = float(cutoff)
    erfc_alpha_rc = float(math.erfc(float(alpha) * rc))
    c2 = erfc_alpha_rc / rc
    c3 = c2 / rc
    c4 = 2.0 * float(alpha) * math.exp(-((float(alpha) * rc) ** 2)) / (rc * math.sqrt(math.pi))
    j_dsf = erfc(float(alpha) * d) / jnp.maximum(d, 1.0e-8)
    j_dsf = j_dsf - c2 + (d - rc) * (c3 + c4)
    q_ij = charges[:, None] * charges[safe_neighbor_idx]
    e = coulomb_factor * q_ij * j_dsf
    return jnp.sum(jnp.where(edge_mask, e, 0.0))


def simple_coulomb_all_pairs(
    positions: Array,
    charges: Array,
    hartree_bohr: float,
) -> Array:
    num_atoms = positions.shape[0]
    delta = positions[:, None, :] - positions[None, :, :]
    distance = jnp.sqrt(jnp.maximum(jnp.sum(delta * delta, axis=-1), 1.0e-12))
    atom_ids = jnp.arange(num_atoms, dtype=jnp.int32)
    pair_mask = atom_ids[:, None] < atom_ids[None, :]
    pair_energy = (
        hartree_bohr
        * charges[:, None]
        * charges[None, :]
        / jnp.maximum(
            distance,
            1.0e-8,
        )
    )
    return jnp.sum(jnp.where(pair_mask, pair_energy, 0.0))


class AIMNet2Layer(eqx.Module):
    afv: Array
    shifts: Array
    eta: Array
    conv_a_agh: Array
    conv_q_agh: Array
    mlp0: MLP
    mlp1: MLP
    mlp2: MLP
    nfeature: int = eqx.field(static=True)
    nshifts: int = eqx.field(static=True)
    ncharge: int = eqx.field(static=True)
    mlp_last_linear: tuple[bool, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        config: dict[str, Any],
        dtype: Any = jnp.float32,
        keys: Array,
    ):
        self.nfeature = int(config["nfeature"])
        self.nshifts = int(config["nshifts"])
        self.ncharge = int(config["ncharge"])
        ncomb_v = int(config["ncomb_v"])
        self.mlp_last_linear = tuple(bool(x) for x in config["mlp_last_linear"])
        self.afv = jnp.zeros(
            (len(config["implemented_species"]), self.nfeature * self.nshifts), dtype=dtype
        )
        self.shifts = jnp.zeros((self.nshifts,), dtype=dtype)
        self.eta = jnp.zeros((), dtype=dtype)
        self.conv_a_agh = jnp.zeros(
            (self.nfeature, self.nshifts, ncomb_v),
            dtype=dtype,
        )
        self.conv_q_agh = jnp.zeros(
            (self.ncharge, self.nshifts, ncomb_v),
            dtype=dtype,
        )
        self.mlp0 = MLP(config["mlp0_sizes"], dtype=dtype, key=keys[0])
        self.mlp1 = MLP(config["mlp1_sizes"], dtype=dtype, key=keys[1])
        self.mlp2 = MLP(config["mlp2_sizes"], dtype=dtype, key=keys[2])

    def atomic_embedding_features(
        self,
        atomic_embeddings: Array,
        g_ijs: Array,
        unit_vectors: Array,
        safe_neighbor_idx: Array,
        edge_mask: Array,
    ) -> Array:
        neighbor_embeddings = atomic_embeddings[safe_neighbor_idx]
        neighbor_embeddings = jnp.where(
            edge_mask[..., None, None],
            neighbor_embeddings,
            0.0,
        )
        scalar_features = jnp.sum(neighbor_embeddings * g_ijs[:, :, None, :], axis=1)
        vector_features = jnp.einsum(
            "nkag,nkg,nkd->nagd",
            neighbor_embeddings,
            g_ijs,
            unit_vectors,
        )
        num_atoms = atomic_embeddings.shape[0]
        scalar_features = scalar_features.reshape(num_atoms, -1)
        vector_features = jnp.einsum(
            "agh,nagd->nahd",
            self.conv_a_agh,
            vector_features,
        )
        vector_features = jnp.sum(vector_features**2, axis=-1).reshape(num_atoms, -1)
        return jnp.concatenate([scalar_features, vector_features], axis=-1)

    def charge_features(
        self,
        partial_charges: Array,
        g_ijs: Array,
        unit_vectors: Array,
        safe_neighbor_idx: Array,
        edge_mask: Array,
    ) -> Array:
        neighbor_charges = partial_charges[safe_neighbor_idx]
        neighbor_charges = jnp.where(edge_mask[..., None], neighbor_charges, 0.0)
        scalar_features = jnp.einsum("nka,nkg->nag", neighbor_charges, g_ijs)
        vector_features = jnp.einsum(
            "nka,nkg,nkd->nagd",
            neighbor_charges,
            g_ijs,
            unit_vectors,
        )
        num_atoms = partial_charges.shape[0]
        scalar_features = scalar_features.reshape(num_atoms, -1)
        vector_features = jnp.einsum(
            "agh,nagd->nahd",
            self.conv_q_agh,
            vector_features,
        )
        vector_features = jnp.sum(vector_features**2, axis=-1).reshape(num_atoms, -1)
        return jnp.concatenate([scalar_features, vector_features], axis=-1)

    def neural_charge_equilibration(
        self,
        partial_charges: Array,
        charge_weights: Array,
        total_charge: Array | float = 0.0,
    ) -> Array:
        weights = charge_weights**2
        weight_sum = jnp.sum(weights, axis=0, keepdims=True) + 1.0e-6
        predicted_charge = jnp.sum(partial_charges, axis=0, keepdims=True)
        return partial_charges + (weights / weight_sum) * (total_charge - predicted_charge)

    def __call__(
        self,
        species,
        unit_vectors,
        g_ijs,
        safe_neighbor_idx,
        edge_mask,
        total_charge,
    ):
        num_atoms = species.shape[0]
        nfeature, nshifts, ncharge = self.nfeature, self.nshifts, self.ncharge

        atomic_embeddings = self.afv[species].reshape(num_atoms, nfeature, nshifts)
        embedding_flat = atomic_embeddings.reshape(num_atoms, -1)
        out0 = self.mlp0(
            jnp.concatenate(
                [
                    embedding_flat,
                    self.atomic_embedding_features(
                        atomic_embeddings,
                        g_ijs,
                        unit_vectors,
                        safe_neighbor_idx,
                        edge_mask,
                    ),
                ],
                axis=-1,
            ),
            last_linear=self.mlp_last_linear[0],
        )
        partial_charges = self.neural_charge_equilibration(
            out0[:, :ncharge],
            out0[:, ncharge : 2 * ncharge],
            total_charge,
        )
        atomic_embeddings = (embedding_flat + out0[:, 2 * ncharge :]).reshape(
            num_atoms,
            nfeature,
            nshifts,
        )

        embedding_flat = atomic_embeddings.reshape(num_atoms, -1)
        out1 = self.mlp1(
            jnp.concatenate(
                [
                    embedding_flat,
                    self.atomic_embedding_features(
                        atomic_embeddings,
                        g_ijs,
                        unit_vectors,
                        safe_neighbor_idx,
                        edge_mask,
                    ),
                    partial_charges,
                    self.charge_features(
                        partial_charges,
                        g_ijs,
                        unit_vectors,
                        safe_neighbor_idx,
                        edge_mask,
                    ),
                ],
                axis=-1,
            ),
            last_linear=self.mlp_last_linear[1],
        )
        partial_charges = self.neural_charge_equilibration(
            partial_charges + out1[:, :ncharge],
            out1[:, ncharge : 2 * ncharge],
            total_charge,
        )
        atomic_embeddings = (embedding_flat + out1[:, 2 * ncharge :]).reshape(
            num_atoms,
            nfeature,
            nshifts,
        )

        embedding_flat = atomic_embeddings.reshape(num_atoms, -1)
        aim_vectors = self.mlp2(
            jnp.concatenate(
                [
                    embedding_flat,
                    self.atomic_embedding_features(
                        atomic_embeddings,
                        g_ijs,
                        unit_vectors,
                        safe_neighbor_idx,
                        edge_mask,
                    ),
                    partial_charges,
                    self.charge_features(
                        partial_charges,
                        g_ijs,
                        unit_vectors,
                        safe_neighbor_idx,
                        edge_mask,
                    ),
                ],
                axis=-1,
            ),
            last_linear=self.mlp_last_linear[2],
        )
        return aim_vectors, partial_charges


class EnergyHead(eqx.Module):
    energy_mlp: MLP
    atomic_shifts: Array

    def __init__(
        self,
        *,
        config: dict[str, Any],
        dtype: Any = jnp.float32,
        atomic_shift_dtype: Any | None = None,
        key: Array,
    ):
        self.energy_mlp = MLP(config["energy_sizes"], dtype=dtype, key=key)
        self.atomic_shifts = jnp.zeros(
            (len(config["implemented_species"]),),
            dtype=dtype if atomic_shift_dtype is None else atomic_shift_dtype,
        )

    def __call__(self, aim_vectors: Array, species: Array) -> Array:
        atom_local_energy = self.energy_mlp(aim_vectors, last_linear=True).squeeze(-1)
        return atom_local_energy + self.atomic_shifts[species]


class AIMNet2(eqx.Module):
    bohr_a: float = eqx.field(static=True)
    coulomb_factor: float = eqx.field(static=True)
    ev_to_kjmol: float = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    coulomb_rc: float = eqx.field(static=True)
    d3_k1: float = eqx.field(static=True)
    d3_k3: float = eqx.field(static=True)
    exp_minus_1: float = eqx.field(static=True)
    hartree_bohr: float = eqx.field(static=True)
    hartree_ev: float = eqx.field(static=True)
    implemented_species: tuple[int, ...] = eqx.field(static=True)
    species_lookup: tuple[int, ...] = eqx.field(static=True)
    d3_s6: float = eqx.field(static=True)
    d3_s8: float = eqx.field(static=True)
    d3_a1: float = eqx.field(static=True)
    d3_a2: float = eqx.field(static=True)
    neighbor_cell_atom_threshold: int = eqx.field(static=True)
    neighbor_cell_capacity_multiplier: float = eqx.field(static=True)
    lr_cutoff: float = eqx.field(static=True)
    dsf_alpha: float = eqx.field(static=True)
    layer: AIMNet2Layer
    energy_head: EnergyHead
    d3_c6ab: Array
    d3_rcov: Array
    d3_r2r4: Array

    def __init__(
        self,
        *,
        config: dict[str, Any],
        dtype: Any = jnp.float32,
        atomic_shift_dtype: Any | None = None,
        key: Array = jax.random.PRNGKey(0),
    ):
        keys = jax.random.split(key, 4)
        self.bohr_a = float(config["bohr_a"])
        self.ev_to_kjmol = float(config["ev_to_kjmol"])
        self.hartree_ev = float(config["hartree_ev"])
        self.hartree_bohr = float(config["hartree_bohr"])
        self.coulomb_factor = float(config["coulomb_factor"])
        self.cutoff = float(config["cutoff"])
        self.coulomb_rc = float(config["coulomb_rc"])
        self.d3_k1 = float(config["d3_k1"])
        self.d3_k3 = float(config["d3_k3"])
        self.exp_minus_1 = float(config["exp_minus_1"])
        self.implemented_species = tuple(int(x) for x in config["implemented_species"])
        species_lookup = [-1] * (max(self.implemented_species) + 1)
        for index, atomic_number in enumerate(self.implemented_species):
            species_lookup[atomic_number] = index
        self.species_lookup = tuple(species_lookup)
        self.d3_s6 = float(config["d3_s6"])
        self.d3_s8 = float(config["d3_s8"])
        self.d3_a1 = float(config["d3_a1"])
        self.d3_a2 = float(config["d3_a2"])
        self.neighbor_cell_atom_threshold = int(config["neighbor_cell_atom_threshold"])
        self.neighbor_cell_capacity_multiplier = float(config["neighbor_cell_capacity_multiplier"])
        self.lr_cutoff = float(config["lr_cutoff"])
        self.dsf_alpha = float(config["dsf_alpha"])
        self.layer = AIMNet2Layer(
            config=config,
            dtype=dtype,
            keys=keys[:3],
        )
        self.energy_head = EnergyHead(
            config=config,
            dtype=dtype,
            atomic_shift_dtype=atomic_shift_dtype,
            key=keys[3],
        )
        self.d3_c6ab = jnp.zeros(tuple(int(v) for v in config["d3_c6ab_shape"]), dtype=dtype)
        self.d3_rcov = jnp.zeros(tuple(int(v) for v in config["d3_rcov_shape"]), dtype=dtype)
        self.d3_r2r4 = jnp.zeros(tuple(int(v) for v in config["d3_r2r4_shape"]), dtype=dtype)

    def validate_species(self, species) -> None:
        species = np.asarray(jax.device_get(species), dtype=np.int64).reshape(-1)
        unsupported = sorted(set(species.tolist()) - set(self.implemented_species))
        if unsupported:
            supported = ", ".join(str(z) for z in self.implemented_species)
            raise ValueError(
                f"AIMNet2 does not support atomic numbers {unsupported}. "
                f"Supported atomic numbers: {supported}."
            )

    def prepare_d3_data(self, atomic_numbers):
        unique_atomic_numbers = np.unique(atomic_numbers)
        atomic_number_to_index = np.zeros(
            int(unique_atomic_numbers.max()) + 1,
            dtype=np.int32,
        )
        for species_index, atomic_number in enumerate(unique_atomic_numbers):
            atomic_number_to_index[int(atomic_number)] = species_index
        species_indices = atomic_number_to_index[atomic_numbers]
        unique_atomic_numbers_array = jnp.asarray(unique_atomic_numbers, dtype=jnp.int32)
        unique_species = jnp.asarray(self.species_lookup, dtype=jnp.int32)[
            unique_atomic_numbers_array
        ]
        return {
            "c6ab": self.d3_c6ab[
                unique_species[:, None],
                unique_species[None, :],
            ],
            "rcov": self.d3_rcov[unique_species],
            "r2r4": self.d3_r2r4[unique_species],
            "species_idx": jnp.asarray(species_indices, dtype=jnp.int32),
            "d3_s6": float(self.d3_s6),
            "d3_s8": float(self.d3_s8),
            "d3_a1": float(self.d3_a1),
            "d3_a2": float(self.d3_a2),
            "d3_k1": float(self.d3_k1),
            "d3_k3": float(self.d3_k3),
            "bohr_a": float(self.bohr_a),
            "hartree_ev": float(self.hartree_ev),
        }

    def coulomb_energy(
        self,
        partial_charges,
        positions,
        r_ij,
        safe_neighbor_idx,
        edge_mask,
        lr_neighbor_idx,
        box_vectors,
    ):
        partial_charges = partial_charges.squeeze(-1)
        local_coulomb = short_range_coulomb_dense(
            partial_charges,
            r_ij,
            safe_neighbor_idx,
            edge_mask,
            coulomb_rc=self.coulomb_rc,
            coulomb_factor=self.coulomb_factor,
            exp_minus_1=self.exp_minus_1,
        )
        if box_vectors is None:
            total_coulomb = simple_coulomb_all_pairs(
                positions,
                partial_charges,
                hartree_bohr=self.hartree_bohr,
            )
        else:
            total_coulomb = dsf_coulomb_dense(
                partial_charges,
                positions,
                lr_neighbor_idx,
                box_vectors=box_vectors,
                cutoff=float(self.lr_cutoff),
                alpha=float(self.dsf_alpha),
                coulomb_factor=self.coulomb_factor,
            )
        return total_coulomb - local_coulomb

    def local_node_energies_and_charges(
        self,
        positions: Array,
        species: Array,
        *,
        neighbor_idx: Array,
        total_charge: Array | float = 0.0,
        box_vectors: Array | None = None,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Return local node energies plus intermediates needed by global terms."""

        atomic_numbers = jnp.asarray(species, dtype=jnp.int32)
        species = jnp.asarray(self.species_lookup, dtype=jnp.int32)[atomic_numbers]
        edge_vectors, safe_neighbor_idx, edge_mask = dense_neighbor_edges(
            positions,
            neighbor_idx,
            box_vectors=box_vectors,
            cutoff=float(self.cutoff),
        )
        r_ij = safe_norm(edge_vectors, axis=-1)
        unit_vectors = edge_vectors / jnp.maximum(r_ij[..., None], 1.0e-8)
        g_ijs = radial_symmetry_functions(
            r_ij,
            self.layer.shifts,
            self.layer.eta,
            self.cutoff,
        )
        g_ijs = jnp.where(edge_mask[..., None], g_ijs, 0.0)
        aim_vectors, partial_charges = self.layer(
            species,
            unit_vectors,
            g_ijs,
            safe_neighbor_idx,
            edge_mask,
            total_charge,
        )

        node_energies = self.energy_head(aim_vectors, species)
        return node_energies, partial_charges, r_ij, safe_neighbor_idx, edge_mask

    def __call__(
        self,
        positions: Array,
        species: Array,
        *,
        d3_data: dict[str, Array],
        box_vectors: Array | None = None,
        neighbors=None,
        neighbor_idx: Array | None = None,
        lr_neighbors=None,
        lr_neighbor_idx: Array | None = None,
        periodic: bool | None = False,
        total_charge: Array | float = 0.0,
    ) -> Array:
        periodic = bool(periodic)
        if neighbor_idx is None:
            neighbors = get_neighbors(
                positions,
                box_vectors if periodic else None,
                cutoff=float(self.cutoff),
                cell_atom_threshold=int(self.neighbor_cell_atom_threshold),
                cell_capacity_multiplier=float(self.neighbor_cell_capacity_multiplier),
                neighbors=neighbors,
                periodic=periodic,
            )
            neighbor_idx = neighbors.idx
        if lr_neighbor_idx is None:
            lr_neighbors = get_neighbors(
                positions,
                box_vectors if periodic else None,
                cutoff=float(self.lr_cutoff),
                cell_atom_threshold=int(self.neighbor_cell_atom_threshold),
                cell_capacity_multiplier=float(self.neighbor_cell_capacity_multiplier),
                neighbors=lr_neighbors,
                periodic=periodic,
            )
            lr_neighbor_idx = lr_neighbors.idx

        box_vectors = box_vectors if periodic else None
        (
            node_energies,
            partial_charges,
            r_ij,
            safe_neighbor_idx,
            edge_mask,
        ) = self.local_node_energies_and_charges(
            positions,
            species,
            neighbor_idx=neighbor_idx,
            total_charge=total_charge,
            box_vectors=box_vectors,
        )
        local_energy = jnp.sum(node_energies)
        coulomb_energy = self.coulomb_energy(
            partial_charges,
            positions,
            r_ij,
            safe_neighbor_idx,
            edge_mask,
            lr_neighbor_idx,
            box_vectors,
        )
        dispersion_energy = d3bj_energy_neighbors(
            positions,
            d3_data,
            lr_neighbor_idx,
            box_vectors=box_vectors,
            cutoff=float(self.lr_cutoff),
        )
        return local_energy + coulomb_energy + dispersion_energy


def load_model(
    model: str | PathLike = "aimnet2-jax",
    *,
    dtype=jnp.float32,
    atomic_numbers=None,
    neighbor_cell_atom_threshold: int | None = None,
    neighbor_cell_capacity_multiplier: float | None = None,
) -> AIMNet2:
    path = (
        AIMNET2_MODEL_PATHS[model]
        if isinstance(model, str) and model in AIMNET2_MODEL_PATHS
        else Path(model)
    )
    use_float64 = jnp.dtype(dtype) == jnp.dtype(jnp.float64)
    with jax.enable_x64(use_float64), path.open("rb") as handle:
        config = json.loads(handle.readline().decode("utf-8"))
        if neighbor_cell_atom_threshold is not None:
            config["neighbor_cell_atom_threshold"] = int(neighbor_cell_atom_threshold)
        if neighbor_cell_capacity_multiplier is not None:
            config["neighbor_cell_capacity_multiplier"] = float(neighbor_cell_capacity_multiplier)
        model_template = AIMNet2(
            config=config,
            dtype=jnp.float32,
            atomic_shift_dtype=jnp.float64 if use_float64 else jnp.float32,
        )
        loaded_model = eqx.tree_deserialise_leaves(handle, model_template)
        loaded_model = jax.tree_util.tree_map(
            lambda value: value.astype(dtype)
            if eqx.is_array(value) and jnp.issubdtype(value.dtype, jnp.floating)
            else value,
            loaded_model,
        )
        if atomic_numbers is not None:
            loaded_model.validate_species(atomic_numbers)
        return loaded_model
