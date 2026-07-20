# Credit to https://github.com/torchmd/torchmd-net

import json
import pickle
from os import PathLike
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax_md import partition, space

jax.config.update("jax_default_matmul_precision", "highest")

ACEFF_MODEL_PATHS = {
    "aceff-jax-1.1": Path(__file__).resolve().with_name("aceff_v1.1.eqx"),
    "aceff-jax-2.0": Path(__file__).resolve().with_name("aceff_v2.0.eqx"),
}
ACEFF_MODEL_NAMES = tuple(ACEFF_MODEL_PATHS)


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
    use_cell_list = periodic and num_atoms >= int(cell_atom_threshold)
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


def dense_neighbor_edges(positions, neighbor_idx, *, box_vectors=None, include_self=False):
    num_atoms = positions.shape[0]
    atom_ids = jnp.arange(num_atoms, dtype=jnp.int32)
    neighbor_idx = jnp.asarray(
        neighbor_idx.idx if hasattr(neighbor_idx, "idx") else neighbor_idx,
        dtype=jnp.int32,
    )
    edge_mask = (neighbor_idx >= 0) & (neighbor_idx < num_atoms)
    edge_mask = edge_mask & (neighbor_idx != atom_ids[:, None])
    safe_neighbor_idx = jnp.where(edge_mask, neighbor_idx, atom_ids[:, None])

    if include_self:
        safe_neighbor_idx = jnp.concatenate([atom_ids[:, None], safe_neighbor_idx], axis=1)
        edge_mask = jnp.concatenate(
            [jnp.ones((num_atoms, 1), dtype=bool), edge_mask],
            axis=1,
        )

    neighbor_positions = positions[safe_neighbor_idx]
    if box_vectors is None:
        edge_vectors = positions[:, None, :] - neighbor_positions
    else:
        displacement, _ = space.periodic_general(
            jnp.swapaxes(jnp.asarray(box_vectors, dtype=positions.dtype), -1, -2),
            fractional_coordinates=False,
        )
        edge_vectors = -space.map_neighbor(displacement)(positions, neighbor_positions)

    edge_vectors = jnp.where(edge_mask[..., None], edge_vectors, 0.0)
    return edge_vectors, safe_neighbor_idx, edge_mask


def unique_pairs(num_atoms: int):
    pair_src, pair_dst = np.triu_indices(int(num_atoms), k=1)
    return (
        jnp.asarray(pair_src, dtype=jnp.int32),
        jnp.asarray(pair_dst, dtype=jnp.int32),
    )


def cosine_cutoff(d, cutoff, cutoff_lower=0.0):
    if cutoff_lower > 0:
        x = 2.0 * (d - cutoff_lower) / (cutoff - cutoff_lower) + 1.0
        c = 0.5 * (jnp.cos(jnp.pi * x) + 1.0)
        return jnp.where((d < cutoff) & (d > cutoff_lower), c, 0.0)
    c = 0.5 * (jnp.cos(d * jnp.pi / cutoff) + 1.0)
    return jnp.where(d < cutoff, c, 0.0)


def decompose_tensor(X):
    """Decompose into scalar, antisymmetric, and symmetric components."""
    antisymmetric = 0.5 * (X - jnp.swapaxes(X, 1, 2))
    symmetric_full = X - antisymmetric
    scalar = jnp.diagonal(X, axis1=1, axis2=2).mean(axis=-1)
    symmetric = (
        symmetric_full
        - scalar[:, None, None, :] * jnp.eye(3, dtype=scalar.dtype)[None, :, :, None]
    )
    return scalar, antisymmetric, symmetric


def vector_to_skewtensor(vec):
    """[N, 3, F] -> [N, 3, 3, F] skew-symmetric."""
    num_atoms, _, num_features = vec.shape
    zero = jnp.zeros((num_atoms, num_features), dtype=vec.dtype)
    tensor = jnp.stack(
        [
            zero,
            -vec[:, 2, :],
            vec[:, 1, :],
            vec[:, 2, :],
            zero,
            -vec[:, 0, :],
            -vec[:, 1, :],
            vec[:, 0, :],
            zero,
        ],
        axis=1,
    )
    return tensor.reshape(num_atoms, 3, 3, num_features)


def skewtensor_to_vector(A):
    """[N, 3, 3, F] -> [N, 3, F]."""
    A_flat = A.reshape(A.shape[0], 9, A.shape[3])
    return 0.5 * jnp.stack(
        [
            A_flat[:, 7] - A_flat[:, 5],
            A_flat[:, 2] - A_flat[:, 6],
            A_flat[:, 3] - A_flat[:, 1],
        ],
        axis=1,
    )


def outer_to_symtensor(T):
    """Symmetrize and remove trace. [N, 3, 3, F] -> [N, 3, 3, F]."""
    symmetric = 0.5 * (T + jnp.swapaxes(T, 1, 2))
    scalar = jnp.diagonal(T, axis1=1, axis2=2).mean(axis=-1)
    return symmetric - scalar[:, None, None, :] * jnp.eye(3, dtype=scalar.dtype)[None, :, :, None]


def tensor_matmul_o3(Y, msg):
    """O(3)-equivariant contraction: Y*msg + msg*Y."""
    Yp = jnp.transpose(Y, (0, 3, 1, 2))
    Mp = jnp.transpose(msg, (0, 3, 1, 2))
    return jnp.transpose(Mp @ Yp, (0, 2, 3, 1)) + jnp.transpose(Yp @ Mp, (0, 2, 3, 1))


def tensor_matmul_so3(Y, msg):
    """SO(3)-equivariant contraction: Y*msg."""
    Yp = jnp.transpose(Y, (0, 3, 1, 2))
    Mp = jnp.transpose(msg, (0, 3, 1, 2))
    return jnp.transpose(Yp @ Mp, (0, 2, 3, 1))


class Linear(eqx.Module):
    kernel: Any
    bias: Any

    def __init__(self, kernel, bias=None):
        self.kernel = kernel
        self.bias = bias

    def __call__(self, x):
        out = x @ self.kernel
        if self.bias is not None:
            out = out + self.bias
        return out


class LayerNorm(eqx.Module):
    weight: Any
    bias: Any
    eps: float = eqx.field(static=True)

    def __init__(self, weight, bias, *, eps: float = 1.0e-5):
        self.weight = weight
        self.bias = bias
        self.eps = float(eps)

    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        return self.weight * (x - mean) * jax.lax.rsqrt(variance + self.eps) + self.bias


def safe_norm(x, *, axis=-1, keepdims: bool = False, eps: float = 1.0e-24):
    return jnp.sqrt(jnp.maximum(jnp.sum(x * x, axis=axis, keepdims=keepdims), eps))


class TensorEmbedding(eqx.Module):
    weights: Any
    cutoff: float = eqx.field(static=True)
    cutoff_lower: float = eqx.field(static=True)

    def __init__(self, weights, *, cutoff: float, cutoff_lower: float):
        self.weights = weights
        self.cutoff = float(cutoff)
        self.cutoff_lower = float(cutoff_lower)

    def __call__(
        self,
        species,
        safe_neighbor_idx,
        distances,
        unit_vectors,
        radial_features,
        *,
        edge_mask,
    ):
        weights = self.weights
        num_features = weights["emb"].shape[1]

        Zi = jnp.broadcast_to(
            weights["emb"][species][:, None, :],
            (*safe_neighbor_idx.shape, num_features),
        )
        Zj = weights["emb"][species[safe_neighbor_idx]]
        Zij = weights["emb2"](
            jnp.concatenate([Zi, Zj], axis=-1),
        )

        distance_projection_1 = weights["distance_proj1"](radial_features)
        distance_projection_2 = weights["distance_proj2"](radial_features)
        distance_projection_3 = weights["distance_proj3"](radial_features)

        cutoff_values = cosine_cutoff(distances, self.cutoff, self.cutoff_lower)
        cutoff_values = cutoff_values * edge_mask
        species_pair_features = cutoff_values[..., None] * Zij

        edge_features = species_pair_features[..., None, :] * jnp.stack(
            [distance_projection_1, distance_projection_2, distance_projection_3],
            axis=-2,
        )

        scalar = jnp.sum(edge_features[..., 0, :], axis=1)

        antisymmetric_vectors = jnp.einsum(
            "nef,nei->nif",
            edge_features[..., 1, :],
            unit_vectors,
        )

        outer = unit_vectors[..., :, None] * unit_vectors[..., None, :]
        symmetric = jnp.einsum(
            "nef,neij->nijf",
            edge_features[..., 2, :],
            outer,
        )

        antisymmetric = vector_to_skewtensor(antisymmetric_vectors)
        symmetric = outer_to_symtensor(symmetric)
        identity = jnp.eye(3, dtype=scalar.dtype)[None, :, :, None]
        tensor_features = scalar[:, None, None, :] * identity + antisymmetric + symmetric

        norm = jnp.sum(tensor_features**2, axis=(1, 2))
        norm = weights["init_norm"](norm)
        for layer in weights["linears_scalar"]:
            norm = jax.nn.silu(layer(norm))
        norm = norm.reshape(-1, 3, num_features)

        scalar_norm = norm[:, 0, :]
        antisymmetric_norm = norm[:, 1, :][:, None, None, :]
        symmetric_norm = norm[:, 2, :][:, None, None, :]

        scalar = weights["linears_tensor"][0](scalar) * scalar_norm
        antisymmetric = weights["linears_tensor"][1](antisymmetric) * antisymmetric_norm
        symmetric = weights["linears_tensor"][2](symmetric) * symmetric_norm

        return scalar[:, None, None, :] * identity + antisymmetric + symmetric


class ChargePredictionHead(eqx.Module):
    weights: Any

    def __init__(self, weights):
        self.weights = weights

    def neural_charge_equilibration(
        self,
        partial_charges,
        charge_weights,
        total_charge=0.0,
    ):
        weights = charge_weights**2
        weight_sum = jnp.sum(weights, axis=0, keepdims=True) + 1.0e-6
        predicted_charge = jnp.sum(partial_charges, axis=0, keepdims=True)
        return partial_charges + (weights / weight_sum) * (total_charge - predicted_charge)

    def __call__(self, tensor_features, total_charge=0.0):
        weights = self.weights
        scalar, antisymmetric, symmetric = decompose_tensor(tensor_features)
        charge_features = jnp.concatenate(
            [
                scalar,
                jnp.sum(antisymmetric**2, axis=(1, 2)),
                jnp.sum(symmetric**2, axis=(1, 2)),
            ],
            axis=-1,
        )

        charge_features = weights["q_norm"](charge_features)
        for i, layer in enumerate(weights["q_mlp"]):
            charge_features = layer(charge_features)
            if i < len(weights["q_mlp"]) - 1:
                charge_features = jax.nn.silu(charge_features)

        ncharge = charge_features.shape[-1] // 2
        partial_charges = charge_features[:, :ncharge]
        charge_weights = charge_features[:, ncharge:]
        return self.neural_charge_equilibration(
            partial_charges,
            charge_weights,
            total_charge,
        )


class AceFFLayer(eqx.Module):
    weights: Any
    cutoff: float = eqx.field(static=True)
    cutoff_lower: float = eqx.field(static=True)
    group: str = eqx.field(static=True)
    edge_charge_features: bool = eqx.field(static=True)
    total_charge_interaction_scale: bool = eqx.field(static=True)

    def __init__(
        self,
        weights,
        *,
        cutoff: float,
        cutoff_lower: float,
        group: str,
        edge_charge_features: bool,
        total_charge_interaction_scale: bool,
    ):
        self.weights = weights
        self.cutoff = float(cutoff)
        self.cutoff_lower = float(cutoff_lower)
        self.group = str(group)
        self.edge_charge_features = bool(edge_charge_features)
        self.total_charge_interaction_scale = bool(total_charge_interaction_scale)

    def __call__(
        self,
        tensor_features,
        partial_charges,
        total_charge,
        safe_neighbor_idx,
        distances,
        radial_features,
        *,
        edge_mask,
    ):
        weights = self.weights
        num_features = tensor_features.shape[3]

        cutoff_values = cosine_cutoff(distances, self.cutoff, self.cutoff_lower)
        cutoff_values = cutoff_values * edge_mask

        if self.edge_charge_features:
            source_charges = jnp.broadcast_to(
                partial_charges[:, None, :],
                (*safe_neighbor_idx.shape, partial_charges.shape[-1]),
            )
            neighbor_charges = partial_charges[safe_neighbor_idx]
            edge_features = jnp.concatenate(
                [radial_features, source_charges, neighbor_charges],
                axis=-1,
            )
        else:
            edge_features = radial_features

        for layer in weights["linears_scalar"]:
            edge_features = jax.nn.silu(layer(edge_features))
        edge_features = (edge_features * cutoff_values[..., None]).reshape(
            *safe_neighbor_idx.shape,
            3,
            num_features,
        )

        tensor_features = (
            tensor_features / (jnp.sum(tensor_features**2, axis=(1, 2)) + 1)[:, None, None, :]
        )

        scalar, antisymmetric, symmetric = decompose_tensor(tensor_features)
        scalar = weights["linears_tensor"][0](scalar)
        antisymmetric = weights["linears_tensor"][1](antisymmetric)
        symmetric = weights["linears_tensor"][2](symmetric)
        identity = jnp.eye(3, dtype=scalar.dtype)[None, :, :, None]
        projected_features = scalar[:, None, None, :] * identity + antisymmetric + symmetric

        antisymmetric_vectors = skewtensor_to_vector(antisymmetric)

        scalar_weights = edge_features[..., 0, :]
        scalar_messages = scalar_weights * scalar[safe_neighbor_idx]
        scalar_message = jnp.sum(scalar_messages, axis=1)

        antisymmetric_message_vectors = jnp.einsum(
            "nef,neif->nif",
            edge_features[..., 1, :],
            antisymmetric_vectors[safe_neighbor_idx],
        )

        symmetric_message = jnp.einsum(
            "nef,neijf->nijf",
            edge_features[..., 2, :],
            symmetric[safe_neighbor_idx],
        )

        antisymmetric_message = vector_to_skewtensor(antisymmetric_message_vectors)
        messages = (
            scalar_message[:, None, None, :] * identity + antisymmetric_message + symmetric_message
        )

        charge_factor = 1.0
        if self.total_charge_interaction_scale:
            charge_factor = 1.0 + 0.1 * jnp.asarray(
                total_charge,
                dtype=tensor_features.dtype,
            )

        if self.group == "O(3)":
            updates = charge_factor * tensor_matmul_o3(projected_features, messages)
        else:
            updates = 2 * tensor_matmul_so3(projected_features, messages)

        scalar_update, antisymmetric_update, symmetric_update = decompose_tensor(updates)

        update_norm = jnp.sum(updates**2, axis=(1, 2)) + 1
        scalar_update = scalar_update / update_norm
        antisymmetric_update = antisymmetric_update / update_norm[:, None, None, :]
        symmetric_update = symmetric_update / update_norm[:, None, None, :]

        scalar_update = weights["linears_tensor"][3](scalar_update)
        antisymmetric_update = weights["linears_tensor"][4](antisymmetric_update)
        symmetric_update = weights["linears_tensor"][5](symmetric_update)
        delta_features = (
            scalar_update[:, None, None, :] * identity + antisymmetric_update + symmetric_update
        )

        return (
            tensor_features
            + delta_features
            + charge_factor * tensor_matmul_so3(delta_features, delta_features)
        )


class LocalEnergyHead(eqx.Module):
    out_norm: Any
    linear: Any
    output_network: Any

    def __init__(self, weights):
        self.out_norm = weights["out_norm"]
        self.linear = weights["linear"]
        self.output_network = weights["output_network"]

    def __call__(self, tensor_features):
        _, antisymmetric, symmetric = decompose_tensor(tensor_features)
        trace = jnp.diagonal(tensor_features, axis1=1, axis2=2).sum(axis=-1)
        warp_one_third = jnp.asarray(
            # This comes from upstream using Warp kernels that round
            # 1/3 to float32 before casting to float64
            float(np.float32(1.0 / 3.0)),
            dtype=trace.dtype,
        )
        scalar_norm = warp_one_third * trace * trace
        energy_features = jnp.concatenate(
            [
                scalar_norm,
                jnp.sum(antisymmetric**2, axis=(1, 2)),
                jnp.sum(symmetric**2, axis=(1, 2)),
            ],
            axis=-1,
        )
        energy_features = self.out_norm(energy_features)
        energy_features = jax.nn.silu(self.linear(energy_features))
        for i, layer in enumerate(self.output_network):
            energy_features = layer(energy_features)
            if i < len(self.output_network) - 1:
                energy_features = jax.nn.silu(energy_features)
        return energy_features.squeeze(-1)


class CoulombHead(eqx.Module):
    qweights: Any
    coulomb_factor: float = eqx.field(static=True)
    coulomb_damp_cutoff: float = eqx.field(static=True)
    coulomb_cutoff: float | None = eqx.field(static=True)
    coulomb_epsilon_solvent: float = eqx.field(static=True)
    exp_minus_1: float = eqx.field(static=True)

    def __init__(
        self,
        qweights,
        *,
        coulomb_factor: float,
        coulomb_damp_cutoff: float,
        coulomb_cutoff: float | None,
        coulomb_epsilon_solvent: float,
        exp_minus_1: float,
    ):
        self.qweights = qweights
        self.coulomb_factor = float(coulomb_factor)
        self.coulomb_damp_cutoff = float(coulomb_damp_cutoff)
        self.coulomb_cutoff = None if coulomb_cutoff is None else float(coulomb_cutoff)
        self.coulomb_epsilon_solvent = float(coulomb_epsilon_solvent)
        self.exp_minus_1 = float(exp_minus_1)

    def __call__(
        self,
        positions,
        partial_charges,
        *,
        box_vectors=None,
        neighbor_idx=None,
    ):
        partial_charges = jnp.concatenate(partial_charges, axis=-1)
        pair_mask = None
        if neighbor_idx is None:
            pair_src, pair_dst = unique_pairs(int(positions.shape[0]))
            if box_vectors is None:
                pair_vectors = positions[pair_src] - positions[pair_dst]
            else:
                displacement, _ = space.periodic_general(
                    jnp.swapaxes(jnp.asarray(box_vectors, dtype=positions.dtype), -1, -2),
                    fractional_coordinates=False,
                )
                pair_vectors = jax.vmap(displacement)(positions[pair_src], positions[pair_dst])
            charge_products = partial_charges[pair_src] * partial_charges[pair_dst]
        else:
            pair_vectors, safe_neighbor_idx, edge_mask = dense_neighbor_edges(
                positions,
                neighbor_idx,
                box_vectors=box_vectors,
            )
            atom_ids = jnp.arange(positions.shape[0], dtype=jnp.int32)
            pair_mask = edge_mask & (atom_ids[:, None] < safe_neighbor_idx)
            charge_products = partial_charges[:, None, :] * partial_charges[safe_neighbor_idx]

        distances = safe_norm(pair_vectors, axis=-1)
        damping_x = jnp.clip(
            distances / self.coulomb_damp_cutoff,
            0.0,
            1.0 - 1e-6,
        )
        damping = jnp.exp(-1.0 / (1.0 - damping_x**2)) / self.exp_minus_1
        cutoff_values = 1.0 - damping
        weighted_charge_products = (charge_products * self.qweights[None, :]).sum(
            axis=-1
        ) / self.qweights.sum()
        if self.coulomb_cutoff is None:
            pair_energies = cutoff_values * weighted_charge_products / distances
        else:
            cutoff = self.coulomb_cutoff
            epsilon = self.coulomb_epsilon_solvent
            k_rf = (1.0 / cutoff**3) * (epsilon - 1.0) / (2.0 * epsilon + 1.0)
            c_rf = (1.0 / cutoff) * (3.0 * epsilon) / (2.0 * epsilon + 1.0)
            pair_energies = (
                cutoff_values
                * weighted_charge_products
                * (1.0 / distances + k_rf * distances**2 - c_rf)
            )
            pair_energies = jnp.where(distances < cutoff, pair_energies, 0.0)
        if pair_mask is not None:
            pair_energies = jnp.where(pair_mask, pair_energies, 0.0)
        return self.coulomb_factor * pair_energies.sum()


class AceFF(eqx.Module):
    rbf_betas: Any
    rbf_means: Any
    tensor_embedding: TensorEmbedding
    charge_predictor_0: ChargePredictionHead | None
    layers: tuple[AceFFLayer, ...]
    charge_predictors_by_layer: tuple[ChargePredictionHead, ...]
    local_energy_head: LocalEnergyHead
    coulomb_head: CoulombHead | None
    cutoff: float = eqx.field(static=True)
    ev_to_kjmol: float = eqx.field(static=True)
    cutoff_lower: float = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    neighbor_cell_atom_threshold: int = eqx.field(static=True)
    neighbor_cell_capacity_multiplier: float = eqx.field(static=True)

    def __init__(self, params, config):
        self.cutoff = float(config["cutoff"])
        self.ev_to_kjmol = float(config["ev_to_kjmol"])
        self.cutoff_lower = float(config["cutoff_lower"])
        self.alpha = float(config["alpha"])
        group = str(config["group"])
        charge_predictors = bool(config["charge_predictors"])
        edge_charge_features = bool(config["edge_charge_features"])
        total_charge_interaction_scale = bool(config["total_charge_interaction_scale"])
        coulomb_energy = bool(config["coulomb_energy"])
        coulomb_factor = float(config["coulomb_factor"])
        coulomb_damp_cutoff = float(config["coulomb_damp_cutoff"])
        coulomb_cutoff = config["coulomb_cutoff"]
        coulomb_cutoff = None if coulomb_cutoff is None else float(coulomb_cutoff)
        coulomb_epsilon_solvent = float(config["coulomb_epsilon_solvent"])
        exp_minus_1 = float(config["exp_minus_1"])
        self.neighbor_cell_atom_threshold = int(config["neighbor_cell_atom_threshold"])
        self.neighbor_cell_capacity_multiplier = float(config["neighbor_cell_capacity_multiplier"])
        self.rbf_betas = params["rbf_betas"]
        self.rbf_means = params["rbf_means"]
        self.tensor_embedding = TensorEmbedding(
            params["tensor_embedding"],
            cutoff=self.cutoff,
            cutoff_lower=self.cutoff_lower,
        )
        self.charge_predictor_0 = (
            ChargePredictionHead(params["charge_predict_0"]) if charge_predictors else None
        )
        self.layers = tuple(
            AceFFLayer(
                layer,
                cutoff=self.cutoff,
                cutoff_lower=self.cutoff_lower,
                group=group,
                edge_charge_features=edge_charge_features,
                total_charge_interaction_scale=total_charge_interaction_scale,
            )
            for layer in params["layers"]
        )
        self.charge_predictors_by_layer = (
            tuple(ChargePredictionHead(weights) for weights in params["charge_predicts"])
            if charge_predictors
            else ()
        )
        self.local_energy_head = LocalEnergyHead(params)
        self.coulomb_head = (
            CoulombHead(
                params["qweights"],
                coulomb_factor=coulomb_factor,
                coulomb_damp_cutoff=coulomb_damp_cutoff,
                coulomb_cutoff=coulomb_cutoff,
                coulomb_epsilon_solvent=coulomb_epsilon_solvent,
                exp_minus_1=exp_minus_1,
            )
            if coulomb_energy
            else None
        )

    def edge_features(
        self,
        edge_vectors,
        safe_neighbor_idx,
        edge_mask,
    ):
        atom_ids = jnp.arange(safe_neighbor_idx.shape[0], dtype=jnp.int32)
        is_self = safe_neighbor_idx == atom_ids[:, None]
        distances = jnp.where(
            is_self,
            0.0,
            safe_norm(edge_vectors, axis=-1, eps=1.0e-30),
        )
        safe_denom = jnp.where(
            is_self[..., None],
            1.0,
            jnp.maximum(distances[..., None], 1e-8),
        )
        unit_vectors = edge_vectors / safe_denom
        edge_mask = jnp.asarray(edge_mask, dtype=edge_vectors.dtype)
        distances_expanded = distances[..., None]
        cutoff_values = cosine_cutoff(distances_expanded, self.cutoff, 0.0)
        radial_features = (
            cutoff_values
            * jnp.exp(
                -self.rbf_betas
                * (
                    jnp.exp(self.alpha * (-distances_expanded + self.cutoff_lower))
                    - self.rbf_means
                )
                ** 2
            )
            * edge_mask[..., None]
        )
        return distances, unit_vectors, radial_features, edge_mask

    def local_node_energies_and_charges(
        self,
        species,
        safe_neighbor_idx,
        edge_vectors,
        edge_mask,
        total_charge,
    ):
        distances, unit_vectors, radial_features, edge_mask = self.edge_features(
            edge_vectors,
            safe_neighbor_idx,
            edge_mask,
        )
        tensor_features = self.tensor_embedding(
            species,
            safe_neighbor_idx,
            distances,
            unit_vectors,
            radial_features,
            edge_mask=edge_mask,
        )

        partial_charges = None
        charge_history = []
        if self.charge_predictor_0 is not None:
            partial_charges = self.charge_predictor_0(
                tensor_features,
                total_charge,
            )
            charge_history.append(partial_charges)

        for layer_index, layer in enumerate(self.layers):
            tensor_features = layer(
                tensor_features,
                partial_charges,
                total_charge,
                safe_neighbor_idx,
                distances,
                radial_features,
                edge_mask=edge_mask,
            )
            if self.charge_predictor_0 is not None:
                partial_charges = self.charge_predictors_by_layer[layer_index](
                    tensor_features,
                    total_charge,
                )
                charge_history.append(partial_charges)

        return self.local_energy_head(tensor_features), charge_history

    def __call__(
        self,
        positions,
        species,
        *,
        box_vectors=None,
        neighbors=None,
        neighbor_idx=None,
        coulomb_neighbors=None,
        coulomb_neighbor_idx=None,
        periodic=False,
        total_charge=0.0,
    ):
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

        edge_vectors, safe_neighbor_idx, edge_mask = dense_neighbor_edges(
            positions,
            neighbor_idx,
            box_vectors=box_vectors if periodic else None,
            include_self=True,
        )
        node_energies, partial_charges = self.local_node_energies_and_charges(
            species,
            safe_neighbor_idx,
            edge_vectors,
            edge_mask,
            total_charge,
        )
        local_energy = jnp.sum(node_energies)
        if self.coulomb_head is None:
            return local_energy
        if self.coulomb_head.coulomb_cutoff is not None and coulomb_neighbor_idx is None:
            coulomb_neighbors = get_neighbors(
                positions,
                box_vectors if periodic else None,
                cutoff=float(self.coulomb_head.coulomb_cutoff),
                cell_atom_threshold=int(self.neighbor_cell_atom_threshold),
                cell_capacity_multiplier=float(self.neighbor_cell_capacity_multiplier),
                neighbors=coulomb_neighbors,
                periodic=periodic,
            )
            coulomb_neighbor_idx = coulomb_neighbors.idx
        coulomb_energy = self.coulomb_head(
            positions,
            partial_charges,
            box_vectors=box_vectors if periodic else None,
            neighbor_idx=coulomb_neighbor_idx,
        )
        return local_energy + coulomb_energy


def load_model(
    model: str | PathLike = "aceff-jax-2.0",
    *,
    dtype=jnp.float32,
    neighbor_cell_atom_threshold: int | None = None,
    neighbor_cell_capacity_multiplier: float | None = None,
) -> AceFF:
    path = (
        ACEFF_MODEL_PATHS[model]
        if isinstance(model, str) and model in ACEFF_MODEL_PATHS
        else Path(model)
    )

    with jax.enable_x64(jnp.dtype(dtype) == jnp.dtype(jnp.float64)), path.open("rb") as handle:
        config = json.loads(handle.readline().decode("utf-8"))
        if neighbor_cell_atom_threshold is not None:
            config["neighbor_cell_atom_threshold"] = int(neighbor_cell_atom_threshold)
        if neighbor_cell_capacity_multiplier is not None:
            config["neighbor_cell_capacity_multiplier"] = float(neighbor_cell_capacity_multiplier)
        checkpoint_weights = pickle.load(handle)
        loaded_model = AceFF(checkpoint_weights, config)
        return jax.tree_util.tree_map(
            lambda value: value.astype(dtype)
            if eqx.is_array(value) and jnp.issubdtype(value.dtype, jnp.floating)
            else value,
            loaded_model,
        )
