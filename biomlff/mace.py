# Credit to https://github.com/atomicarchitects/nequix/blob/da0fb241f417dad1afafa4e723ad867667ee7445/nequix/model.py#L385-L386
# for the basis functions
# Credit to https://github.com/abhijeetgangan/mace-eqx/blob/main/mace_eqx/mace.py
# for the overall design

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

MACE_OFF_MODEL_PATHS = {
    "mace-jax-off-s-23": Path(__file__).resolve().parent / "mace-off-s(23).eqx",
    "mace-jax-off-m-24": Path(__file__).resolve().parent / "mace-off-m(24).eqx",
}
MACELES_MODEL_PATHS = {
    "maceles-jax-off-small": Path(__file__).resolve().parent / "maceles-off-small.eqx",
}
MACE_MODEL_PATHS = {**MACE_OFF_MODEL_PATHS, **MACELES_MODEL_PATHS}


LatentEwaldGrid = tuple[Array, Array]


def get_latent_ewald_grid(
    box_vectors,
    *,
    dl: float,
    box_scale: float = 1.25,
    dtype=jnp.float32,
) -> LatentEwaldGrid:
    if box_vectors is None:
        raise ValueError("Periodic LES preparation requires box vectors.")
    if box_scale < 1.0:
        raise ValueError("box_scale must be at least 1.0.")
    lengths = np.linalg.norm(np.asarray(box_vectors), axis=1)
    allocation_lengths = lengths * float(box_scale)
    max_indices = tuple(max(1, int(length / float(dl))) for length in allocation_lengths)
    axes = [np.arange(-n, n + 1, dtype=np.int32) for n in max_indices]
    reciprocal_indices = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    nonzero_components = reciprocal_indices != 0
    first_nonzero_axis = np.argmax(nonzero_components, axis=1)
    first_nonzero_value = reciprocal_indices[
        np.arange(reciprocal_indices.shape[0]), first_nonzero_axis
    ]
    reciprocal_indices = reciprocal_indices[
        np.any(nonzero_components, axis=1) & (first_nonzero_value > 0)
    ]
    return (
        jnp.asarray(reciprocal_indices, dtype=jnp.int32),
        jnp.asarray(allocation_lengths, dtype=dtype),
    )


class LatentEwald(eqx.Module):
    sigma: float = eqx.field(static=True)
    dl: float = eqx.field(static=True)
    norm_factor_hartree: float = eqx.field(static=True)

    def __init__(
        self,
        sigma: float,
        dl: float,
        norm_factor_hartree: float,
    ):
        self.sigma = float(sigma)
        self.dl = float(dl)
        self.norm_factor_hartree = float(norm_factor_hartree)

    def __call__(
        self,
        positions: Array,
        latent_charges: Array,
        *,
        box_vectors: Array | None,
        grid: LatentEwaldGrid | None,
    ) -> Array:
        charges = jnp.asarray(latent_charges, dtype=positions.dtype)
        two_pi = jnp.asarray(2.0 * np.pi, dtype=positions.dtype)
        sigma = jnp.asarray(self.sigma, dtype=positions.dtype)
        norm = jnp.asarray(self.norm_factor_hartree, dtype=positions.dtype)

        if box_vectors is None:
            pair_displacements = positions[None, :, :] - positions[:, None, :]
            squared_distances = jnp.sum(pair_displacements * pair_displacements, axis=-1)
            off_diagonal_mask = ~jnp.eye(positions.shape[0], dtype=jnp.bool_)
            distances = jnp.sqrt(jnp.maximum(squared_distances, 1.0e-24))
            inverse_distance = jnp.where(off_diagonal_mask, 1.0 / distances, 0.0)
            kernel = (
                jax.lax.erf(distances / (sigma * jnp.sqrt(2.0)))
                * inverse_distance
                * norm
                / two_pi
            )
            return 0.5 * jnp.einsum("i,ij,j->", charges, kernel, charges)

        if grid is None:
            raise ValueError("Periodic LES requires a grid from get_latent_ewald_grid().")
        reciprocal_indices, max_box_lengths = grid
        box_matrix = jnp.asarray(box_vectors, dtype=positions.dtype)
        reciprocal_indices = reciprocal_indices.astype(positions.dtype)
        volume = jnp.linalg.det(box_matrix)
        reciprocal_cell = two_pi * jnp.linalg.inv(box_matrix).T
        reciprocal_vectors = reciprocal_indices @ reciprocal_cell
        squared_wavevectors = jnp.sum(reciprocal_vectors * reciprocal_vectors, axis=1)
        max_squared_wavevector = (two_pi / jnp.asarray(self.dl, dtype=positions.dtype)) ** 2
        active_wavevectors = (squared_wavevectors > 0.0) & (
            squared_wavevectors <= max_squared_wavevector
        )
        safe_squared_wavevectors = jnp.where(active_wavevectors, squared_wavevectors, 1.0)

        phase = positions @ reciprocal_vectors.T
        structure_real = jnp.sum(charges[:, None] * jnp.cos(phase), axis=0)
        structure_imag = jnp.sum(charges[:, None] * jnp.sin(phase), axis=0)
        structure_sq = structure_real * structure_real + structure_imag * structure_imag
        reciprocal_kernel = jnp.where(
            active_wavevectors,
            jnp.exp(-0.5 * sigma * sigma * safe_squared_wavevectors)
            / safe_squared_wavevectors,
            0.0,
        )
        energy = 2.0 * norm * jnp.sum(reciprocal_kernel * structure_sq) / volume
        self_energy = jnp.sum(charges * charges) * norm / (
            sigma * jnp.power(two_pi, 1.5)
        )
        within_allocation_bounds = jnp.all(
            jnp.linalg.norm(box_matrix, axis=1)
            <= max_box_lengths.astype(positions.dtype)
        )
        return jnp.where(within_allocation_bounds, energy - self_energy, jnp.nan)


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
        neighbors = neighbors.update(positions)
    else:
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
        neighbors = neighbor_fn.allocate(
            positions,
            **neighbor_kwargs,
        )
    return neighbors


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


def polynomial_cutoff(r: Array, r_max: float, p: int = 5) -> Array:
    u = r / r_max
    up = jnp.power(u, p)
    envelope = (
        1.0 - 0.5 * (p + 1) * (p + 2) * up + p * (p + 2) * up * u - 0.5 * p * (p + 1) * up * u * u
    )
    return jnp.where(r < r_max, envelope, 0.0)


def bessel_basis(r: Array, r_max: float, num_basis: int) -> Array:
    ns = jnp.arange(1, num_basis + 1, dtype=r.dtype)
    return (
        jnp.sqrt(2.0 / r_max)
        * jnp.pi
        * ns
        / r_max
        * jnp.sinc(ns * r[..., None] / r_max)
    )


def spherical_harmonics_0_to_3(
    edge_vectors: Array,
    coeffs: Array,
    monomials: tuple[tuple[int, int, int], ...],
) -> Array:
    norms = safe_norm(edge_vectors, axis=-1, keepdims=True)
    xyz = edge_vectors / norms
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    cols = []
    for px, py, pz in monomials:
        value = jnp.ones_like(x)
        if px:
            value = value * x**px
        if py:
            value = value * y**py
        if pz:
            value = value * z**pz
        cols.append(value)
    return jnp.stack(cols, axis=-1) @ coeffs.T


class Linear(eqx.Module):
    w: Array
    in_dim: int = eqx.field(static=True)

    def __init__(self, in_dim: int, out_dim: int, *, dtype: Any = jnp.float32, key: Array):
        self.in_dim = in_dim
        self.w = jax.random.normal(key, (in_dim, out_dim), dtype)

    def __call__(self, x: Array) -> Array:
        scale = jnp.sqrt(1.0 / self.in_dim)
        if x.shape[-1] == self.in_dim:
            return scale * jnp.matmul(x, self.w)
        if x.ndim == 3 and x.shape[1] == self.in_dim:
            return scale * jnp.einsum("nci,co->noi", x, self.w)
        raise ValueError(f"Expected feature axis of size {self.in_dim}, got shape {x.shape}.")


class SpeciesLinear(eqx.Module):
    w: Array
    species_normalization_count: int = eqx.field(static=True)
    in_dim: int = eqx.field(static=True)

    def __init__(
        self,
        num_species_embeddings: int,
        in_dim: int,
        out_dim: int,
        *,
        species_normalization_count: int,
        dtype: Any = jnp.float32,
        key: Array,
    ):
        self.species_normalization_count = species_normalization_count
        self.in_dim = in_dim
        self.w = jax.random.normal(key, (num_species_embeddings, in_dim, out_dim), dtype)

    def __call__(self, x: Array, species: Array) -> Array:
        scale = jnp.sqrt(1.0 / (self.in_dim * self.species_normalization_count))
        weights = self.w[species]
        if x.shape[-1] == self.in_dim:
            return scale * jnp.einsum("n...i,nio->n...o", x, weights)
        if x.ndim == 3 and x.shape[1] == self.in_dim:
            return scale * jnp.einsum("nci,nco->noi", x, weights)
        raise ValueError(f"Expected feature axis of size {self.in_dim}, got shape {x.shape}.")


class MLP(eqx.Module):
    linears: list[Array]
    silu_normalization: float = eqx.field(static=True)

    def __init__(self, layer_sizes, *, silu_normalization: float, dtype=jnp.float32, key):
        self.silu_normalization = silu_normalization
        layer_sizes = tuple(layer_sizes)
        keys = jax.random.split(key, len(layer_sizes) - 1)
        self.linears = [
            jax.random.normal(k, (i, o), dtype)
            for k, i, o in zip(keys, layer_sizes[:-1], layer_sizes[1:])
        ]

    def __call__(self, x: Array) -> Array:
        for i, w in enumerate(self.linears):
            x = jnp.sqrt(1.0 / w.shape[0]) * jnp.matmul(x, w)
            if i < len(self.linears) - 1:
                x = jax.nn.silu(x) / self.silu_normalization
        return x


class SymmetricContraction(eqx.Module):
    w0: Array
    w1: Array | None
    u0_1: Array
    u0_2: Array
    u0_3: Array
    u1_1: Array | None
    u1_2: Array | None
    u1_3: Array | None
    output_vector: bool = eqx.field(static=True)

    def __init__(
        self, num_species: int, num_features: int, output_vector: bool, *, dtype=jnp.float32, key
    ):
        self.output_vector = output_vector
        self.w0 = jax.random.normal(key, (num_species, 28, num_features), dtype)
        self.u0_1 = jnp.zeros((16, 1, 1), dtype=dtype)
        self.u0_2 = jnp.zeros((16, 16, 4, 1), dtype=dtype)
        self.u0_3 = jnp.zeros((16, 16, 16, 23, 1), dtype=dtype)
        self.w1 = (
            jnp.zeros((num_species, 58, num_features), dtype=dtype) if output_vector else None
        )
        self.u1_1 = jnp.zeros((16, 1, 3), dtype=dtype) if output_vector else None
        self.u1_2 = jnp.zeros((16, 16, 6, 3), dtype=dtype) if output_vector else None
        self.u1_3 = jnp.zeros((16, 16, 16, 51, 3), dtype=dtype) if output_vector else None

    def contraction_features(self, x: Array, u1: Array, u2: Array, u3: Array) -> Array:
        phi1 = jnp.einsum("nfa,ami->nfmi", x, u1)
        phi2 = jnp.einsum("nfa,nfb,abmi->nfmi", x, x, u2)
        phi3 = jnp.einsum("nfa,nfb,nfc,abcmi->nfmi", x, x, x, u3)
        return jnp.concatenate([phi1, phi2, phi3], axis=2)

    def __call__(
        self, blocks: tuple[Array, Array, Array, Array], species: Array
    ) -> tuple[Array, Array | None]:
        x = jnp.concatenate(blocks, axis=-1)
        f0 = self.contraction_features(x, self.u0_1, self.u0_2, self.u0_3)
        w0 = jnp.swapaxes(self.w0[species], 1, 2)[..., None]
        out0 = jnp.sum(f0 * w0, axis=2)[..., 0]
        out1 = None
        if self.output_vector:
            assert self.w1 is not None
            assert self.u1_1 is not None
            assert self.u1_2 is not None
            assert self.u1_3 is not None
            f1 = self.contraction_features(x, self.u1_1, self.u1_2, self.u1_3)
            w1 = jnp.swapaxes(self.w1[species], 1, 2)[..., None]
            out1 = jnp.sum(f1 * w1, axis=2)
        return out0, out1


class MACELayer(eqx.Module):
    linear_up0: Linear
    linear_up1: Linear | None
    radial_mlp: MLP
    linear_down: list[Linear]
    skip0: SpeciesLinear | None
    linz: list[SpeciesLinear] | None
    sc: SymmetricContraction
    linear_sc0: Linear
    linear_sc1: Linear | None
    clebsch_gordan_coefficients: tuple[Array, ...] | None
    vector_input: bool = eqx.field(static=True)
    vector_output: bool = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    conv_widths: tuple[int, int, int, int] = eqx.field(static=True)
    sh_dims: tuple[int, ...] = eqx.field(static=True)
    sh_starts: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        num_features,
        num_species_embeddings,
        species_normalization_count,
        radial_dim,
        epsilon,
        vector_input,
        vector_output,
        has_skip,
        has_linz,
        *,
        radial_mlp_widths: tuple[int, ...],
        sh_dims: tuple[int, ...],
        sh_starts: tuple[int, ...],
        silu_normalization: float,
        dtype=jnp.float32,
        key,
    ):
        keys = iter(jax.random.split(key, 16))
        self.vector_input = vector_input
        self.vector_output = vector_output
        self.epsilon = epsilon
        self.sh_dims = sh_dims
        self.sh_starts = sh_starts
        self.conv_widths = (
            (2 * num_features, 3 * num_features, 3 * num_features, 2 * num_features)
            if vector_input
            else (num_features,) * 4
        )
        self.linear_up0 = Linear(num_features, num_features, dtype=dtype, key=next(keys))
        self.linear_up1 = (
            Linear(num_features, num_features, dtype=dtype, key=next(keys))
            if vector_input
            else None
        )
        self.radial_mlp = MLP(
            [radial_dim, *radial_mlp_widths, sum(self.conv_widths)],
            silu_normalization=silu_normalization,
            dtype=dtype,
            key=next(keys),
        )
        self.linear_down = [
            Linear(w, num_features, dtype=dtype, key=next(keys)) for w in self.conv_widths
        ]
        self.skip0 = (
            SpeciesLinear(
                num_species_embeddings,
                num_features,
                num_features,
                species_normalization_count=species_normalization_count,
                dtype=dtype,
                key=next(keys),
            )
            if has_skip
            else None
        )
        self.linz = (
            [
                SpeciesLinear(
                    num_species_embeddings,
                    num_features,
                    num_features,
                    species_normalization_count=species_normalization_count,
                    dtype=dtype,
                    key=next(keys),
                )
                for _ in range(4)
            ]
            if has_linz
            else None
        )
        self.sc = SymmetricContraction(
            num_species_embeddings,
            num_features,
            vector_output,
            dtype=dtype,
            key=next(keys),
        )
        self.linear_sc0 = Linear(num_features, num_features, dtype=dtype, key=next(keys))
        self.linear_sc1 = (
            Linear(num_features, num_features, dtype=dtype, key=next(keys))
            if vector_output
            else None
        )
        self.clebsch_gordan_coefficients = (
            (
                jnp.zeros((3, 3, 1), dtype=dtype),
                jnp.zeros((3, 1, 3), dtype=dtype),
                jnp.zeros((3, 5, 3), dtype=dtype),
                jnp.zeros((3, 3, 5), dtype=dtype),
                jnp.zeros((3, 7, 5), dtype=dtype),
                jnp.zeros((3, 5, 7), dtype=dtype),
            )
            if vector_input
            else None
        )

    def messages_scalar_input(self, h0: Array, y: Array) -> list[Array]:
        hs = h0[..., :, None]
        out = []
        for ell, start in enumerate(self.sh_starts):
            yl = y[..., start : start + self.sh_dims[ell]]
            out.append(hs * yl[..., None, :])
        return out

    def messages_vector_input(self, h0: Array, h1: Array, y: Array) -> list[Array]:
        hs = h0[..., :, None]
        y0 = y[..., 0:1]
        y1 = y[..., 1:4]
        y2 = y[..., 4:9]
        y3 = y[..., 9:16]
        assert self.clebsch_gordan_coefficients is not None
        c11_0, c10_1, c12_1, c11_2, c13_2, c12_3 = self.clebsch_gordan_coefficients
        out0 = jnp.concatenate(
            [hs * y0[..., None, :], jnp.einsum("...fi,...j,ijk->...fk", h1, y1, c11_0)], axis=-2
        )
        out1 = jnp.concatenate(
            [
                hs * y1[..., None, :],
                jnp.einsum("...fi,...j,ijk->...fk", h1, y0, c10_1),
                jnp.einsum("...fi,...j,ijk->...fk", h1, y2, c12_1),
            ],
            axis=-2,
        )
        out2 = jnp.concatenate(
            [
                hs * y2[..., None, :],
                jnp.einsum("...fi,...j,ijk->...fk", h1, y1, c11_2),
                jnp.einsum("...fi,...j,ijk->...fk", h1, y3, c13_2),
            ],
            axis=-2,
        )
        out3 = jnp.concatenate(
            [hs * y3[..., None, :], jnp.einsum("...fi,...j,ijk->...fk", h1, y2, c12_3)], axis=-2
        )
        return [out0, out1, out2, out3]

    def __call__(
        self,
        h0: Array,
        h1: Array | None,
        species: Array,
        y: Array,
        radial: Array,
        senders: Array,
        edge_mask: Array,
    ) -> tuple[Array, Array | None]:
        skip = self.skip0(h0, species) if self.skip0 is not None else None
        h0u = self.linear_up0(h0)

        # Tensor Product
        if self.vector_input:
            assert h1 is not None
            assert self.linear_up1 is not None
            h1u = self.linear_up1(h1)
            msg = self.messages_vector_input(h0u[senders], h1u[senders], y)
        else:
            msg = self.messages_scalar_input(h0u[senders], y)

        # Multiply the lmax-3 message with the radial weight
        mix = self.radial_mlp(radial)
        pieces = jnp.split(mix, np.cumsum(self.conv_widths)[:-1], axis=-1)
        blocks = []
        for ell in range(4):
            weighted = self.epsilon * msg[ell] * pieces[ell][..., :, None]
            weighted = jnp.where(edge_mask[..., None, None], weighted, 0.0)
            agg = jnp.sum(weighted, axis=1)
            down = self.linear_down[ell](agg)
            blocks.append(down)

        if self.linz is not None:
            blocks = [self.linz[ell](blocks[ell], species) for ell in range(4)]

        out0, out1 = self.sc(tuple(blocks), species)

        out0 = self.linear_sc0(out0)

        if self.vector_output:
            assert out1 is not None
            assert self.linear_sc1 is not None
            out1 = self.linear_sc1(out1)
        if skip is not None:
            out0 = out0 + skip

        return out0, out1


class ScalarReadout(eqx.Module):
    hidden: Linear | None
    output: Linear
    silu_normalization: float = eqx.field(static=True)

    def __init__(
        self,
        num_features: int,
        *,
        nonlinear: bool,
        hidden_dim: int,
        silu_normalization: float,
        dtype=jnp.float32,
        key,
    ):
        self.silu_normalization = float(silu_normalization)
        if nonlinear:
            hidden_key, output_key = jax.random.split(key)
            self.hidden = Linear(num_features, hidden_dim, dtype=dtype, key=hidden_key)
        else:
            self.hidden = None
            output_key = key
        self.output = Linear(
            hidden_dim if nonlinear else num_features,
            1,
            dtype=dtype,
            key=output_key,
        )

    def __call__(self, features: Array) -> Array:
        if self.hidden is not None:
            features = jax.nn.silu(self.hidden(features)) / self.silu_normalization
        return jnp.squeeze(self.output(features), axis=-1)


class EnergyHead(eqx.Module):
    local_energy_readouts: list[ScalarReadout]
    latent_charge_readouts: list[ScalarReadout] | None
    atomic_shifts: Array

    def __init__(self, offsets, config: dict[str, Any], *, dtype=jnp.float32, key):
        num_layers = int(config["num_layers"])
        num_features = int(config["num_features"])
        hidden_dim = int(config["readout_hidden_dim"])
        silu_normalization = float(config["silu_normalization"])
        has_long_range = config["long_range"] is not None
        num_readouts = num_layers * (2 if has_long_range else 1)
        keys = iter(jax.random.split(key, num_readouts))
        self.local_energy_readouts = [
            ScalarReadout(
                num_features,
                nonlinear=(i == num_layers - 1),
                hidden_dim=hidden_dim,
                silu_normalization=silu_normalization,
                dtype=dtype,
                key=next(keys),
            )
            for i in range(num_layers)
        ]
        self.latent_charge_readouts = (
            [
                ScalarReadout(
                    num_features,
                    nonlinear=(i == num_layers - 1),
                    hidden_dim=hidden_dim,
                    silu_normalization=silu_normalization,
                    dtype=dtype,
                    key=next(keys),
                )
                for i in range(num_layers)
            ]
            if has_long_range
            else None
        )
        self.atomic_shifts = jnp.asarray(np.asarray(offsets).reshape(-1), dtype=dtype)

    def __call__(self, layer_features: list[Array], species: Array) -> tuple[Array, Array | None]:
        node_energy = sum(
            (head(features) for head, features in zip(self.local_energy_readouts, layer_features)),
            start=jnp.zeros(layer_features[0].shape[0], dtype=layer_features[0].dtype),
        )
        node_energy = node_energy + self.atomic_shifts[species].astype(node_energy.dtype)
        latent_charges = None
        if self.latent_charge_readouts is not None:
            latent_charges = sum(
                (
                    head(features)
                    for head, features in zip(self.latent_charge_readouts, layer_features)
                ),
                start=jnp.zeros(layer_features[0].shape[0], dtype=layer_features[0].dtype),
            )
        return node_energy, latent_charges


class MACE(eqx.Module):
    embedding: Array
    layers: list[MACELayer]
    sh_coeffs: Array
    energy_head: EnergyHead
    long_range: LatentEwald | None
    sh_monomials: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    species_normalization_count: int = eqx.field(static=True)
    implemented_species: tuple[int, ...] = eqx.field(static=True)
    species_lookup: tuple[int, ...] = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    num_radial_basis: int = eqx.field(static=True)
    radial_polynomial_p: int = eqx.field(static=True)
    neighbor_cell_atom_threshold: int = eqx.field(static=True)
    neighbor_cell_capacity_multiplier: float = eqx.field(static=True)

    def __init__(self, offsets, config: dict[str, Any], *, dtype=jnp.float32, key):
        layer_key, energy_head_key = jax.random.split(key)
        self.species_normalization_count = int(config["species_normalization_count"])
        self.implemented_species = tuple(int(z) for z in config["implemented_species"])
        species_lookup = [-1] * self.species_normalization_count
        for index, atomic_number in enumerate(self.implemented_species):
            species_lookup[atomic_number] = index
        self.species_lookup = tuple(species_lookup)
        num_species_embeddings = len(self.implemented_species)
        self.cutoff = float(config["cutoff"])
        self.num_radial_basis = int(config["num_radial_basis"])
        self.radial_polynomial_p = int(config["radial_polynomial_p"])
        num_features = int(config["num_features"])
        hidden_has_vector = bool(config["hidden_has_vector"])
        self.neighbor_cell_atom_threshold = int(config["neighbor_cell_atom_threshold"])
        self.neighbor_cell_capacity_multiplier = float(config["neighbor_cell_capacity_multiplier"])
        num_layers = int(config["num_layers"])
        first_layer_residual = bool(config["first_layer_residual"])
        radial_mlp_widths = tuple(int(x) for x in config["radial_mlp_widths"])
        sh_dims = tuple(int(x) for x in config["sh_dims"])
        sh_starts = tuple(int(x) for x in config["sh_starts"])
        self.sh_monomials = tuple(
            tuple(int(power) for power in monomial) for monomial in config["sh_monomials"]
        )
        silu_normalization = float(config["silu_normalization"])
        epsilon = 1.0 / float(config["avg_num_neighbors"])
        keys = jax.random.split(layer_key, num_layers + 1)
        self.embedding = jax.random.normal(keys[0], (num_species_embeddings, num_features), dtype)
        self.sh_coeffs = jnp.zeros((sum(sh_dims), len(self.sh_monomials)), dtype=dtype)
        self.layers = [
            MACELayer(
                num_features,
                num_species_embeddings,
                self.species_normalization_count,
                self.num_radial_basis,
                epsilon,
                vector_input=(i > 0 and hidden_has_vector),
                vector_output=(i == 0 and hidden_has_vector),
                has_skip=(i > 0 or (i == 0 and first_layer_residual)),
                has_linz=(i == 0 and not first_layer_residual),
                radial_mlp_widths=radial_mlp_widths,
                sh_dims=sh_dims,
                sh_starts=sh_starts,
                silu_normalization=silu_normalization,
                dtype=dtype,
                key=keys[i + 1],
            )
            for i in range(num_layers)
        ]
        self.energy_head = EnergyHead(offsets, config, dtype=dtype, key=energy_head_key)
        long_range = config["long_range"]
        self.long_range = (
            LatentEwald(
                long_range["sigma"],
                long_range["dl"],
                long_range["norm_factor_hartree"],
            )
            if long_range is not None
            else None
        )

    def __call__(
        self,
        positions,
        species,
        *,
        box_vectors=None,
        neighbors=None,
        neighbor_idx=None,
        long_range_grid: LatentEwaldGrid | None = None,
        periodic: bool = False,
    ):
        box_vectors = box_vectors if periodic else None
        if neighbor_idx is None:
            neighbors = get_neighbors(
                positions,
                box_vectors,
                cutoff=float(self.cutoff),
                cell_atom_threshold=int(self.neighbor_cell_atom_threshold),
                cell_capacity_multiplier=float(self.neighbor_cell_capacity_multiplier),
                neighbors=neighbors,
                periodic=periodic,
            )
            neighbor_idx = neighbors.idx

        atomic_numbers = jnp.asarray(species, dtype=jnp.int32)
        species = jnp.asarray(self.species_lookup, dtype=jnp.int32)[atomic_numbers]
        edge_vectors, senders, edge_mask = dense_neighbor_edges(
            positions,
            neighbor_idx,
            box_vectors=box_vectors,
            cutoff=float(self.cutoff),
        )
        senders = jnp.where(edge_mask, senders, jnp.zeros_like(senders))
        h0 = self.embedding[species] / jnp.sqrt(self.species_normalization_count)
        h1 = None
        distances = safe_norm(edge_vectors, axis=-1)
        radial_basis = (
            bessel_basis(distances, self.cutoff, self.num_radial_basis)
            * polynomial_cutoff(distances, self.cutoff, self.radial_polynomial_p)[..., None]
        )
        radial_basis = jnp.where(edge_mask[..., None], radial_basis, 0.0)
        sph = spherical_harmonics_0_to_3(edge_vectors, self.sh_coeffs, self.sh_monomials)
        sph = jnp.where(edge_mask[..., None], sph, 0.0)

        layer_features = []
        for layer in self.layers:
            h0, h1 = layer(h0, h1, species, sph, radial_basis, senders, edge_mask)
            layer_features.append(h0)

        node_energies, latent_charges = self.energy_head(layer_features, species)
        energy = jnp.sum(node_energies)
        if self.long_range is not None:
            assert latent_charges is not None
            energy = energy + self.long_range(
                positions,
                latent_charges,
                box_vectors=box_vectors,
                grid=long_range_grid,
            )
        return energy


def load_model(
    model: str | PathLike = "mace-jax-off-s-23",
    *,
    dtype=jnp.float32,
    atomic_numbers=None,
    neighbor_cell_atom_threshold: int | None = None,
    neighbor_cell_capacity_multiplier: float | None = None,
) -> MACE:
    path = (
        MACE_MODEL_PATHS[model]
        if isinstance(model, str) and model in MACE_MODEL_PATHS
        else Path(model)
    )

    use_float64 = jnp.dtype(dtype) == jnp.dtype(jnp.float64)
    with jax.enable_x64(use_float64), path.open("rb") as handle:
        config = json.loads(handle.readline().decode("utf-8"))
        if config["long_range"] is not None:
            if config.get("storage_dtype") != "float32":
                raise ValueError("MACE-LES checkpoints must be stored in float32.")
            if jnp.dtype(dtype) != jnp.dtype(jnp.float32):
                raise ValueError("MACE-LES models only support float32.")
        if neighbor_cell_atom_threshold is not None:
            config["neighbor_cell_atom_threshold"] = int(neighbor_cell_atom_threshold)
        if neighbor_cell_capacity_multiplier is not None:
            config["neighbor_cell_capacity_multiplier"] = float(
                neighbor_cell_capacity_multiplier
            )
        model_template = MACE(
            np.zeros((len(config["implemented_species"]),), dtype=np.dtype(dtype)),
            config,
            dtype=dtype,
            key=jax.random.PRNGKey(0),
        )
        loaded_model = eqx.tree_deserialise_leaves(handle, model_template)
        if atomic_numbers is not None:
            atomic_numbers = np.asarray(jax.device_get(atomic_numbers), dtype=np.int64).reshape(-1)
            unsupported = sorted(
                set(atomic_numbers.tolist()) - set(loaded_model.implemented_species)
            )
            if unsupported:
                raise ValueError(f"MACE does not support atomic numbers {unsupported}.")
        return loaded_model
