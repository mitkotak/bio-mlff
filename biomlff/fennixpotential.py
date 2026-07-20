from __future__ import annotations

from functools import partial
from typing import ClassVar, Iterable, Optional, Sequence

import jax
import jax.numpy as jnp
import openmm
import openmm.app as app
import openmmjax
from fennol.utils.atomic_units import au
from openmm import unit
from openmmjax_export import (
    configure_pjrt_plugin,
    export_jax_model,
)
from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory
from openmmml.models.fennixpotential import FeNNixPotentialImpl as OpenMMFeNNixPotentialImpl

jax.config.update("jax_default_matmul_precision", "highest")


class FeNNixPotentialImplFactory(MLPotentialImplFactory):
    def createImpl(
        self,
        name: str,
        modelPath: str | None = None,
        total_charge: int = 0,
        **args,
    ) -> MLPotentialImpl:
        return FeNNixPotentialImpl(name, modelPath, total_charge=total_charge)


class FeNNixPotentialImpl(MLPotentialImpl):
    KNOWN_MODELS: ClassVar[dict[str, tuple[str, bool]]] = OpenMMFeNNixPotentialImpl.KNOWN_MODELS

    def __init__(
        self,
        name: str,
        modelPath: str | None = None,
        total_charge: int = 0,
    ) -> None:
        self.name = name
        self.modelPath = modelPath
        self.total_charge = total_charge

    def addForces(
        self,
        topology: app.Topology,
        system: openmm.System,
        atoms: Optional[Iterable[int]],
        forceGroup: int,
        total_charge: Optional[int] = None,
        use_float64: bool = False,
        energy_terms: Optional[Sequence[str]] = None,
        periodic_neighborlist: bool = True,
        minimum_image: bool = True,
        nblist_skin: float | None = 1.5,
        nblist_mult_size: float | None = 1.5,
        nblist_add_neigh: int | None = None,
        preprocessing_positions=None,
        preprocessing_positions_unit=unit.nanometer,
        **args,
    ):
        with jax.enable_x64(use_float64):
            import fennol
            import numpy as np

            if preprocessing_positions is None:
                raise ValueError(
                    "FeNNix JAX requires preprocessing_positions to initialize "
                    "fixed preprocessing shapes for export."
                )
            if self.modelPath is not None:
                model_path = self.modelPath
            elif self.name in FeNNixPotentialImpl.KNOWN_MODELS:
                model_url, _ = FeNNixPotentialImpl.KNOWN_MODELS[self.name]
                model_path = self._downloadOrFindFile(f"{self.name}.fnx", model_url)
            else:
                raise ValueError("modelPath must be provided for custom FeNNix models")

            model = fennol.FENNIX.load(model_path, **args)
            if energy_terms is not None:
                model.set_energy_terms(energy_terms)
            ev_to_kjmol = (
                unit.elementary_charge * unit.volt * unit.AVOGADRO_CONSTANT_NA
            ).value_in_unit(unit.kilojoules_per_mole)
            energy_scale = au.EV / model.Ha_to_model_energy * ev_to_kjmol
            force_scale = (energy_scale / unit.angstrom).value_in_unit(unit.nanometer**-1)

            # Get the atoms that should be included.
            included_atoms = list(topology.atoms())
            if atoms is not None:
                atoms = list(atoms)
                included_atoms = [included_atoms[i] for i in atoms]
            species = jnp.asarray(
                [atom.element.atomic_number for atom in included_atoms], dtype=jnp.int32
            )
            atom_indices = None if atoms is None else np.asarray(atoms, dtype=np.int32)
            num_system_atoms = system.getNumParticles() or topology.getNumAtoms()

            model_inputs = {
                "species": species,
                "natoms": jnp.array([species.size], dtype=jnp.int32),
                "batch_index": jnp.zeros(species.size, dtype=jnp.int32),
                "total_charge": self.total_charge if total_charge is None else total_charge,
            }

            force_periodic = periodic_neighborlist and (
                topology.getPeriodicBoxVectors() is not None
                or system.usesPeriodicBoundaryConditions()
            )
            if force_periodic and minimum_image:
                model_inputs["flags"] = {"minimum_image": None}
            preprocessing_dtype = np.float64 if use_float64 else np.float32

            # Prepare static inputs and initialize preprocessing state on CPU
            static_inputs = {
                key: np.asarray(value) if isinstance(value, jax.Array) else value
                for key, value in model_inputs.items()
            }
            preprocessing_coordinates = initial_preprocessing_coordinates_angstrom(
                preprocessing_positions,
                dtype=preprocessing_dtype,
                indices=atom_indices,
                system_shape=(num_system_atoms, 3),
                fallback_shape=(species.size, 3),
                positions_unit=preprocessing_positions_unit,
            )
            preprocessing_inputs = {
                **static_inputs,
                "coordinates": preprocessing_coordinates,
            }
            if force_periodic:
                box_vectors = topology.getPeriodicBoxVectors()
                if box_vectors is None:
                    box_vectors = system.getDefaultPeriodicBoxVectors()
                cells_angstrom = np.asarray(
                    [vector.value_in_unit(unit.angstrom) for vector in box_vectors],
                    dtype=preprocessing_dtype,
                ).reshape(1, 3, 3)
                preprocessing_inputs["cells"] = cells_angstrom
                preprocessing_inputs["reciprocal_cells"] = np.linalg.inv(cells_angstrom)
            preprocessing_state = configured_preprocessing_state(
                model.preprocessing,
                nblist_skin=nblist_skin,
                nblist_mult_size=nblist_mult_size,
                nblist_add_neigh=nblist_add_neigh,
            )
            preprocessing_state, _ = model.preprocessing(
                preprocessing_state,
                preprocessing_inputs,
            )

            coordinate_dtype = jnp.float64 if use_float64 else jnp.float32

            energy_fn = partial(
                energyFeNNix,
                model=model,
                static_inputs=static_inputs,
                preprocessing_state=preprocessing_state,
                pbc=force_periodic,
                energy_scale=energy_scale,
                coordinate_dtype=coordinate_dtype,
            )
            energy_and_forces_fn = partial(
                energyAndForcesFeNNix,
                model=model,
                static_inputs=static_inputs,
                preprocessing_state=preprocessing_state,
                pbc=force_periodic,
                energy_scale=energy_scale,
                force_scale=force_scale,
                coordinate_dtype=coordinate_dtype,
            )

            def energy_kjmol(positions_nm, box_vectors_nm=None):
                return energy_fn((positions_nm, box_vectors_nm))

            def energy_and_forces_kjmol(positions_nm, box_vectors_nm=None):
                energy, forces = energy_and_forces_fn((positions_nm, box_vectors_nm))
                return energy, forces

            def forces_kjmol(positions_nm, box_vectors_nm=None):
                _, forces = energy_and_forces_fn((positions_nm, box_vectors_nm))
                return forces

            (
                force_mlir,
                energy_mlir,
                energy_and_forces_mlir,
                compile_options_base64,
            ) = export_jax_model(
                num_model_atoms=species.size,
                force_function=forces_kjmol,
                energy_function=energy_kjmol,
                energy_and_forces_function=energy_and_forces_kjmol,
                periodic=force_periodic,
                input_dtype=coordinate_dtype,
            )

            force = openmmjax.JaxForce(
                force_mlir,
                energy_mlir,
                energy_and_forces_mlir,
                compile_options_base64,
            )
            configure_pjrt_plugin(force)
            force.setForceGroup(forceGroup)
            force.setUsesPeriodicBoundaryConditions(force_periodic)
            if atoms is not None:
                force.setParticles(atoms)
            system.addForce(force)


for model_name in FeNNixPotentialImpl.KNOWN_MODELS:
    MLPotential.registerImplFactory(model_name, FeNNixPotentialImplFactory())

__all__ = [
    "FeNNixPotentialImpl",
    "FeNNixPotentialImplFactory",
    "MLPotential",
]


def initial_preprocessing_coordinates_angstrom(
    positions,
    *,
    dtype,
    indices,
    system_shape: tuple[int, int],
    fallback_shape: tuple[int, int],
    positions_unit,
):
    import numpy as np

    if hasattr(positions, "value_in_unit"):
        coordinates = np.asarray(positions.value_in_unit(unit.angstrom), dtype=dtype)
    else:
        coordinates = np.asarray(positions, dtype=dtype)
        coordinates *= positions_unit.conversion_factor_to(unit.angstrom)

    if coordinates.shape != system_shape:
        raise ValueError(
            f"preprocessing_positions must have shape {system_shape}, got {coordinates.shape}"
        )

    if indices is not None:
        coordinates = coordinates[indices]

    if coordinates.shape != fallback_shape:
        raise ValueError(
            "selected preprocessing_positions must have shape "
            f"{fallback_shape}, got {coordinates.shape}"
        )
    return coordinates


def configured_preprocessing_state(
    preprocessing,
    *,
    nblist_skin: float | None,
    nblist_mult_size: float | None,
    nblist_add_neigh: int | None,
):
    from flax.core import freeze, unfreeze

    state = unfreeze(preprocessing.init())
    layer_states = []
    for layer_state in state["layers_state"]:
        layer_state = unfreeze(layer_state)
        if nblist_skin is not None and nblist_skin > 0:
            layer_state["nblist_skin"] = float(nblist_skin)
        if nblist_mult_size is not None:
            layer_state["nblist_mult_size"] = float(nblist_mult_size)
        if nblist_add_neigh is not None:
            layer_state["add_neigh"] = int(nblist_add_neigh)
        layer_states.append(freeze(layer_state))
    state["layers_state"] = tuple(layer_states)
    return freeze(state)


def preprocessFeNNix(
    state,
    model,
    static_inputs,
    preprocessing_state,
    pbc: bool,
    coordinate_dtype,
):
    positions_nm, box_vectors_nm = state
    preprocessing_inputs = {
        **static_inputs,
        "coordinates": (positions_nm * unit.nanometer.conversion_factor_to(unit.angstrom)).astype(
            coordinate_dtype
        ),
    }
    if pbc and box_vectors_nm is not None:
        cells_angstrom = box_vectors_nm * unit.nanometer.conversion_factor_to(unit.angstrom)
        cells_angstrom = cells_angstrom.reshape(1, 3, 3).astype(coordinate_dtype)
        preprocessing_inputs["cells"] = cells_angstrom
        preprocessing_inputs["reciprocal_cells"] = inverse_3x3(cells_angstrom)
    return model.preprocessing.process(preprocessing_state, preprocessing_inputs)


def energyFeNNix(
    state,
    model,
    static_inputs,
    preprocessing_state,
    pbc: bool,
    energy_scale: float,
    coordinate_dtype,
):
    """Evaluate FeNNix energy in kJ/mol from OpenMM positions in nm."""
    processed_inputs = preprocessFeNNix(
        state,
        model=model,
        static_inputs=static_inputs,
        preprocessing_state=preprocessing_state,
        pbc=pbc,
        coordinate_dtype=coordinate_dtype,
    )
    energy, _ = model._total_energy(model.variables, processed_inputs)
    return (energy.squeeze() * energy_scale).astype(coordinate_dtype)


def energyAndForcesFeNNix(
    state,
    model,
    static_inputs,
    preprocessing_state,
    pbc: bool,
    energy_scale: float,
    force_scale: float,
    coordinate_dtype,
):
    """Evaluate FeNNix energy and forces in OpenMM units from positions in nm."""
    processed_inputs = preprocessFeNNix(
        state,
        model=model,
        static_inputs=static_inputs,
        preprocessing_state=preprocessing_state,
        pbc=pbc,
        coordinate_dtype=coordinate_dtype,
    )
    energy, forces, _ = model._energy_and_forces(model.variables, processed_inputs)
    return (
        (energy.squeeze() * energy_scale).astype(coordinate_dtype),
        (forces * force_scale).astype(coordinate_dtype),
    )


def inverse_3x3(matrix):
    """Invert one or more 3x3 matrices without lowering to a solver FFI call."""
    a = matrix[..., 0, 0]
    b = matrix[..., 0, 1]
    c = matrix[..., 0, 2]
    d = matrix[..., 1, 0]
    e = matrix[..., 1, 1]
    f = matrix[..., 1, 2]
    g = matrix[..., 2, 0]
    h = matrix[..., 2, 1]
    i = matrix[..., 2, 2]

    cofactors = jnp.stack(
        [
            jnp.stack([e * i - f * h, c * h - b * i, b * f - c * e], axis=-1),
            jnp.stack([f * g - d * i, a * i - c * g, c * d - a * f], axis=-1),
            jnp.stack([d * h - e * g, b * g - a * h, a * e - b * d], axis=-1),
        ],
        axis=-2,
    )
    determinant = a * cofactors[..., 0, 0] + b * cofactors[..., 1, 0] + c * cofactors[..., 2, 0]
    return cofactors / determinant[..., None, None]
