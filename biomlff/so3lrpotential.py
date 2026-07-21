from __future__ import annotations

from collections.abc import Iterable

import jax
import jax.numpy as jnp
import openmm
import openmm.app as app
import openmmjax
from openmm import unit
from openmmjax_export import (
    configure_pjrt_plugin,
    export_jax_model,
)
from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory

from .so3lr import SO3LR_MODEL_PATHS, get_sparse_neighbors, load_model


class SO3LRPotentialImplFactory(MLPotentialImplFactory):
    def createImpl(self, name, modelPath=None, total_charge: float = 0.0, **args):
        return SO3LRPotentialImpl(
            name,
            modelPath=modelPath,
            total_charge=total_charge,
        )


class SO3LRPotentialImpl(MLPotentialImpl):
    def __init__(self, name, modelPath=None, total_charge: float = 0.0):
        self.name = name
        self.modelPath = modelPath
        self.total_charge = total_charge

    def addForces(
        self,
        topology: app.Topology,
        system: openmm.System,
        atoms: Iterable[int] | None,
        forceGroup: int,
        modelPath: str | None = None,
        total_charge: float | None = None,
        neighbor_cell_atom_threshold: int | None = None,
        neighbor_cell_capacity_multiplier: float | None = None,
        periodic_neighborlist: bool = True,
        preprocessing_positions=None,
        preprocessing_positions_unit=unit.nanometer,
        use_float64: bool = False,
        **args,
    ):
        with jax.enable_x64(use_float64):
            included_atoms = list(topology.atoms())
            if atoms is not None:
                atoms = list(atoms)
                included_atoms = [included_atoms[i] for i in atoms]
            species = jnp.asarray(
                [atom.element.atomic_number for atom in included_atoms],
                dtype=jnp.int32,
            )
            num_model_atoms = len(included_atoms)

            model_ref = modelPath if modelPath is not None else self.modelPath
            if model_ref is None:
                if self.name in SO3LR_MODEL_PATHS:
                    model_ref = self.name
                else:
                    raise ValueError("modelPath must be provided for custom SO3LR models")
            dtype = jnp.float64 if use_float64 else jnp.float32
            model = load_model(
                model_ref,
                dtype=dtype,
                neighbor_cell_atom_threshold=neighbor_cell_atom_threshold,
                neighbor_cell_capacity_multiplier=neighbor_cell_capacity_multiplier,
            )

            force_periodic = periodic_neighborlist and (
                topology.getPeriodicBoxVectors() is not None
                or system.usesPeriodicBoundaryConditions()
            )
            allocation_box_vectors_angstrom = None
            if force_periodic:
                box_vectors = topology.getPeriodicBoxVectors()
                if box_vectors is None:
                    box_vectors = system.getDefaultPeriodicBoxVectors()
                allocation_box_vectors_angstrom = jnp.asarray(
                    [vector.value_in_unit(unit.angstrom) for vector in box_vectors],
                    dtype=dtype,
                )
            if preprocessing_positions is None:
                raise ValueError("SO3LR JAX requires preprocessing_positions.")
            if hasattr(preprocessing_positions, "value_in_unit"):
                allocation_positions_angstrom = preprocessing_positions.value_in_unit(
                    unit.angstrom
                )
            else:
                allocation_positions_angstrom = jnp.asarray(
                    preprocessing_positions, dtype=dtype
                ) * preprocessing_positions_unit.conversion_factor_to(unit.angstrom)
            allocation_positions_angstrom = jnp.asarray(
                allocation_positions_angstrom,
                dtype=dtype,
            )
            if atoms is not None:
                allocation_positions_angstrom = allocation_positions_angstrom[
                    jnp.asarray(atoms, dtype=jnp.int32)
                ]

            neighbor_list = get_sparse_neighbors(
                allocation_positions_angstrom,
                allocation_box_vectors_angstrom,
                cutoff=float(model.cutoff),
                cell_atom_threshold=int(model.neighbor_cell_atom_threshold),
                cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
                periodic=force_periodic,
            )
            long_range_neighbor_list = get_sparse_neighbors(
                allocation_positions_angstrom,
                allocation_box_vectors_angstrom,
                cutoff=float(model.long_range_cutoff),
                cell_atom_threshold=int(model.neighbor_cell_atom_threshold),
                cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
                periodic=force_periodic,
            )
            model_total_charge = jnp.asarray(
                self.total_charge if total_charge is None else total_charge,
                dtype=dtype,
            )

            def energy_kjmol(positions_nm, box_vectors_nm=None):
                positions_angstrom = positions_nm * unit.nanometer.conversion_factor_to(
                    unit.angstrom
                )
                box_vectors_angstrom = None
                if force_periodic and box_vectors_nm is not None:
                    box_vectors_angstrom = box_vectors_nm * unit.nanometer.conversion_factor_to(
                        unit.angstrom
                    )
                energy = model(
                    positions_angstrom,
                    species,
                    box_vectors=box_vectors_angstrom,
                    neighbors=neighbor_list,
                    neighbors_lr=long_range_neighbor_list,
                    periodic=force_periodic,
                    total_charge=model_total_charge,
                )
                return (energy * model.ev_to_kjmol).astype(positions_nm.dtype)

            def energy_and_forces_kjmol(positions_nm, box_vectors_nm=None):
                energy, energy_gradient = jax.value_and_grad(energy_kjmol)(
                    positions_nm,
                    box_vectors_nm,
                )
                return energy, -energy_gradient

            def forces_kjmol(positions_nm, box_vectors_nm=None):
                return -jax.grad(energy_kjmol)(positions_nm, box_vectors_nm)

            (
                force_mlir,
                energy_mlir,
                energy_and_forces_mlir,
                compile_options_base64,
            ) = export_jax_model(
                num_model_atoms=num_model_atoms,
                force_function=forces_kjmol,
                energy_function=energy_kjmol,
                energy_and_forces_function=energy_and_forces_kjmol,
                periodic=force_periodic,
                input_dtype=dtype,
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


for model_name in SO3LR_MODEL_PATHS:
    MLPotential.registerImplFactory(model_name, SO3LRPotentialImplFactory())
