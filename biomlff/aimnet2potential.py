from __future__ import annotations

from functools import partial
from typing import Iterable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import openmm
import openmm.app as app
import openmmjax
from openmm import unit
from openmmjax_export import (
    configure_pjrt_plugin,
    export_jax_model,
)
from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory

from .aimnet2 import (
    AIMNET2_MODEL_NAMES,
    get_neighbors,
    load_model,
)


class AIMNet2PotentialImplFactory(MLPotentialImplFactory):
    def createImpl(
        self,
        name,
        modelPath=None,
        total_charge: float = 0.0,
        multiplicity: int = 1,
        **args,
    ):
        return AIMNet2PotentialImpl(
            name,
            modelPath=modelPath,
            total_charge=total_charge,
            multiplicity=multiplicity,
        )


class AIMNet2PotentialImpl(MLPotentialImpl):
    def __init__(
        self,
        name,
        modelPath=None,
        total_charge: float = 0.0,
        multiplicity: int = 1,
    ):
        self.name = name
        self.modelPath = modelPath
        self.total_charge = total_charge
        self.multiplicity = multiplicity

    def addForces(
        self,
        topology: app.Topology,
        system: openmm.System,
        atoms: Optional[Iterable[int]],
        forceGroup: int,
        modelPath: Optional[str] = None,
        total_charge: Optional[float] = None,
        neighbor_cell_atom_threshold: Optional[int] = None,
        neighbor_cell_capacity_multiplier: Optional[float] = None,
        periodic_neighborlist: bool = True,
        multiplicity: Optional[int] = None,
        preprocessing_positions=None,
        preprocessing_positions_unit=unit.nanometer,
        use_float64: bool = False,
        **args,
    ):
        with jax.enable_x64(use_float64):
            if (self.multiplicity if multiplicity is None else multiplicity) != 1:
                raise ValueError("AIMNet2 JAX only supports multiplicity=1")
            included_atoms = list(topology.atoms())
            if atoms is not None:
                atoms = list(atoms)
                included_atoms = [included_atoms[i] for i in atoms]
            atomic_numbers = np.asarray(
                [atom.element.atomic_number for atom in included_atoms],
                dtype=np.int32,
            )
            model_ref = modelPath if modelPath is not None else self.modelPath
            if model_ref is None:
                if self.name in AIMNET2_MODEL_NAMES:
                    model_ref = self.name
                else:
                    raise ValueError("modelPath must be provided for custom AIMNet2 models")
            dtype = jnp.float64 if use_float64 else jnp.float32
            model = load_model(
                model_ref,
                dtype=dtype,
                atomic_numbers=atomic_numbers,
                neighbor_cell_atom_threshold=neighbor_cell_atom_threshold,
                neighbor_cell_capacity_multiplier=neighbor_cell_capacity_multiplier,
            )
            species = jnp.asarray(atomic_numbers, dtype=jnp.int32)
            num_model_atoms = len(included_atoms)

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
                raise ValueError("AIMNet2 JAX requires preprocessing_positions.")
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

            def allocate_neighbor_lists(box_vectors_angstrom, positions_angstrom):
                neighbor_list = allocate_neighbor_list(
                    box_vectors_angstrom,
                    positions_angstrom,
                    cell_atom_threshold=int(model.neighbor_cell_atom_threshold),
                    cutoff=float(model.cutoff),
                    cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
                    periodic=force_periodic,
                )
                long_range_neighbor_list = allocate_neighbor_list(
                    box_vectors_angstrom,
                    positions_angstrom,
                    cell_atom_threshold=int(model.neighbor_cell_atom_threshold),
                    cutoff=float(model.lr_cutoff),
                    cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
                    periodic=force_periodic,
                )
                return neighbor_list, long_range_neighbor_list

            neighbor_list, long_range_neighbor_list = allocate_neighbor_lists(
                allocation_box_vectors_angstrom,
                allocation_positions_angstrom,
            )
            d3_data = model.prepare_d3_data(atomic_numbers)
            energy_fn = partial(
                energyAIMNet2,
                model=model,
                species=species,
                total_charge=jnp.asarray(
                    self.total_charge if total_charge is None else total_charge,
                    dtype=dtype,
                ),
                d3_data=d3_data,
                pbc=force_periodic,
                neighbor_list=neighbor_list,
                long_range_neighbor_list=long_range_neighbor_list,
            )

            def energy_kjmol(positions_nm, box_vectors_nm=None):
                return energy_fn((positions_nm, box_vectors_nm))

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


for model_name in AIMNET2_MODEL_NAMES:
    MLPotential.registerImplFactory(model_name, AIMNet2PotentialImplFactory())

__all__ = [
    "AIMNet2PotentialImpl",
    "AIMNet2PotentialImplFactory",
    "MLPotential",
]


def allocate_neighbor_list(
    box_vectors_angstrom,
    positions_angstrom,
    *,
    cell_atom_threshold: int,
    cutoff: float,
    cell_capacity_multiplier: float,
    periodic: bool,
):
    if periodic and box_vectors_angstrom is None:
        raise ValueError("periodic neighbor-list allocation requires a box.")
    return get_neighbors(
        positions_angstrom,
        box_vectors_angstrom,
        cell_atom_threshold=cell_atom_threshold,
        cutoff=float(cutoff),
        cell_capacity_multiplier=cell_capacity_multiplier,
        periodic=periodic,
    )


def energyAIMNet2(
    state,
    model,
    species,
    total_charge,
    d3_data,
    pbc: bool,
    neighbor_list,
    long_range_neighbor_list,
):
    """Evaluate AIMNet2 energy in kJ/mol from OpenMM positions in nm."""
    positions_nm, box_vectors_nm = state
    angstrom_per_nm = unit.nanometer.conversion_factor_to(unit.angstrom)
    positions_angstrom = positions_nm * angstrom_per_nm
    box_vectors_angstrom = None
    if pbc and box_vectors_nm is not None:
        box_vectors_angstrom = box_vectors_nm * angstrom_per_nm
    energy = model(
        positions_angstrom,
        species,
        d3_data=d3_data,
        box_vectors=box_vectors_angstrom,
        neighbors=neighbor_list,
        lr_neighbors=long_range_neighbor_list,
        periodic=pbc,
        total_charge=total_charge,
    )
    return (energy * model.ev_to_kjmol).astype(positions_nm.dtype)
