# /// script
# dependencies = ["numpy", "openmm", "setuptools<81", "torchani==2.2.4"]
# ///

from pathlib import Path

import numpy as np
import torch
import torchani
from openmm import app, unit
from reference_utils import (
    HARTREE_A_TO_KJMOL_A,
    HARTREE_TO_KJMOL,
)

DATA_DIR = Path(__file__).resolve().parent
SYSTEMS = {"toluene": DATA_DIR / "toluene" / "toluene.pdb"}
MODELS = ("ani2x-jax-model0", "ani2x-jax-ensemble")


def calculate_reference(
    structure_path: Path,
    model_name: str,
    ensemble,
) -> dict[str, float | np.ndarray]:
    pdb = app.PDBFile(str(structure_path))
    species = torch.tensor(
        [atom.element.atomic_number for atom in pdb.topology.atoms()],
        device="cuda",
        dtype=torch.long,
    ).unsqueeze(0)
    model = ensemble[0] if model_name == "ani2x-jax-model0" else ensemble
    coordinates = torch.tensor(
        pdb.getPositions(asNumpy=True).value_in_unit(unit.angstrom),
        device="cuda",
        dtype=torch.float64,
        requires_grad=True,
    ).unsqueeze(0)
    converted = model.species_converter((species, coordinates))
    species_aev = model.aev_computer(converted)
    species_energy = model.neural_networks(species_aev)
    energy = model.energy_shifter(species_energy).energies.squeeze()
    forces = -torch.autograd.grad(energy, coordinates)[0].squeeze(0)

    return {
        "energy": np.float64(energy.detach().cpu().item() * HARTREE_TO_KJMOL).item(),
        "forces": np.asarray(
            forces.detach().cpu().numpy() * HARTREE_A_TO_KJMOL_A,
            dtype=np.float64,
        ),
    }


def main() -> None:
    ensemble = torchani.models.ANI2x(periodic_table_index=True).to(
        device="cuda",
        dtype=torch.float64,
    )
    results: dict[str, float | np.ndarray] = {}
    for system_name, structure_path in SYSTEMS.items():
        for model_name in MODELS:
            reference = calculate_reference(
                structure_path,
                model_name,
                ensemble,
            )
            result_key = f"{system_name}/{model_name}"
            results[result_key] = reference["energy"]
            results[f"{result_key}/forces"] = reference["forces"]

    for key, value in results.items():
        if isinstance(value, np.ndarray):
            value = np.array2string(value, precision=17, separator=", ", threshold=np.inf)
            print(f"{key}: np.array({value}, dtype=np.float64)")
        else:
            print(f"{key}: {value!r}")


if __name__ == "__main__":
    main()
