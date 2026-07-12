# /// script
# dependencies = ["ase", "numpy", "openmm", "torchani"]
# ///
# This script computes reference energies and forces for the ANI2x models.

from pathlib import Path

import ase.io
import numpy as np
import torch
import torchani
from openmm import unit

DATA_DIR = Path(__file__).resolve().parent
HARTREE_TO_KJMOL = (unit.hartree * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
    unit.kilojoules_per_mole
)
HARTREE_A_TO_KJMOL_A = (
    unit.hartree * unit.AVOGADRO_CONSTANT_NA / unit.angstrom
).value_in_unit(unit.kilojoules_per_mole / unit.angstrom)
SYSTEMS = {
    "toluene": DATA_DIR / "toluene" / "toluene.pdb",
}
MODELS = ("ani2x-jax-model0", "ani2x-jax-ensemble")


def calculate_reference(path: Path, model_name: str) -> dict[str, float | np.ndarray]:
    atoms = ase.io.read(path)
    if not torch.cuda.is_available():
        raise RuntimeError("ANI reference generation requires a CUDA GPU")
    device = torch.device("cuda")
    dtype = torch.float32
    model = torchani.models.ANI2x(periodic_table_index=True).to(device=device, dtype=dtype)
    species = torch.tensor(
        atoms.get_atomic_numbers(),
        device=device,
        dtype=torch.long,
    ).unsqueeze(0)
    coordinates = torch.tensor(
        atoms.get_positions(),
        device=device,
        dtype=dtype,
    ).unsqueeze(0)
    coordinates.requires_grad_(True)
    ensemble_values = model_name == "ani2x-jax-model0"
    result = model((species, coordinates), ensemble_values=ensemble_values)
    if ensemble_values:
        energy = result.energies[0, 0]
    else:
        energy = result.energies.squeeze()
    gradient = torch.autograd.grad(energy, coordinates)[0]
    forces = (-gradient.squeeze(0) * HARTREE_A_TO_KJMOL_A).detach().cpu().numpy()
    return {
        "energy": float(energy.detach().cpu().numpy() * HARTREE_TO_KJMOL),
        "forces": forces,
    }


def calculate_results() -> dict[str, float | np.ndarray]:
    results = {}
    for system, path in SYSTEMS.items():
        for model_name in MODELS:
            reference = calculate_reference(path, model_name)
            results[f"{system}/{model_name}"] = reference["energy"]
            results[f"{system}/{model_name}/forces"] = reference["forces"]
    return results


def print_results(results: dict[str, float | np.ndarray]) -> None:
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            value = np.array2string(value, precision=12, separator=", ", threshold=np.inf)
        print(f"{key}: {value!r}")


def main() -> None:
    print_results(calculate_results())


if __name__ == "__main__":
    main()
