# /// script
# dependencies = [
#   "ase",
#   "huggingface-hub",
#   "numpy",
#   "openmm",
#   "torchmd-net @ git+https://github.com/torchmd/torchmd-net.git@2a2c913352a8b5fd297a33dd0ec35c4d69fb1eea",
# ]
# ///
# This script computes AceFF reference energies and forces from upstream TorchMD-Net checkpoints.

from pathlib import Path

import ase.io
import numpy as np
from huggingface_hub import hf_hub_download
from openmm import unit
from torchmdnet.calculators import TMDNETCalculator

DATA_DIR = Path(__file__).resolve().parent
EV_TO_KJMOL = (unit.elementary_charge * unit.volt * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
    unit.kilojoules_per_mole
)
EV_A_TO_KJMOL_A = (
    unit.elementary_charge * unit.volt / unit.angstrom * unit.AVOGADRO_CONSTANT_NA
).value_in_unit(unit.kilojoules_per_mole / unit.angstrom)
SYSTEMS = {
    "toluene": DATA_DIR / "toluene" / "toluene.pdb",
    "alanine-dipeptide-explicit": DATA_DIR
    / "alanine-dipeptide"
    / "alanine-dipeptide-explicit.pdb",
}
MODELS = {
    "aceff-jax-1.1": (
        "Acellera/AceFF-1.1",
        "aceff_v1.1.ckpt",
        {},
    ),
    "aceff-jax-2.0": (
        "Acellera/AceFF-2.0",
        "aceff_v2.0.ckpt",
        {"coulomb_cutoff": 12.0},
    ),
}


def calculate_reference(
    path: Path,
    model_name: str,
    *,
    include_forces: bool = True,
) -> dict[str, float | np.ndarray]:
    repo_id, filename, kwargs = MODELS[model_name]
    model_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
    )

    atoms = ase.io.read(path)
    atoms.info["charge"] = 0
    atoms.calc = TMDNETCalculator(
        model_file,
        device="cuda",
        **kwargs,
    )
    reference: dict[str, float | np.ndarray] = {
        "energy": atoms.get_potential_energy() * EV_TO_KJMOL,
    }
    if include_forces:
        reference["forces"] = atoms.get_forces() * EV_A_TO_KJMOL_A
    return reference


def calculate_results() -> dict[str, float | np.ndarray]:
    results = {}

    for system_name, path in SYSTEMS.items():
        for model_name in MODELS:
            reference = calculate_reference(path, model_name)
            results[f"{system_name}/{model_name}"] = reference["energy"]
            results[f"{system_name}/{model_name}/forces"] = reference["forces"]

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
