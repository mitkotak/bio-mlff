# /// script
# dependencies = ["ase", "mace-torch", "numpy", "openmm"]
# ///
# This script computes reference energies for the MACE JAX foundation models.

from pathlib import Path

import ase.io
import numpy as np
from mace.calculators.foundations_models import mace_off
from openmm import unit

DATA_DIR = Path(__file__).resolve().parent
EV_TO_KJMOL = (unit.elementary_charge * unit.volt * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
    unit.kilojoules_per_mole
)
EV_A_TO_KJMOL_NM = (
    unit.elementary_charge * unit.volt / unit.angstrom * unit.AVOGADRO_CONSTANT_NA
).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
SYSTEMS = {
    "toluene": DATA_DIR / "toluene" / "toluene.pdb",
    "alanine-dipeptide-explicit": DATA_DIR
    / "alanine-dipeptide"
    / "alanine-dipeptide-explicit.pdb",
}
MODELS = {
    "mace-jax-off-s-23": "small",
    "mace-jax-off-m-24": (
        "https://github.com/ACEsuit/mace-off/blob/main/mace_off24/MACE-OFF24_medium.model?raw=true"
    ),
}


def calculate_reference(
    path: Path,
    checkpoint: str,
    *,
    include_forces: bool = True,
) -> dict[str, float | np.ndarray]:
    atoms = ase.io.read(path)
    atoms.calc = mace_off(checkpoint, device="cpu", default_dtype="float32")
    reference: dict[str, float | np.ndarray] = {
        "energy": atoms.get_potential_energy() * EV_TO_KJMOL,
    }
    if include_forces:
        reference["forces"] = atoms.get_forces() * EV_A_TO_KJMOL_NM
    return reference


def calculate_results() -> dict[str, float | np.ndarray]:
    results = {}

    for model_name, checkpoint in MODELS.items():
        reference = calculate_reference(SYSTEMS["toluene"], checkpoint)
        results[f"toluene/{model_name}"] = reference["energy"]
        results[f"toluene/{model_name}/forces"] = reference["forces"]

    reference = calculate_reference(
        SYSTEMS["alanine-dipeptide-explicit"],
        MODELS["mace-jax-off-s-23"],
    )
    results["alanine-dipeptide-explicit/mace-jax-off-s-23"] = reference["energy"]
    results["alanine-dipeptide-explicit/mace-jax-off-s-23/forces"] = reference["forces"]
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
