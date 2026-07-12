# /// script
# dependencies = ["aimnet", "ase", "numpy", "openmm"]
# ///
# This script computes reference energies for the AIMNet2 model.

from pathlib import Path

import ase.io
import numpy as np
from aimnet.calculators import AIMNet2ASE
from openmm import unit

DATA_DIR = Path(__file__).resolve().parent
EV_TO_KJMOL = (unit.elementary_charge * unit.volt * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
    unit.kilojoules_per_mole
)
EV_A_TO_KJMOL_A = (
    unit.elementary_charge * unit.volt / unit.angstrom * unit.AVOGADRO_CONSTANT_NA
).value_in_unit(unit.kilojoules_per_mole / unit.angstrom)
MODEL = "aimnet2-jax"
SYSTEMS = {
    "toluene": DATA_DIR / "toluene" / "toluene.pdb",
    "alanine-dipeptide-explicit": DATA_DIR
    / "alanine-dipeptide"
    / "alanine-dipeptide-explicit.pdb",
}


def calculate_reference(path: Path, *, include_forces: bool = True) -> dict[str, float | np.ndarray]:
    atoms = ase.io.read(path)
    atoms.calc = AIMNet2ASE("aimnet2", charge=0)
    reference: dict[str, float | np.ndarray] = {
        "energy": atoms.get_potential_energy() * EV_TO_KJMOL,
    }
    if include_forces:
        reference["forces"] = atoms.get_forces() * EV_A_TO_KJMOL_A
    return reference


def calculate_results() -> dict[str, float | np.ndarray]:
    results = {}
    for system, path in SYSTEMS.items():
        reference = calculate_reference(path)
        results[f"{system}/{MODEL}"] = reference["energy"]
        results[f"{system}/{MODEL}/forces"] = reference["forces"]
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
