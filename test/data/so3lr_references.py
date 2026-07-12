# /// script
# dependencies = [
#   "ase",
#   "numpy",
#   "openmm",
#   "so3lr @ git+https://github.com/general-molecular-simulations/so3lr.git@5c6f36914bd2424563c1fd80bc21610960d947a4",
#   "mlff @ git+https://github.com/kabylda/mlff.git@aeb80dcd208a4607c01dbac6c9574b7b32bcf93e",
#   "glp @ git+https://github.com/kabylda/glp.git@f6955d50b34b352b2ea27b0ac1264909f4cde278",
#   "e3x",
#   "jax-pme",
#   "flax",
#   "jax[cuda12]",
#   "ml-collections",
#   "orbax-checkpoint",
#   "pyyaml",
# ]
# ///
# This script computes SO3LR reference energies from the upstream SO3LR calculator.
#
# Run with:
#   uv run --script test/data/so3lr_references.py

from pathlib import Path

import ase.io
import numpy as np
from openmm import unit
from so3lr import So3lrCalculator

DATA_DIR = Path(__file__).resolve().parent
EV_TO_KJMOL = (unit.elementary_charge * unit.volt * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
    unit.kilojoules_per_mole
)
EV_A_TO_KJMOL_A = (
    unit.elementary_charge * unit.volt / unit.angstrom * unit.AVOGADRO_CONSTANT_NA
).value_in_unit(unit.kilojoules_per_mole / unit.angstrom)
MODEL = "so3lr"
SYSTEMS = {
    "toluene": DATA_DIR / "toluene" / "toluene.pdb",
    "alanine-dipeptide-explicit": DATA_DIR
    / "alanine-dipeptide"
    / "alanine-dipeptide-explicit.pdb",
}


def calculate_reference(path: Path, *, include_forces: bool = True) -> dict[str, float | np.ndarray]:
    atoms = ase.io.read(path)
    atoms.info["charge"] = 0.0
    atoms.calc = So3lrCalculator(
        calculate_stress=False,
        lr_cutoff=12.0,
        dtype=np.float32,
    )
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
