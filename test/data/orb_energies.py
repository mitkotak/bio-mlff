# /// script
# dependencies = ["ase", "numpy", "openmm", "orb-models"]
# ///
# This script computes reference energies and forces for the ORB conservative OMOL model.

from pathlib import Path

import ase.io
import numpy as np
from openmm import unit
from orb_models.forcefield import pretrained as orb

try:
    from orb_models.forcefield.inference.calculator import ORBCalculator
except ModuleNotFoundError:
    from orb_models.forcefield.calculator import ORBCalculator

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


MODEL = "orb-v3-conservative-omol"
MODEL_PRECISION = "float32-highest"


def make_calculator():
    pretrained = orb.ORB_PRETRAINED_MODELS[MODEL](precision=MODEL_PRECISION)
    if isinstance(pretrained, tuple):
        orbff, atoms_adapter = pretrained
        return ORBCalculator(orbff, atoms_adapter=atoms_adapter)
    return ORBCalculator(pretrained)


def calculate_reference(
    path: Path,
    charge: int,
    spin: int,
    *,
    include_forces: bool = True,
) -> dict[str, float | np.ndarray]:
    atoms = ase.io.read(path)
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    atoms.calc = make_calculator()
    reference: dict[str, float | np.ndarray] = {
        "energy": atoms.get_potential_energy() * EV_TO_KJMOL,
    }
    if include_forces:
        reference["forces"] = atoms.get_forces() * EV_A_TO_KJMOL_NM
    return reference


def calculate_results() -> dict[str, float | np.ndarray]:
    results = {}
    reference = calculate_reference(SYSTEMS["toluene"], charge=0, spin=1)
    results[f"toluene/{MODEL}"] = reference["energy"]
    results[f"toluene/{MODEL}/forces"] = reference["forces"]
    reference = calculate_reference(
        SYSTEMS["toluene"],
        charge=-1,
        spin=3,
        include_forces=False,
    )
    results[f"toluene/{MODEL}/override-charge-spin"] = reference["energy"]
    reference = calculate_reference(
        SYSTEMS["alanine-dipeptide-explicit"],
        charge=0,
        spin=1,
    )
    results[f"alanine-dipeptide-explicit/{MODEL}"] = reference["energy"]
    results[f"alanine-dipeptide-explicit/{MODEL}/forces"] = reference["forces"]
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
