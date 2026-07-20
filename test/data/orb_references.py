# /// script
# dependencies = ["ase", "numpy", "openmm", "orb-models==0.5.5"]
# ///

from pathlib import Path

import ase.io
import numpy as np
from orb_models.forcefield import pretrained as orb
from orb_models.forcefield.calculator import ORBCalculator
from reference_utils import EV_A_TO_KJMOL_A, EV_TO_KJMOL

DATA_DIR = Path(__file__).resolve().parent
MODEL = "orb-jax-v3-conservative-omol"
UPSTREAM_MODEL = "orb-v3-conservative-omol"
OVERRIDE_MODEL = f"{MODEL}/override-charge-spin"
SYSTEMS = {
    "toluene": DATA_DIR / "toluene" / "toluene.pdb",
    "alanine-dipeptide-explicit": DATA_DIR
    / "alanine-dipeptide"
    / "alanine-dipeptide-explicit.pdb",
}


def calculate_reference(
    structure_path: Path,
    calculator: ORBCalculator,
    *,
    charge: int,
    spin: int,
    include_forces: bool = True,
) -> dict[str, float | np.ndarray]:
    atoms = ase.io.read(structure_path)
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    calculator.reset()
    atoms.calc = calculator
    reference: dict[str, float | np.ndarray] = {
        "energy": np.float64(atoms.get_potential_energy() * EV_TO_KJMOL).item(),
    }
    if include_forces:
        reference["forces"] = np.asarray(
            atoms.get_forces() * EV_A_TO_KJMOL_A,
            dtype=np.float64,
        )
    return reference


def main() -> None:
    model = orb.ORB_PRETRAINED_MODELS[UPSTREAM_MODEL](
        precision="float64",
        compile=False,
        device="cuda",
    )
    calculator = ORBCalculator(model, device="cuda")
    results: dict[str, float | np.ndarray] = {}

    reference = calculate_reference(
        SYSTEMS["toluene"],
        calculator,
        charge=0,
        spin=1,
    )
    result_key = f"toluene/{MODEL}"
    results[result_key] = reference["energy"]
    results[f"{result_key}/forces"] = reference["forces"]

    reference = calculate_reference(
        SYSTEMS["toluene"],
        calculator,
        charge=-1,
        spin=3,
        include_forces=False,
    )
    results[f"toluene/{OVERRIDE_MODEL}"] = reference["energy"]

    reference = calculate_reference(
        SYSTEMS["alanine-dipeptide-explicit"],
        calculator,
        charge=0,
        spin=1,
    )
    result_key = f"alanine-dipeptide-explicit/{MODEL}"
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
