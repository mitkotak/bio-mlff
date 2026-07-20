# /// script
# dependencies = ["ase", "mace-torch", "numpy", "openmm"]
# ///

from pathlib import Path

import ase.io
import numpy as np
from mace.calculators.foundations_models import mace_off
from reference_utils import EV_A_TO_KJMOL_A, EV_TO_KJMOL

DATA_DIR = Path(__file__).resolve().parent
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
    structure_path: Path,
    checkpoint: str,
) -> dict[str, float | np.ndarray]:
    atoms = ase.io.read(structure_path)
    atoms.calc = mace_off(checkpoint, device="cuda", default_dtype="float64")
    return {
        "energy": np.float64(atoms.get_potential_energy() * EV_TO_KJMOL).item(),
        "forces": np.asarray(atoms.get_forces() * EV_A_TO_KJMOL_A, dtype=np.float64),
    }


def main() -> None:
    results: dict[str, float | np.ndarray] = {}
    for model_name, checkpoint in MODELS.items():
        reference = calculate_reference(SYSTEMS["toluene"], checkpoint)
        result_key = f"toluene/{model_name}"
        results[result_key] = reference["energy"]
        results[f"{result_key}/forces"] = reference["forces"]

    reference = calculate_reference(
        SYSTEMS["alanine-dipeptide-explicit"],
        MODELS["mace-jax-off-s-23"],
    )
    result_key = "alanine-dipeptide-explicit/mace-jax-off-s-23"
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
