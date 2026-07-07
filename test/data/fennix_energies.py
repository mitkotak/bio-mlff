# /// script
# dependencies = ["ase", "fennol", "numpy", "openmm"]
# ///
# This script computes reference energies for the FeNNix models.

import os
import tempfile
import urllib.request
from pathlib import Path

import ase.io
import numpy as np
from fennol.ase import FENNIXCalculator
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
    "methanol-ions": DATA_DIR / "methanol-ions" / "methanol-ions.pdb",
    "alanine-dipeptide-explicit": DATA_DIR
    / "alanine-dipeptide"
    / "alanine-dipeptide-explicit.pdb",
}
MODEL_URLS = {
    "fennix-bio1-small": "https://raw.githubusercontent.com/FeNNol-tools/FeNNol-PMC/main/FENNIX-BIO1/v1.0/fennix-bio1S.fnx",
    "fennix-bio1-medium": "https://raw.githubusercontent.com/FeNNol-tools/FeNNol-PMC/main/FENNIX-BIO1/v1.0/fennix-bio1M.fnx",
    "fennix-bio1-small-finetune-ions": "https://raw.githubusercontent.com/FeNNol-tools/FeNNol-PMC/main/FENNIX-BIO1/v1.0-finetuneIons/fennix-bio1S-finetuneIons.fnx",
    "fennix-bio1-medium-finetune-ions": "https://raw.githubusercontent.com/FeNNol-tools/FeNNol-PMC/main/FENNIX-BIO1/v1.0-finetuneIons/fennix-bio1M-finetuneIons.fnx",
}
TOLUENE_MODELS = ["fennix-bio1-small", "fennix-bio1-medium"]
ION_MODELS = [
    "fennix-bio1-small",
    "fennix-bio1-medium",
    "fennix-bio1-small-finetune-ions",
    "fennix-bio1-medium-finetune-ions",
]
ALANINE_DIPEPTIDE_MODELS = ["fennix-bio1-small"]
ION_CHARGES = [0, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1]


def model_path(model: str) -> str:
    default_cache_dir = Path(tempfile.gettempdir()) / "openmm-jax-fennix-models"
    cache_dir = Path(os.environ.get("FENNIX_MODEL_DIR", default_cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / Path(MODEL_URLS[model]).name
    if not path.is_file():
        urllib.request.urlretrieve(MODEL_URLS[model], path)
    return str(path)


def calculate_reference(
    path: Path,
    model_name: str,
    charges: list[int] | None = None,
    *,
    include_forces: bool = True,
) -> dict[str, float | np.ndarray]:
    atoms = ase.io.read(path)
    if charges is not None:
        atoms.set_initial_charges(charges)
    atoms.calc = FENNIXCalculator(model_path(model_name), use_float64=True)
    reference: dict[str, float | np.ndarray] = {
        "energy": atoms.get_potential_energy() * EV_TO_KJMOL,
    }
    if include_forces:
        reference["forces"] = atoms.get_forces() * EV_A_TO_KJMOL_NM
    return reference


def calculate_results() -> dict[str, float | np.ndarray]:
    results = {}

    for model_name in TOLUENE_MODELS:
        reference = calculate_reference(
            SYSTEMS["toluene"],
            model_name,
        )
        results[f"toluene/{model_name}"] = reference["energy"]
        results[f"toluene/{model_name}/forces"] = reference["forces"]

    for model_name in ION_MODELS:
        reference = calculate_reference(
            SYSTEMS["methanol-ions"],
            model_name,
            charges=ION_CHARGES,
        )
        results[f"methanol-ions/{model_name}"] = reference["energy"]
        results[f"methanol-ions/{model_name}/forces"] = reference["forces"]

    for model_name in ALANINE_DIPEPTIDE_MODELS:
        reference = calculate_reference(
            SYSTEMS["alanine-dipeptide-explicit"],
            model_name,
        )
        results[f"alanine-dipeptide-explicit/{model_name}"] = reference["energy"]
        results[f"alanine-dipeptide-explicit/{model_name}/forces"] = reference["forces"]

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
