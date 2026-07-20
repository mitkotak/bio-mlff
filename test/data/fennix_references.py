# /// script
# dependencies = ["ase", "fennol", "jax[cuda13]", "numpy", "openmm", "openmmml>=1.7"]
# ///

from pathlib import Path

import ase.io
import jax
import numpy as np
from fennol.ase import FENNIXCalculator
from openmmml.models.fennixpotential import FeNNixPotentialImpl as OpenMMFeNNixPotentialImpl
from reference_utils import EV_A_TO_KJMOL_A, EV_TO_KJMOL

DATA_DIR = Path(__file__).resolve().parent
SYSTEMS = {
    "toluene": DATA_DIR / "toluene" / "toluene.pdb",
    "methanol-ions": DATA_DIR / "methanol-ions" / "methanol-ions.pdb",
    "alanine-dipeptide-explicit": DATA_DIR
    / "alanine-dipeptide"
    / "alanine-dipeptide-explicit.pdb",
}
UPSTREAM_MODELS = OpenMMFeNNixPotentialImpl.KNOWN_MODELS
MODELS_BY_SYSTEM = {
    "toluene": ("fennix-bio1-small", "fennix-bio1-medium"),
    "methanol-ions": tuple(UPSTREAM_MODELS),
    "alanine-dipeptide-explicit": ("fennix-bio1-small",),
}
ION_CHARGES = [0, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1]


def model_path(model_name: str) -> str:
    model_url, _ = UPSTREAM_MODELS[model_name]
    return OpenMMFeNNixPotentialImpl(model_name, None)._downloadOrFindFile(
        f"{model_name}.fnx", model_url
    )


def calculate_reference(
    structure_path: Path,
    model_name: str,
) -> dict[str, float | np.ndarray]:
    atoms = ase.io.read(structure_path)
    if structure_path == SYSTEMS["methanol-ions"]:
        atoms.set_initial_charges(ION_CHARGES)
    with jax.enable_x64(True):
        atoms.calc = FENNIXCalculator(model_path(model_name), use_float64=True)
        return {
            "energy": np.float64(atoms.get_potential_energy() * EV_TO_KJMOL).item(),
            "forces": np.asarray(
                atoms.get_forces() * EV_A_TO_KJMOL_A,
                dtype=np.float64,
            ),
        }


def main() -> None:
    results: dict[str, float | np.ndarray] = {}
    for system_name, model_names in MODELS_BY_SYSTEM.items():
        for model_name in model_names:
            reference = calculate_reference(SYSTEMS[system_name], model_name)
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
