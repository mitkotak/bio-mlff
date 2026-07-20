# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "ase",
#   "e3x",
#   "flax",
#   "glp @ git+https://github.com/kabylda/glp.git@f6955d50b34b352b2ea27b0ac1264909f4cde278",
#   "jax[cuda12]",
#   "jax-pme",
#   "ml-collections",
#   "mlff @ git+https://github.com/kabylda/mlff.git@aeb80dcd208a4607c01dbac6c9574b7b32bcf93e",
#   "numpy",
#   "openmm",
#   "orbax-checkpoint",
#   "pyyaml",
#   "so3lr @ git+https://github.com/general-molecular-simulations/so3lr.git@5c6f36914bd2424563c1fd80bc21610960d947a4",
# ]
# ///

from pathlib import Path

import ase.io
import numpy as np
from jax.experimental import enable_x64
from reference_utils import EV_A_TO_KJMOL_A, EV_TO_KJMOL
from so3lr import So3lrCalculator

DATA_DIR = Path(__file__).resolve().parent
MODEL = "so3lr"
SYSTEMS = {
    "toluene": DATA_DIR / "toluene" / "toluene.pdb",
    "alanine-dipeptide-explicit": DATA_DIR
    / "alanine-dipeptide"
    / "alanine-dipeptide-explicit.pdb",
}


def calculate_reference(structure_path: Path) -> dict[str, float | np.ndarray]:
    atoms = ase.io.read(structure_path)
    atoms.info["charge"] = 0.0
    with enable_x64(True):
        atoms.calc = So3lrCalculator(
            calculate_stress=False,
            lr_cutoff=12.0,
            dtype=np.float64,
        )
        return {
            "energy": np.float64(atoms.get_potential_energy() * EV_TO_KJMOL).item(),
            "forces": np.asarray(
                atoms.get_forces() * EV_A_TO_KJMOL_A,
                dtype=np.float64,
            ),
        }


def main() -> None:
    results: dict[str, float | np.ndarray] = {}
    for system_name, structure_path in SYSTEMS.items():
        reference = calculate_reference(structure_path)
        result_key = f"{system_name}/{MODEL}"
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
