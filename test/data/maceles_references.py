# /// script
# dependencies = [
#   "ase",
#   "les @ git+https://github.com/ChengUCB/les.git",
#   "mace-torch",
#   "numpy",
#   "openmm",
# ]
# ///
# Generate MACELES-OFF references directly from upstream in CUDA float32.

import tempfile
import urllib.request
from pathlib import Path

import ase.io
import numpy as np
from mace.calculators import MACECalculator
from reference_utils import EV_A_TO_KJMOL_A, EV_TO_KJMOL

DATA_DIR = Path(__file__).resolve().parent
CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/ChengUCB/les_fit/main/"
    "MACELES-OFF/MACELES-OFF_small_converted.model"
)
CONFIGURATIONS = {
    "nonperiodic": None,
    "periodic": np.diag([20.0, 21.0, 22.0]),
    "expanded": np.diag([22.0, 23.1, 24.2]),
    "triclinic": np.array(
        [
            [20.0, 0.0, 0.0],
            [2.0, 21.0, 0.0],
            [1.0, 1.5, 22.0],
        ]
    ),
}


def calculate_reference(
    calculator: MACECalculator,
    box_vectors: np.ndarray | None,
) -> dict[str, float | np.ndarray]:
    atoms = ase.io.read(DATA_DIR / "toluene" / "toluene.pdb")
    if box_vectors is not None:
        atoms.set_cell(box_vectors)
        atoms.pbc = True
    atoms.calc = calculator
    return {
        "energy": np.float64(atoms.get_potential_energy() * EV_TO_KJMOL).item(),
        "forces": np.asarray(
            atoms.get_forces() * EV_A_TO_KJMOL_A,
            dtype=np.float64,
        ),
    }


def main() -> None:
    checkpoint = Path(tempfile.gettempdir()) / "MACELES-OFF_small_converted.model"
    if not checkpoint.is_file():
        urllib.request.urlretrieve(CHECKPOINT_URL, checkpoint)
    calculator = MACECalculator(
        model_paths=checkpoint,
        device="cuda",
        default_dtype="float32",
    )

    results: dict[str, dict[str, float | np.ndarray]] = {}
    for configuration_name, box_vectors in CONFIGURATIONS.items():
        results[configuration_name] = calculate_reference(calculator, box_vectors)

    for configuration_name, reference in results.items():
        print(f"{configuration_name}/energy: {reference['energy']!r}")
        forces = np.array2string(
            reference["forces"],
            precision=17,
            separator=", ",
            threshold=np.inf,
        )
        print(f"{configuration_name}/forces: np.array({forces}, dtype=np.float64)")


if __name__ == "__main__":
    main()
