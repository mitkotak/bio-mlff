# /// script
# dependencies = [
#   "ase",
#   "huggingface-hub",
#   "numpy",
#   "openmm",
#   "torchmd-net @ git+https://github.com/torchmd/torchmd-net.git@2a2c913352a8b5fd297a33dd0ec35c4d69fb1eea",
# ]
# ///

from pathlib import Path

import ase.io
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from reference_utils import EV_A_TO_KJMOL_A, EV_TO_KJMOL
from torchmdnet.models.model import load_model

DATA_DIR = Path(__file__).resolve().parent
SYSTEMS = {
    "toluene": DATA_DIR / "toluene" / "toluene.pdb",
    "alanine-dipeptide-explicit": DATA_DIR
    / "alanine-dipeptide"
    / "alanine-dipeptide-explicit.pdb",
}
MODELS = {
    "aceff-jax-1.1": ("Acellera/AceFF-1.1", "aceff_v1.1.ckpt", {}),
    "aceff-jax-2.0": (
        "Acellera/AceFF-2.0",
        "aceff_v2.0.ckpt",
        {"coulomb_cutoff": 12.0},
    ),
}


def calculate_reference(
    structure_path: Path,
    model: torch.nn.Module,
) -> dict[str, float | np.ndarray]:
    atoms = ase.io.read(structure_path)
    positions = torch.as_tensor(atoms.positions, device="cuda", dtype=torch.float64)
    atomic_numbers = torch.as_tensor(atoms.numbers, device="cuda", dtype=torch.long)
    batch = torch.zeros(len(atoms), device="cuda", dtype=torch.long)
    charge = torch.zeros(1, device="cuda", dtype=torch.float64)
    box_vectors = (
        torch.as_tensor(atoms.cell.array, device="cuda", dtype=torch.float64)
        if atoms.pbc.any()
        else None
    )
    energy, forces = model(
        atomic_numbers,
        positions,
        batch=batch,
        q=charge,
        box=box_vectors,
    )
    return {
        "energy": np.float64(energy.detach().sum().cpu().item() * EV_TO_KJMOL).item(),
        "forces": np.asarray(
            forces.detach().cpu().numpy() * EV_A_TO_KJMOL_A,
            dtype=np.float64,
        ),
    }


def main() -> None:
    results: dict[str, float | np.ndarray] = {}
    for model_name, (repo_id, filename, model_kwargs) in MODELS.items():
        checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
        model = load_model(
            checkpoint_path,
            device="cuda",
            derivative=True,
            precision=64,
            remove_ref_energy=True,
            max_num_neighbors=64,
            static_shapes=False,
            **model_kwargs,
        ).eval()
        model.requires_grad_(False)
        for system_name, structure_path in SYSTEMS.items():
            reference = calculate_reference(structure_path, model)
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
