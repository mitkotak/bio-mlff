# /// script
# dependencies = [
#   "aimnet @ git+https://github.com/isayevlab/aimnetcentral.git@dced9686f2e0000ebc179dd9986c491bf16044d8",
#   "ase",
#   "numpy",
#   "openmm",
#   "torch==2.8.0",
# ]
# ///

from pathlib import Path

import ase.io
import numpy as np
import torch
from reference_utils import EV_A_TO_KJMOL_A, EV_TO_KJMOL

# AIMNet2Base captures this value when aimnet.models is imported.
torch.set_default_dtype(torch.float64)

import aimnet.models as aimnet_models  # noqa: E402
from aimnet.calculators import AIMNet2Calculator  # noqa: E402
from aimnet.calculators.model_registry import get_model_path  # noqa: E402
from aimnet.config import build_module  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent
MODEL = "aimnet2-jax"
SYSTEMS = {
    "toluene": DATA_DIR / "toluene" / "toluene.pdb",
    "alanine-dipeptide-explicit": DATA_DIR
    / "alanine-dipeptide"
    / "alanine-dipeptide-explicit.pdb",
}


def calculate_reference(
    structure_path: Path,
    calculator: AIMNet2Calculator,
) -> dict[str, float | np.ndarray]:
    atoms = ase.io.read(structure_path)
    model_input = {
        "coord": torch.as_tensor(atoms.positions, device="cuda", dtype=torch.float64),
        "numbers": torch.as_tensor(atoms.numbers, device="cuda", dtype=torch.long),
        "charge": torch.zeros(1, device="cuda", dtype=torch.float64),
    }
    if atoms.pbc.any():
        model_input["cell"] = torch.as_tensor(
            atoms.cell.array,
            device="cuda",
            dtype=torch.float64,
        )
        calculator.set_lrcoulomb_method("dsf", cutoff=15.0)

    # Upstream prepare_input() first hardcodes floating tensors to float32, so
    # run its remaining preparation stages on the float64 tensors directly.
    prepared = calculator.mol_flatten(model_input)
    if prepared["coord"].ndim == 2:
        prepared = calculator.make_nbmat(prepared)
        prepared = calculator.pad_input(prepared)
    prepared = calculator.set_grad_tensors(prepared, forces=True)
    prepared = calculator.model(prepared)
    prepared = calculator.get_derivatives(prepared, forces=True)
    output = calculator.process_output(prepared)
    energy = output["energy"].sum()
    forces = output["forces"]
    return {
        "energy": np.float64(energy.detach().cpu().item() * EV_TO_KJMOL).item(),
        "forces": np.asarray(
            forces.detach().cpu().numpy() * EV_A_TO_KJMOL_A,
            dtype=np.float64,
        ),
    }


def main() -> None:
    checkpoint = torch.jit.load(get_model_path("aimnet2"), map_location="cuda")
    config_path = Path(aimnet_models.__file__).with_name("aimnet2_dftd3_wb97m.yaml")
    # Rebuild model in float64
    model = build_module(str(config_path)).to(device="cuda", dtype=torch.float64)
    model.load_state_dict(checkpoint.state_dict(), strict=False)
    model.eval()
    model.cutoff = float(model.aev.rc_s)
    model.cutoff_lr = float("inf")
    calculator = AIMNet2Calculator(model)

    results: dict[str, float | np.ndarray] = {}
    for system_name, structure_path in SYSTEMS.items():
        reference = calculate_reference(structure_path, calculator)
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
