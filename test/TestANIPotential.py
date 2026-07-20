import os

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest
from openmmml import MLPotential

equinox = pytest.importorskip("equinox", reason="equinox is not installed")

import biomlff.anipotential  # noqa: E402,F401
from biomlff.ani import ANI2X_MODEL_PATHS  # noqa: E402

cuda_platform = mm.Platform.getPlatformByName("CUDA")
pytestmark = pytest.mark.skipif(cuda_platform is None, reason="CUDA platform is not available")

# Generated independently by test/data/ani_references.py on CUDA in float64.
ENERGIES = {
    "toluene": {
        "ani2x-jax-model0": -712776.3721682105,
        "ani2x-jax-ensemble": -712776.6445064595,
    },
}
FORCES = {
    "toluene": {
        "ani2x-jax-model0": np.array([[  24.46120266872006   , -115.9055403345898    ,   86.348964384648     ],
 [ -12.139412984672548  ,   49.68856363363853   ,  -26.058718847496117  ],
 [  89.50539021849171   ,   76.14466619115395   ,   12.853976516588936  ],
 [  -4.0975173171909685 ,   87.6135439675488    ,    8.950148975031233  ],
 [ -25.41510439614252   ,   20.25572786973515   ,    0.27613099252430695],
 [ -76.1196849485012    , -124.91341522917605   ,  -13.781740193737217  ],
 [   3.8669542841705455 ,   -5.002044184777637  ,    1.257796716710591  ],
 [  17.531910179302713  ,   36.585389483902794  ,  -52.14007318263866   ],
 [  39.46128760695432   ,   -2.640424636397467  ,  -22.71030353392475   ],
 [  -3.3672060384352815 ,   64.68166635819226   ,   16.346034491230743  ],
 [ -38.199805670017334  ,  -11.345839458093863  ,   -3.468504574394377  ],
 [  -1.3362907785456113 ,  -38.758286772109415  ,   -3.8766208704938077 ],
 [  -4.257747565863926  ,  -19.705909437633775  ,   -1.8686364009951515 ],
 [   6.617760085784963  ,   16.97272748437646   ,    1.6544215826978466 ],
 [ -16.51173534405491   ,  -33.67082493577      ,   -3.782876055751584  ]], dtype=np.float64),
        "ani2x-jax-ensemble": np.array([[ 2.8668967613700588e+01, -1.1419465652328087e+02,
   8.6121664685976327e+01],
 [-1.4649727478721127e+01,  4.6918789777687920e+01,
  -2.5216266410595619e+01],
 [ 8.9526937624571346e+01,  7.8152700696092182e+01,
   1.2592046589048367e+01],
 [-8.1744579514990445e-01,  8.9341716523143546e+01,
   9.0875446404253601e+00],
 [-3.3189886827546736e+01,  1.9887881325674588e+01,
  -5.1341353808798944e-02],
 [-7.5863269261886558e+01, -1.2539726212485584e+02,
  -1.3954032174934065e+01],
 [ 5.5170334185644698e+00, -5.6501775098309182e+00,
   9.4085170541491470e-01],
 [ 1.7824238445432076e+01,  3.5566099488133673e+01,
  -5.1629525484441935e+01],
 [ 3.7942145082714767e+01, -3.7458929143152959e+00,
  -2.3028536995781064e+01],
 [-3.5936551883049543e+00,  6.4428338549864051e+01,
   1.6130701944774504e+01],
 [-3.8855703800772972e+01, -1.1337192522479096e+01,
  -3.4425801335944031e+00],
 [-7.5639201310076287e-01, -4.0091141834223322e+01,
  -3.9922307041311216e+00],
 [-3.1168844349216083e+00, -2.0056436203677436e+01,
  -1.8574874692219621e+00],
 [ 7.5026519132706682e+00,  1.8180674556035061e+01,
   1.8060311431512011e+00],
 [-1.6139009297849299e+01, -3.2003441283968201e+01,
  -3.5068399822816803e+00]], dtype=np.float64),
    },
}

test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ani_model_names = ("ani2x-jax-model0", "ani2x-jax-ensemble")
available_models = [model for model in ani_model_names if ANI2X_MODEL_PATHS[model].is_file()]
PURE_FORCE_ATOL = 3e-6

@pytest.mark.parametrize("model", ani_model_names)
class TestANIPotential:
    def testCreatePureMLSystem(self, model):
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        potential = MLPotential(model)
        system = potential.createSystem(pdb.topology, preprocessing_positions=pdb.positions, use_float64=True)
        context = mm.Context(system, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "double"})
        context.setPositions(pdb.positions)
        state = context.getState(energy=True, forces=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.angstrom)
        np.testing.assert_allclose(ENERGIES["toluene"][model], energy, rtol=1e-10)
        np.testing.assert_allclose(
            FORCES["toluene"][model],
            forces,
            rtol=1e-10,
            atol=PURE_FORCE_ATOL,
        )

    def testSimulate(self, model):
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        potential = MLPotential(model)
        system = potential.createSystem(pdb.topology, preprocessing_positions=pdb.positions, use_float64=True)
        integrator = mm.LangevinIntegrator(300.0, 1.0, 0.001)
        context = mm.Context(system, integrator, cuda_platform, {"DeviceIndex": "0", "Precision": "double"})
        context.setPositions(pdb.positions)
        integrator.step(10)
        positions = context.getState(positions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        assert np.all(np.isfinite(positions))

    def testCreateMixedSystem(self, model, assert_mixed_system_interpolation):
        pdb = app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))
        ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        mmSystem = ff.createSystem(pdb.topology, nonbondedMethod=app.PME)
        potential = MLPotential(model)
        mlAtoms = [atom.index for atom in next(pdb.topology.chains()).atoms()]
        mixedSystem = potential.createMixedSystem(pdb.topology, mmSystem, mlAtoms, interpolate=False, preprocessing_positions=pdb.positions, use_float64=True)
        interpSystem = potential.createMixedSystem(pdb.topology, mmSystem, mlAtoms, interpolate=True, preprocessing_positions=pdb.positions, use_float64=True)
        mmContext = mm.Context(mmSystem, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "double"})
        mixedContext = mm.Context(mixedSystem, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "double"})
        interpContext = mm.Context(interpSystem, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "double"})
        mmContext.setPositions(pdb.positions)
        mixedContext.setPositions(pdb.positions)
        interpContext.setPositions(pdb.positions)
        assert_mixed_system_interpolation(mmContext, mixedContext, interpContext)
