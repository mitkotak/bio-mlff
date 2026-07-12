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

ENERGIES = {
    "toluene": {
        "ani2x-jax-model0": -712776.3979138145,
        "ani2x-jax-ensemble": -712776.6702520634,
    },
}
FORCES = {
    "toluene": {
        "ani2x-jax-model0": np.array(
                                [[  24.46201   , -115.90563   ,   86.3488    ],
                                 [ -12.139924  ,   49.688004  ,  -26.058739  ],
                                 [  89.506256  ,   76.14513   ,   12.854026  ],
                                 [  -4.098486  ,   87.61389   ,    8.950142  ],
                                 [ -25.4153    ,   20.255814  ,    0.27614233],
                                 [ -76.11981   , -124.913376  ,  -13.781749  ],
                                 [   3.8670928 ,   -5.0013857 ,    1.2578658 ],
                                 [  17.531755  ,   36.585346  ,  -52.140007  ],
                                 [  39.461243  ,   -2.6403558 ,  -22.710318  ],
                                 [  -3.3673573 ,   64.68177   ,   16.34613   ],
                                 [ -38.20007   ,  -11.346236  ,   -3.4685426 ],
                                 [  -1.3361334 ,  -38.758736  ,   -3.876657  ],
                                 [  -4.2574987 ,  -19.705889  ,   -1.8686271 ],
                                 [   6.617785  ,   16.97285   ,    1.6544379 ],
                                 [ -16.511541  ,  -33.671185  ,   -3.7829    ]]
                            ),
        "ani2x-jax-ensemble": np.array(
                                  [[ 2.8669722e+01, -1.1419479e+02,  8.6121422e+01],
                                   [-1.4650016e+01,  4.6918201e+01, -2.5216291e+01],
                                   [ 8.9527527e+01,  7.8153343e+01,  1.2592115e+01],
                                   [-8.1814009e-01,  8.9341965e+01,  9.0875368e+00],
                                   [-3.3190250e+01,  1.9888214e+01, -5.1324479e-02],
                                   [-7.5863319e+01, -1.2539761e+02, -1.3954075e+01],
                                   [ 5.5170174e+00, -5.6495318e+00,  9.4091415e-01],
                                   [ 1.7824089e+01,  3.5566063e+01, -5.1629471e+01],
                                   [ 3.7942101e+01, -3.7458200e+00, -2.3028543e+01],
                                   [-3.5938420e+00,  6.4428490e+01,  1.6130873e+01],
                                   [-3.8855999e+01, -1.1337622e+01, -3.4426274e+00],
                                   [-7.5626522e-01, -4.0091499e+01, -3.9922557e+00],
                                   [-3.1166177e+00, -2.0056419e+01, -1.8574795e+00],
                                   [ 7.5027556e+00,  1.8180866e+01,  1.8060628e+00],
                                   [-1.6138769e+01, -3.2003803e+01, -3.5068648e+00]]
                              ),
    },
}
test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ani_model_names = ("ani2x-jax-model0", "ani2x-jax-ensemble")
available_models = [model for model in ani_model_names if ANI2X_MODEL_PATHS[model].is_file()]

@pytest.mark.parametrize("model", ani_model_names)
class TestANIPotential:
    def testCreatePureMLSystem(self, model):
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        potential = MLPotential(model)
        system = potential.createSystem(pdb.topology, preprocessing_positions=pdb.positions)
        context = mm.Context(system, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "mixed"})
        context.setPositions(pdb.positions)
        state = context.getState(energy=True, forces=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.angstrom)
        np.testing.assert_allclose(ENERGIES["toluene"][model], energy, rtol=1e-5)
        np.testing.assert_allclose(FORCES["toluene"][model], forces, rtol=1e-5, atol=1e-5)

    def testSimulate(self, model):
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        potential = MLPotential(model)
        system = potential.createSystem(pdb.topology, preprocessing_positions=pdb.positions)
        integrator = mm.LangevinIntegrator(300.0, 1.0, 0.001)
        context = mm.Context(system, integrator, cuda_platform, {"DeviceIndex": "0", "Precision": "mixed"})
        context.setPositions(pdb.positions)
        integrator.step(10)
        positions = context.getState(positions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        assert np.all(np.isfinite(positions))

    def testCreateMixedSystem(self, model):
        pdb = app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))
        ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        mmSystem = ff.createSystem(pdb.topology, nonbondedMethod=app.PME)
        potential = MLPotential(model)
        mlAtoms = [atom.index for atom in next(pdb.topology.chains()).atoms()]
        mixedSystem = potential.createMixedSystem(pdb.topology, mmSystem, mlAtoms, interpolate=False, preprocessing_positions=pdb.positions)
        interpSystem = potential.createMixedSystem(pdb.topology, mmSystem, mlAtoms, interpolate=True, preprocessing_positions=pdb.positions)
        mmContext = mm.Context(mmSystem, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "mixed"})
        mixedContext = mm.Context(mixedSystem, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "mixed"})
        interpContext = mm.Context(interpSystem, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "mixed"})
        mmContext.setPositions(pdb.positions)
        mixedContext.setPositions(pdb.positions)
        interpContext.setPositions(pdb.positions)
        mmState = mmContext.getState(energy=True, forces=True)
        mixedState = mixedContext.getState(energy=True, forces=True)
        interpState1 = interpContext.getState(energy=True, forces=True)
        interpContext.setParameter("lambda_interpolate", 0)
        interpState2 = interpContext.getState(energy=True, forces=True)
        assert np.isclose(
            mixedState.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
            interpState1.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
            rtol=1e-5,
        )
        assert np.isclose(
            mmState.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
            interpState2.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
            rtol=1e-5,
        )
        assert np.allclose(
            mixedState.getForces().value_in_unit(unit.kilojoules_per_mole / unit.angstrom),
            interpState1.getForces().value_in_unit(unit.kilojoules_per_mole / unit.angstrom),
            rtol=1e-5,
            atol=1e-5,
        )
        assert np.allclose(
            mmState.getForces().value_in_unit(unit.kilojoules_per_mole / unit.angstrom),
            interpState2.getForces().value_in_unit(unit.kilojoules_per_mole / unit.angstrom),
            rtol=1e-5,
            atol=1e-5,
        )
