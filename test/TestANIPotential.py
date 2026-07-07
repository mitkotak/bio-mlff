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
            [
                [244.612026687197, -1159.055403345899, 863.489643846479],
                [-121.394129846718, 496.885636336388, -260.58718847496],
                [895.053902184917, 761.446661911539, 128.539765165889],
                [-40.975173171913, 876.135439675486, 89.501489750312],
                [-254.151043961403, 202.557278697354, 2.761309925244],
                [-761.196849485036, -1249.134152291756, -137.817401937373],
                [38.669542841721, -50.02044184778, 12.577967167106],
                [175.319101793027, 365.853894839027, -521.400731826387],
                [394.612876069544, -26.404246363974, -227.103035339248],
                [-33.672060384352, 646.816663581922, 163.460344912308],
                [-381.998056700173, -113.458394580939, -34.685045743944],
                [-13.362907785458, -387.582867721092, -38.766208704938],
                [-42.577475658655, -197.059094376339, -18.686364009952],
                [66.17760085785, 169.727274843761, 16.544215826978],
                [-165.117353440547, -336.708249357699, -37.828760557516],
            ]
        ),
        "ani2x-jax-ensemble": np.array(
            [
                [286.689676137, -1141.946565233, 861.21664686],
                [-146.497274787, 469.187897777, -252.162664106],
                [895.269376246, 781.527006961, 125.92046589],
                [-8.174457951, 893.417165231, 90.875446404],
                [-331.898868275, 198.878813257, -0.513413538],
                [-758.632692619, -1253.972621249, -139.540321749],
                [55.170334186, -56.501775098, 9.408517054],
                [178.242384454, 355.660994881, -516.295254844],
                [379.421450827, -37.458929143, -230.285369958],
                [-35.936551883, 644.283385499, 161.307019448],
                [-388.557038008, -113.371925225, -34.425801336],
                [-7.563920131, -400.911418342, -39.922307041],
                [-31.168844349, -200.564362037, -18.574874692],
                [75.026519133, 181.80674556, 18.060311432],
                [-161.390092978, -320.03441284, -35.068399823],
            ]
        ),
    },
}

test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ani_model_names = ("ani2x-jax-model0", "ani2x-jax-ensemble")
available_models = [model for model in ani_model_names if ANI2X_MODEL_PATHS[model].is_file()]

@pytest.mark.parametrize("model", ani_model_names)
class TestANIPotential:
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
            mixedState.getForces().value_in_unit(unit.kilojoules_per_mole / unit.nanometer),
            interpState1.getForces().value_in_unit(unit.kilojoules_per_mole / unit.nanometer),
            rtol=1e-5,
        )
        assert np.allclose(
            mmState.getForces().value_in_unit(unit.kilojoules_per_mole / unit.nanometer),
            interpState2.getForces().value_in_unit(unit.kilojoules_per_mole / unit.nanometer),
            rtol=1e-5,
        )
