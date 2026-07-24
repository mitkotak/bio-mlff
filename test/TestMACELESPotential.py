import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest
from openmmml import MLPotential

pytest.importorskip("equinox", reason="equinox is not installed")

import biomlff.macepotential  # noqa: E402,F401
from biomlff.mace import MACELES_MODEL_PATHS, load_model  # noqa: E402

try:
    cuda_platform = mm.Platform.getPlatformByName("CUDA")
except Exception:
    cuda_platform = None

pytestmark = pytest.mark.skipif(cuda_platform is None, reason="CUDA platform is not available")

MODEL = "maceles-jax-off-small"
test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# Generated independently by test/data/maceles_references.py on CUDA in float32.
ENERGY = np.float64(-713467.8551826022)
FORCES = np.array(
    [
        [37.73308563232422, -109.11942291259766, 80.63806915283203],
        [-19.351224899291992, 43.31661605834961, -25.17091941833496],
        [80.33866882324219, 77.29529571533203, 11.308426856994629],
        [2.2060537338256836, 84.0157699584961, 11.22201156616211],
        [-30.41483497619629, 22.79800796508789, 0.24623984098434448],
        [-67.14672088623047, -124.85470581054688, -11.681756019592285],
        [4.324480056762695, -0.6958423256874084, 1.0975780487060547],
        [9.064122200012207, 40.74252700805664, -64.44746398925781],
        [25.071258544921875, -20.919170379638672, -20.5236759185791],
        [-13.079081535339355, 71.71183013916016, 31.0998592376709],
        [-50.03254699707031, -28.833715438842773, -6.535176753997803],
        [9.202895164489746, -59.76517868041992, -5.697510719299316],
        [19.154674530029297, -21.418170928955078, -1.469578742980957],
        [19.16758918762207, 36.65605163574219, 3.62485408782959],
        [-26.238414764404297, -10.929911613464355, -3.7109522819519043],
    ],
    dtype=np.float64,
)
PERIODIC_ENERGY = np.float64(-713467.9494065593)
PERIODIC_FORCES = np.array(
    [
        [37.75658416748047, -109.11626434326172, 80.6245346069336],
        [-19.298498153686523, 43.30126190185547, -25.171558380126953],
        [80.33804321289062, 77.32357025146484, 11.31592082977295],
        [2.2168025970458984, 84.04097747802734, 11.221546173095703],
        [-30.421205520629883, 22.793367385864258, 0.24828243255615234],
        [-67.1559829711914, -124.8980941772461, -11.687713623046875],
        [4.3241472244262695, -0.6893627643585205, 1.1015568971633911],
        [9.04929256439209, 40.739009857177734, -64.46553802490234],
        [25.0523738861084, -20.933805465698242, -20.523216247558594],
        [-13.099408149719238, 71.71873474121094, 31.12407684326172],
        [-50.0764274597168, -28.837533950805664, -6.535342693328857],
        [9.215156555175781, -59.78470230102539, -5.699302673339844],
        [19.184797286987305, -21.416950225830078, -1.4692736864089966],
        [19.18309783935547, 36.67428207397461, 3.6261463165283203],
        [-26.26878547668457, -10.914512634277344, -3.7101335525512695],
    ],
    dtype=np.float64,
)
EXPANDED_ENERGY = np.float64(-713467.9494065593)
EXPANDED_FORCES = np.array(
    [
        [37.756805419921875, -109.11593627929688, 80.62510681152344],
        [-19.303903579711914, 43.30236053466797, -25.17165184020996],
        [80.34054565429688, 77.32268524169922, 11.315475463867188],
        [2.2151997089385986, 84.04086303710938, 11.221634864807129],
        [-30.424640655517578, 22.794164657592773, 0.24798063933849335],
        [-67.15707397460938, -124.89727783203125, -11.687810897827148],
        [4.325486660003662, -0.6900347471237183, 1.101011037826538],
        [9.050776481628418, 40.73857116699219, -64.46398162841797],
        [25.054157257080078, -20.93283462524414, -20.523292541503906],
        [-13.097453117370605, 71.71758270263672, 31.122478485107422],
        [-50.074459075927734, -28.836647033691406, -6.535450458526611],
        [9.214132308959961, -59.785587310791016, -5.699705600738525],
        [19.185009002685547, -21.417930603027344, -1.468984603881836],
        [19.183135986328125, 36.67525863647461, 3.6269359588623047],
        [-26.26771354675293, -10.915186882019043, -3.7097625732421875],
    ],
    dtype=np.float64,
)
TRICLINIC_ENERGY = np.float64(-713467.9494065593)
TRICLINIC_FORCES = np.array(
    [
        [37.75667953491211, -109.11588287353516, 80.62420654296875],
        [-19.298635482788086, 43.30120086669922, -25.171850204467773],
        [80.33869171142578, 77.3238754272461, 11.315659523010254],
        [2.2167601585388184, 84.04141998291016, 11.221360206604004],
        [-30.42133140563965, 22.79326629638672, 0.2485799938440323],
        [-67.15623474121094, -124.89794921875, -11.687828063964844],
        [4.324446678161621, -0.6893972754478455, 1.1017986536026],
        [9.049482345581055, 40.738731384277344, -64.46524810791016],
        [25.052133560180664, -20.93405532836914, -20.523056030273438],
        [-13.099433898925781, 71.71825408935547, 31.124126434326172],
        [-50.076690673828125, -28.837116241455078, -6.5350341796875],
        [9.21509838104248, -59.78468322753906, -5.6990203857421875],
        [19.184988021850586, -21.41748809814453, -1.4693647623062134],
        [19.18303871154785, 36.67424774169922, 3.626124143600464],
        [-26.268999099731445, -10.914429664611816, -3.710453510284424],
    ],
    dtype=np.float64,
)


class TestMACELES:
    def testFloat32Only(self):
        checkpoint = MACELES_MODEL_PATHS[MODEL]
        with checkpoint.open("rb") as handle:
            config = json.loads(handle.readline().decode("utf-8"))
            stored_dtypes = []
            while handle.tell() < checkpoint.stat().st_size:
                stored_dtypes.append(np.load(handle).dtype)
        assert config["storage_dtype"] == "float32"
        assert stored_dtypes
        assert set(stored_dtypes) == {np.dtype(np.float32)}

        model = load_model(MODEL, dtype=jnp.float32)
        floating_dtypes = {
            leaf.dtype
            for leaf in jax.tree_util.tree_leaves(model)
            if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)
        }
        assert floating_dtypes == {jnp.dtype(jnp.float32)}
        with pytest.raises(ValueError, match="MACE-LES models only support float32"):
            load_model(MODEL, dtype=jnp.float64)

    def testCreatePureMLSystem(self):
        assert MACELES_MODEL_PATHS[MODEL].is_file()
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        positions = pdb.getPositions(asNumpy=True)
        system = MLPotential(MODEL).createSystem(
            pdb.topology,
            returnEnergyType="energy",
            preprocessing_positions=positions,
            use_float64=False,
        )
        context = mm.Context(
            system,
            mm.VerletIntegrator(0.001),
            cuda_platform,
            {"DeviceIndex": "0", "Precision": "mixed"},
        )
        context.setPositions(positions)
        state = context.getState(energy=True, forces=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.angstrom
        )
        np.testing.assert_allclose(energy, ENERGY, rtol=1e-10)
        np.testing.assert_allclose(forces, FORCES, rtol=1e-10)

    def testCreateMixedSystem(self, assert_mixed_system_interpolation):
        prmtop = app.AmberPrmtopFile(
            os.path.join(test_data_dir, "toluene", "toluene-explicit.prm7")
        )
        inpcrd = app.AmberInpcrdFile(
            os.path.join(test_data_dir, "toluene", "toluene-explicit.rst7")
        )
        ml_atoms = list(range(15))
        mm_system = prmtop.createSystem(nonbondedMethod=app.PME)
        potential = MLPotential(MODEL)
        mixed_system = potential.createMixedSystem(
            prmtop.topology,
            mm_system,
            ml_atoms,
            interpolate=False,
            preprocessing_positions=inpcrd.positions,
            use_float64=False,
        )
        interp_system = potential.createMixedSystem(
            prmtop.topology,
            mm_system,
            ml_atoms,
            interpolate=True,
            preprocessing_positions=inpcrd.positions,
            use_float64=False,
        )
        context_properties = {"DeviceIndex": "0", "Precision": "mixed"}
        mm_context = mm.Context(
            mm_system,
            mm.VerletIntegrator(0.001),
            cuda_platform,
            context_properties,
        )
        mixed_context = mm.Context(
            mixed_system,
            mm.VerletIntegrator(0.001),
            cuda_platform,
            context_properties,
        )
        interp_context = mm.Context(
            interp_system,
            mm.VerletIntegrator(0.001),
            cuda_platform,
            context_properties,
        )
        mm_context.setPositions(inpcrd.positions)
        mixed_context.setPositions(inpcrd.positions)
        interp_context.setPositions(inpcrd.positions)
        assert_mixed_system_interpolation(
            mm_context,
            mixed_context,
            interp_context
        )

    def testPeriodicTriclinicAndExpandedBox(self):
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        initial_box = (
            mm.Vec3(2.0, 0.0, 0.0),
            mm.Vec3(0.0, 2.1, 0.0),
            mm.Vec3(0.0, 0.0, 2.2),
        ) * unit.nanometer
        pdb.topology.setPeriodicBoxVectors(initial_box)
        positions = pdb.getPositions(asNumpy=True)
        system = MLPotential(MODEL).createSystem(
            pdb.topology,
            returnEnergyType="energy",
            preprocessing_positions=positions,
            use_float64=False,
            periodic_box_scale=1.1,
        )
        context = mm.Context(
            system,
            mm.VerletIntegrator(0.001),
            cuda_platform,
            {"DeviceIndex": "0", "Precision": "mixed"},
        )
        context.setPositions(positions)
        for box, reference_energy, reference_forces in (
            (initial_box, PERIODIC_ENERGY, PERIODIC_FORCES),
            (
                (
                    mm.Vec3(2.2, 0.0, 0.0),
                    mm.Vec3(0.0, 2.31, 0.0),
                    mm.Vec3(0.0, 0.0, 2.42),
                )
                * unit.nanometer,
                EXPANDED_ENERGY,
                EXPANDED_FORCES,
            ),
        ):
            context.setPeriodicBoxVectors(*box)
            state = context.getState(energy=True, forces=True)
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            forces = state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoules_per_mole / unit.angstrom
            )
            np.testing.assert_allclose(
                energy,
                reference_energy,
                rtol=1e-10,
                atol=ENERGY_ATOL,
            )
            np.testing.assert_allclose(
                forces,
                reference_forces,
                rtol=1e-10,
                atol=FORCE_ATOL,
            )

        triclinic_box = (
            mm.Vec3(2.0, 0.0, 0.0),
            mm.Vec3(0.2, 2.1, 0.0),
            mm.Vec3(0.1, 0.15, 2.2),
        ) * unit.nanometer
        pdb.topology.setPeriodicBoxVectors(triclinic_box)
        triclinic_system = MLPotential(MODEL).createSystem(
            pdb.topology,
            returnEnergyType="energy",
            preprocessing_positions=positions,
            use_float64=False,
            periodic_box_scale=1.1,
        )
        triclinic_context = mm.Context(
            triclinic_system,
            mm.VerletIntegrator(0.001),
            cuda_platform,
            {"DeviceIndex": "0", "Precision": "mixed"},
        )
        triclinic_context.setPositions(positions)
        state = triclinic_context.getState(energy=True, forces=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.angstrom
        )
        np.testing.assert_allclose(
            energy,
            TRICLINIC_ENERGY,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            forces,
            TRICLINIC_FORCES,
            rtol=1e-10,
        )
