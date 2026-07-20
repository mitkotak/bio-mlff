"""Unit conversions shared by the upstream reference generators."""

from openmm import unit

EV_TO_KJMOL = (unit.elementary_charge * unit.volt * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
    unit.kilojoules_per_mole
)
EV_A_TO_KJMOL_A = (
    unit.elementary_charge * unit.volt / unit.angstrom * unit.AVOGADRO_CONSTANT_NA
).value_in_unit(unit.kilojoules_per_mole / unit.angstrom)
HARTREE_TO_KJMOL = (unit.hartree * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
    unit.kilojoules_per_mole
)
HARTREE_A_TO_KJMOL_A = (unit.hartree * unit.AVOGADRO_CONSTANT_NA / unit.angstrom).value_in_unit(
    unit.kilojoules_per_mole / unit.angstrom
)
