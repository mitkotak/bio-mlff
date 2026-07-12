"""Run force-reference tests and report maximum absolute/relative errors."""

import sys
import inspect
from pathlib import Path

import numpy as np
import pytest


TESTS = [
    "test/TestAIMNet2Potential.py::TestAIMNet2::testCreatePureMLSystem",
    "test/TestANIPotential.py::TestANIPotential::testCreatePureMLSystem",
    "test/TestAceFFPotential.py::TestAceFF::testCreatePureMLSystem",
    # "test/TestFeNNixPotential.py::TestFeNNix::testCreatePureMLSystem",
    "test/TestMACEPotential.py::TestMACE::testCreatePureMLSystem",
    # "test/TestORBPotential.py::TestORB::testCreatePureMLSystem",
    # "test/TestSO3LRPotential.py::TestSO3LR::testCreatePureMLSystem",
]


def print_error(reference, calculated):
    frame = next(frame for frame in inspect.stack() if frame.function.startswith("test"))
    test = f"{Path(frame.filename).name}::{frame.function}"
    if "model" in frame.frame.f_locals:
        test += f"[{frame.frame.f_locals['model']}]"
    reference = np.asarray(reference, dtype=float)
    calculated = np.asarray(calculated, dtype=float)
    difference = np.abs(reference - calculated)
    relative = difference / np.maximum(np.abs(reference), np.finfo(float).tiny)
    shape = "x".join(map(str, reference.shape))
    print(f"{test:90} {difference.max():12.6g} {relative.max():12.6g}")


def run_pytest():
    assert_allclose = np.testing.assert_allclose

    def compare(reference, calculated, *args, **kwargs):
        if isinstance(reference, np.ndarray) and reference.ndim > 1 and reference.shape[-1] == 3:
            print_error(reference, calculated)
        assert_allclose(reference, calculated, *args, **kwargs)

    print(f"{'test':90} {'max abs':>12} {'max rel':>12}")
    np.testing.assert_allclose = compare
    try:
        args = sys.argv[1:] or [
            "-q",
            "-s",
            "--tb=no",
            "--no-summary",
            "--disable-warnings",
            *TESTS,
        ]
        return pytest.main(args)
    finally:
        np.testing.assert_allclose = assert_allclose


if __name__ == "__main__":
    raise SystemExit(run_pytest())
