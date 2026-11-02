import subprocess
import sys

HEAVY = ("mrh", "pyscf", "qiskit", "qiskit_addon_sqd")


def test_records_import_without_heavy_dependencies(tmp_path):
    # A fresh interpreter, as in an environment without mrh or Qiskit installed.
    code = f"""
import sys
import numpy as np
import lassqd
import lassqd._hdf5
from lassqd.records import LASState

state = LASState(np.eye(2), [np.zeros((2, 1, 1))], [np.zeros((1, 1, 1, 1))])
state.save({str(tmp_path / "state.h5")!r})
LASState.load({str(tmp_path / "state.h5")!r})
loaded = sorted(m for m in {HEAVY!r} if m in sys.modules)
assert not loaded, loaded
"""
    subprocess.run([sys.executable, "-c", code], check=True)
