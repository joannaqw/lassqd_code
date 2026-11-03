import subprocess
import sys

from conftest import full_counts

from qiskit import QuantumCircuit

from lassqd import (
    FragmentSamples,
    FragmentSQD,
    build_circuits,
    capture_embedding,
    initial_state,
)

HEAVY = ("mrh", "pyscf", "qiskit", "qiskit_addon_sqd")


def test_records_load_without_heavy_dependencies(tmp_path, h6):
    _, las, mo = h6
    embedding = capture_embedding(las, initial_state(las, mo))
    counts = [full_counts(f.norb, f.nelec) for f in embedding.fragments]
    samples = FragmentSamples(embedding, counts, ("job",))
    solver = FragmentSQD(50, max_iterations=1, seed=0)
    result = solver.solve(embedding.fragments[0], counts[0], ao_ovlp=embedding.ao_ovlp)
    records = {
        "LASState": embedding.state,
        "Fragment": embedding.fragments[0],
        "Embedding": embedding,
        "FragmentSamples": samples,
        "SQDState": solver.state,
        "FragmentResult": result,
    }
    for name, record in records.items():
        record.save(tmp_path / f"{name}.h5")

    # A fresh interpreter, as in an environment without mrh or Qiskit installed.
    code = f"""
import sys
import lassqd
from lassqd import records

for name in {list(records)!r}:
    getattr(records, name).load({str(tmp_path)!r} + f"/{{name}}.h5")
loaded = sorted(m for m in {HEAVY!r} if m in sys.modules)
assert not loaded, loaded
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_circuits_load_without_mrh_or_sqd(tmp_path, h6):
    # A QPU gateway loads the circuits, submits them and collects the samples.
    def x_builder(fragment):
        circuit = QuantumCircuit(2 * fragment.norb)
        circuit.x(0)
        return circuit

    _, las, mo = h6
    embedding = capture_embedding(las, initial_state(las, mo))
    path = tmp_path / "circuits.h5"
    build_circuits(embedding, x_builder).save(path)
    code = f"""
import sys
from lassqd.circuits import FragmentCircuits, collect_samples

FragmentCircuits.load({str(path)!r})
loaded = sorted(m for m in ("mrh", "pyscf", "qiskit_addon_sqd") if m in sys.modules)
assert not loaded, loaded
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_capture_and_replay_do_not_import_sqd():
    # SQD detects MPI when it is imported, so capture and replay workers launched
    # with several ranks must not import it.
    code = """
import sys
from lassqd import capture_embedding, initial_state, update_orbitals
assert "qiskit_addon_sqd" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
