"""The molecular problem: the H20 chain, its LAS object, circuits and compilation.

This is the H20 chain of ``docs/lassqd.ipynb``: two H10 fragments, each ten
electrons in ten spatial orbitals, in STO-6G.

This module imports mrh, PySCF, ffsim and Qiskit. It must **never** import
``qiskit_addon_sqd``: the capture and replay workers import it, and SQD
initializes MPI at import time whenever a launcher variable such as
``SLURM_NTASKS`` is set and mpi4py is installed. ``tests/test_imports.py`` makes
that a rule of the library; this example keeps it by putting everything SQD
needs in ``sqd_settings.py``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import ffsim
import numpy as np
from lassqd import LASSCF, Fragment
from pyscf import cc, gto, lib, scf
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

import settings

# Basis gates of the "unmapped" compilation target: a universal set on virtual
# qubits, with no coupling map, so no layout or routing is applied.
UNMAPPED_BASIS_GATES = ("id", "rz", "sx", "x", "cx")


def fragment_atoms() -> tuple[list[int], ...]:
    """Zero-based atom indices of each fragment."""
    size = settings.FRAGMENT_ATOMS
    return tuple(list(range(i * size, (i + 1) * size)) for i in range(settings.NFRAGS))


def build_molecule() -> gto.Mole:
    """The hydrogen chain: fragments of equally spaced atoms, separated by a gap."""
    positions: list[float] = []
    for _ in range(settings.NFRAGS):
        start = 0.0 if not positions else positions[-1] + settings.GAP
        positions += [
            start + settings.SPACING * i for i in range(settings.FRAGMENT_ATOMS)
        ]
    return gto.M(
        atom=[("H", (x, 0.0, 0.0)) for x in positions],
        unit="Angstrom",
        basis=settings.BASIS,
        charge=0,
        spin=0,
        max_memory=settings.MAX_MEMORY,
        verbose=0,
    )


def build_mean_field(
    mol: gto.Mole, chkfile: str | os.PathLike | None = None
) -> scf.hf.RHF:
    """RHF for ``mol``, cached in ``chkfile``.

    Without the cache every job of every macrocycle reruns RHF, and a capture
    job at 8 threads and a replay job at 16 can differ in the last bits of
    ``mf.mo_coeff`` -- around 1e-15, well inside ``lassqd.las.REPLAY_ATOL``, so
    not a failure, but the cache removes the question. Only cycle 0's capture
    writes the file, so no two jobs race for it.

    Args:
        mol: The molecule.
        chkfile: Where to cache the converged orbitals. None always solves.

    Returns:
        The converged mean field.

    Raises:
        RuntimeError: If RHF does not converge.
    """
    mf = scf.RHF(mol)
    if chkfile is not None and Path(chkfile).exists():
        stored = lib.chkfile.load(str(chkfile), "scf")
        mf.mo_coeff = np.asarray(stored["mo_coeff"])
        mf.mo_energy = np.asarray(stored["mo_energy"])
        mf.mo_occ = np.asarray(stored["mo_occ"])
        mf.e_tot = float(stored["e_tot"])
        mf.converged = True
        return mf
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge; inspect the mean-field calculation")
    if chkfile is not None:
        _save_mean_field(mf, chkfile)
    return mf


def _save_mean_field(mf: scf.hf.RHF, chkfile: str | os.PathLike) -> None:
    """Write the converged orbitals, through a temporary file and one rename."""
    chkfile = Path(chkfile)
    temporary = chkfile.with_name(f"{chkfile.name}.{os.getpid()}.tmp")
    lib.chkfile.save(
        str(temporary),
        "scf",
        {
            "e_tot": float(mf.e_tot),
            "mo_coeff": np.asarray(mf.mo_coeff),
            "mo_energy": np.asarray(mf.mo_energy),
            "mo_occ": np.asarray(mf.mo_occ),
        },
    )
    os.replace(temporary, chkfile)


@dataclass
class Problem:
    """The molecule, its mean field and the LAS object every stage rebuilds.

    Attributes:
        mol: The molecule.
        mf: The converged RHF reference.
        las: mrh's RDM-based LASSCF object for this fragmentation.
    """

    mol: gto.Mole
    mf: scf.hf.RHF
    las: object

    @cached_property
    def mo_initial(self) -> np.ndarray:
        """Localized initial orbitals, which only macrocycle 0 needs."""
        mo = self.las.localize_init_guess(fragment_atoms(), self.mf.mo_coeff)
        overlap = mo.T @ self.mf.get_ovlp() @ mo
        np.testing.assert_allclose(overlap, np.eye(mo.shape[1]), atol=1e-10, rtol=0)
        return mo

    def record(self) -> dict:
        """The contents of ``problem.json``; see :func:`write_problem_record`."""
        return {
            "nfrags": settings.NFRAGS,
            "seed": settings.SEED,
            "basis": settings.BASIS,
            "natoms": self.mol.natm,
            "nelectron": self.mol.nelectron,
            "nao": int(self.mf.mo_coeff.shape[1]),
            "ncas_sub": list(settings.NCAS_SUB),
            "nelecas_sub": [list(n) for n in settings.NELECAS_SUB],
            "spin_sub": list(settings.SPIN_SUB),
            "spacing": settings.SPACING,
            "gap": settings.GAP,
            "e_rhf": float(self.mf.e_tot),
            "module_digests": settings.module_digests(),
        }


def build_problem(chkfile: str | os.PathLike | None = None) -> Problem:
    """Build the molecule, its mean field and the LAS object.

    Args:
        chkfile: Where the mean field is cached; see :func:`build_mean_field`.

    Returns:
        The problem.
    """
    mol = build_molecule()
    mf = build_mean_field(mol, chkfile)
    las = LASSCF(
        mf, settings.NCAS_SUB, settings.NELECAS_SUB, spin_sub=settings.SPIN_SUB
    )
    return Problem(mol=mol, mf=mf, las=las)


def las_from_run(run) -> Problem:
    """:func:`build_problem` with the run's cached mean field.

    Every worker calls this: the records carry orbitals and RDMs, not the PySCF
    and mrh objects that interpret them.
    """
    return build_problem(run.mean_field())


def write_problem_record(run, problem: Problem) -> dict:
    """Write ``problem.json`` if absent, else check the run's against this build.

    Returns:
        The record on file.

    Raises:
        ValueError: If a record is already there and differs, which means the
            run directory was started from different code or settings.
    """
    path = run.problem()
    record = problem.record()
    if path.exists():
        stored = json.loads(path.read_text())
        if stored != record:
            differing = sorted(
                key
                for key in set(stored) | set(record)
                if stored.get(key) != record.get(key)
            )
            raise ValueError(
                f"{path} was written for a different problem; these differ: "
                f"{', '.join(differing)}"
            )
        return stored
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return record


def prepare_fragment(fragment: Fragment) -> QuantumCircuit:
    """Prepare this example's CCSD-initialized LUCJ state in the sampling basis.

    This is ``docs/lassqd.ipynb``'s builder: one LUCJ layer from the compressed
    double factorization of CCSD amplitudes, with nearest-neighbour same-spin
    interactions and opposite-spin pairs ``(p, p)`` for ``p = 0, 4, ...``.

    Args:
        fragment: The captured fragment, whose ``h1`` and ``h2`` are in its
            sampling basis.

    Returns:
        An unmeasured ``2 * fragment.norb``-qubit circuit with alpha orbitals on
        qubits ``0..norb-1``, in native ffsim gates.
    """
    norb, nelec = fragment.norb, fragment.nelec
    ccsd = cc.CCSD(fragment.rohf())
    ccsd.max_memory = settings.MAX_MEMORY
    ccsd.verbose = 0
    ccsd.kernel()

    interaction_pairs = (
        [(p, p + 1) for p in range(norb - 1)],  # alpha-alpha
        [(p, p) for p in range(0, norb, settings.ALPHA_BETA_STRIDE)],  # alpha-beta
        [(p, p + 1) for p in range(norb - 1)],  # beta-beta
    )
    ucj_op = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
        ccsd.t2,
        t1=ccsd.t1,
        n_reps=settings.N_REPS,
        interaction_pairs=interaction_pairs,
        optimize=True,
    )
    circuit = QuantumCircuit(2 * norb)
    # Alpha orbitals precede beta orbitals, as required by FragmentSQD.
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), circuit.qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinUnbalancedJW(ucj_op), circuit.qubits)
    return circuit


def transpile_for_target(circuit: QuantumCircuit, target, config: dict | None = None):
    """Compile ``circuit`` onto a backend ``target``: layout, routing and ISA.

    The signature is the ``(circuit, target, config)`` one that QCSC calls an
    application's ``transpiler`` hook with, and ``config`` holds only JSON-safe
    preset options, because only JSON crosses QCSC's compilation boundary. This
    example's prepare worker calls the function; adopting QRMI's Slurm path,
    where the backend ``Target`` exists only on the sampling node, means naming
    it in a ``"module:function"`` string instead.

    Args:
        circuit: A measured circuit, with native ffsim gates unless
            ``config["pre_init"]`` is false.
        target: The backend's ``qiskit.transpiler.Target``.
        config: ``optimization_level``, ``initial_layout`` as a list of physical
            qubit indices, and ``pre_init``, which runs ffsim's pre-init stage.
            On the QRMI path set it to false: the host-side ``prepare`` hook has
            already removed the ffsim gates, and the node has no ffsim.

    Returns:
        The compiled circuit.
    """
    config = dict(config or {})
    pass_manager = generate_preset_pass_manager(
        target=target,
        optimization_level=config.get("optimization_level", 3),
        initial_layout=config.get("initial_layout"),
    )
    if config.get("pre_init", True):
        pass_manager.pre_init = ffsim.qiskit.PRE_INIT
    return pass_manager.run(circuit)


def pass_manager(
    transpile_target: str, config: dict | None = None
) -> PassManager | None:
    """The pass manager for a compilation target named in ``params.json``.

    ``build_circuits`` runs the result on the measured fragment circuits, and
    the transpiled circuits are what ``FragmentCircuits`` writes to QPY. QPY
    loads native ffsim gates as opaque gates, which no sampler can run, so every
    target except ``"native"`` has to remove them.

    Args:
        transpile_target: One of

            ``"backend:<name>"``
                Compile for that IBM backend through
                :func:`transpile_for_target`, giving the ISA circuits the
                ``QCSCSamplerV2`` path needs. Requires Runtime credentials
                wherever the prepare stage runs.
            ``"unmapped"``
                Standard gates on virtual qubits, with no coupling map and so no
                layout or routing. This is what Aer runs, and what QRMI's
                node-side transpiler expects.
            ``"native"``
                No pass manager. Only ``serial.py --inline-circuits`` can use
                it, because the circuits then never go through QPY.

        config: Options for the target; see :func:`transpile_for_target`.

    Returns:
        The pass manager, or None for ``"native"``.

    Raises:
        ValueError: If the target is not one of the above.
    """
    config = dict(config or {})
    if transpile_target == "native":
        return None
    if transpile_target == "unmapped":
        unmapped = generate_preset_pass_manager(
            optimization_level=0,
            basis_gates=list(config.get("basis_gates", UNMAPPED_BASIS_GATES)),
        )
        if config.get("pre_init", True):
            unmapped.pre_init = ffsim.qiskit.PRE_INIT
        return unmapped
    if transpile_target.startswith("backend:"):
        from qiskit_ibm_runtime import QiskitRuntimeService

        backend = QiskitRuntimeService().backend(transpile_target.split(":", 1)[1])
        return _TargetPassManager(backend.target, config)
    raise ValueError(
        f"unknown transpile_target {transpile_target!r}: expected 'unmapped', "
        "'native' or 'backend:<name>'"
    )


class _TargetPassManager(PassManager):
    """Adapts :func:`transpile_for_target` to ``build_circuits``' pass manager.

    Keeping the backend branch behind the hook's three-argument function is the
    point; this wrapper only lets ``build_circuits`` call it.
    """

    def __init__(self, target, config: dict | None = None) -> None:
        super().__init__()
        self.target = target
        self.config = dict(config or {})

    def run(self, circuits, *args, **kwargs):
        if isinstance(circuits, QuantumCircuit):
            return transpile_for_target(circuits, self.target, self.config)
        return [transpile_for_target(c, self.target, self.config) for c in circuits]


def classical_reference(problem: Problem) -> float:
    """Classical LASSCF with mrh's exact fragment solvers, from the same orbitals.

    Uses a separate LAS object so the run's own ``las`` is untouched.

    Returns:
        The total energy.

    Raises:
        RuntimeError: If the reference does not converge.
    """
    reference = LASSCF(
        problem.mf, settings.NCAS_SUB, settings.NELECAS_SUB, spin_sub=settings.SPIN_SUB
    )
    reference.kernel(problem.mo_initial.copy())
    if not reference.converged:
        raise RuntimeError("the classical LASSCF reference did not converge")
    return float(reference.e_tot)
