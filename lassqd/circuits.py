"""Fragment circuits and their sampling.

:func:`build_circuits` builds one circuit per fragment group of an
:class:`lassqd.records.Embedding`, :func:`submit_circuits` samples some or all of
them in one sampler job, and :func:`collect_samples` splits the pub results into
per-fragment counts. :class:`FragmentCircuits` is saved and loaded like the
records of :mod:`lassqd.records`, but needs Qiskit.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qpy
from qiskit.primitives import BasePrimitiveJob, BaseSamplerV2, SamplerPubResult
from qiskit.transpiler import PassManager

from lassqd.records import Embedding, Fragment, FragmentSamples, _Record

if TYPE_CHECKING:
    import h5py

# ``circuit_builder(fragment) -> circuit``; see build_circuits for the contract.
CircuitBuilder = Callable[[Fragment], QuantumCircuit]


@dataclass
class FragmentCircuits(_Record):
    """Measured, compiled circuits for one embedding, from :func:`build_circuits`.

    ``save`` stores the circuits in QPY, so load them with the same Qiskit
    version or a newer one. Save compiled circuits: QPY loads native ffsim gates
    as opaque gates, which ``FfsimSampler`` cannot run.

    Attributes:
        embedding: The fragment problems the circuits were built from.
        circuits: One circuit per fragment group, in group order.
        output_registers: ``(circuit_index, register_name)`` holding each
            fragment's measurements, in fragment order.
    """

    embedding: Embedding
    circuits: list[QuantumCircuit]
    output_registers: list[tuple[int, str]]

    def _write(self, group: h5py.Group) -> None:
        self._write_child(group, "embedding", self.embedding)
        buffer = io.BytesIO()
        qpy.dump(self.circuits, buffer)
        group.create_dataset("circuits", data=np.void(buffer.getvalue()))
        group.attrs["output_registers"] = json.dumps(self.output_registers)

    @classmethod
    def _read(cls, group: h5py.Group) -> FragmentCircuits:
        circuits = qpy.load(io.BytesIO(group["circuits"][()].tobytes()))
        return cls(
            cls._read_child(group, "embedding", Embedding),
            list(circuits),
            [
                (int(index), name)
                for index, name in json.loads(group.attrs["output_registers"])
            ],
        )


def build_circuits(
    embedding: Embedding,
    circuit_builder: CircuitBuilder,
    *,
    pass_manager: PassManager | None = None,
    groups: Sequence[Sequence[int]] | None = None,
) -> FragmentCircuits:
    """Build, measure and compile the fragment circuits of ``embedding``.

    Args:
        embedding: Fragment problems, from :func:`lassqd.las.capture_embedding`.
        circuit_builder: Callable ``(fragment) -> QuantumCircuit`` receiving a
            :class:`lassqd.records.Fragment`. Return an unmeasured
            ``2 * fragment.norb``-qubit state-preparation circuit in the
            fragment's sampling basis, whose integrals are ``fragment.h1`` and
            ``fragment.h2`` (chemists' notation). Alpha orbitals occupy qubits
            ``0..norb-1`` and beta orbitals ``norb..2*norb-1``. The caller
            chooses the ansatz, interaction pairs and initialization settings.
        pass_manager: Optional pass manager applied to the measured circuits.
        groups: Fragment indices of each circuit; together they must contain
            every fragment exactly once. A group of several fragments is glued
            into one circuit with one classical register per fragment; a group
            of one fragment is measured on its own. None glues all fragments
            into one circuit.

    Returns:
        The circuits and where each fragment's measurements will be.

    Raises:
        ValueError: If ``groups`` does not contain every fragment exactly once.
    """
    nfrags = len(embedding.fragments)
    fragment_circuits = [circuit_builder(fragment) for fragment in embedding.fragments]
    if groups is None:
        circuits = [_glue_circuits(fragment_circuits)]
        output_registers = [(0, creg.name) for creg in circuits[0].cregs]
    else:
        groups = [list(group) for group in groups]
        if sorted(i for group in groups for i in group) != list(
            range(nfrags)
        ) or not all(groups):
            raise ValueError(
                f"groups {groups} must contain each of the {nfrags} fragments once"
            )
        circuits = []
        output_registers = [None] * nfrags
        for index, group in enumerate(groups):
            if len(group) == 1:
                circuit = fragment_circuits[group[0]].measure_all(inplace=False)
                output_registers[group[0]] = (index, circuit.cregs[-1].name)
            else:
                circuit = _glue_circuits([fragment_circuits[i] for i in group])
                for i, creg in zip(group, circuit.cregs):
                    output_registers[i] = (index, creg.name)
            circuits.append(circuit)
    if pass_manager is not None:
        circuits = [pass_manager.run(circuit) for circuit in circuits]
    return FragmentCircuits(embedding, circuits, output_registers)


def submit_circuits(
    sampler: BaseSamplerV2,
    circuits: FragmentCircuits,
    *,
    shots: int | None = None,
    indices: Sequence[int] | None = None,
) -> BasePrimitiveJob:
    """Submit circuits to ``sampler`` in one job.

    Keep ``job.job_id()`` to recover a remote result later, and pass the job's
    pub results to :func:`collect_samples`.

    Args:
        sampler: SamplerV2 primitive.
        circuits: Circuits from :func:`build_circuits`.
        shots: Shots per fragment; None uses the sampler's default.
        indices: Indices of the circuits to submit, in pub order; None submits
            them all.

    Returns:
        The submitted job.
    """
    pubs = circuits.circuits
    if indices is not None:
        pubs = [pubs[i] for i in indices]
    return sampler.run(pubs, shots=shots)


def collect_samples(
    circuits: FragmentCircuits,
    results: Sequence[SamplerPubResult],
    *,
    job_ids: str | Sequence[str | None] | None = None,
) -> FragmentSamples:
    """Split the circuits' pub results into per-fragment counts.

    Args:
        circuits: The circuits that were submitted.
        results: One pub result per circuit, in circuit order: a job's result,
            from ``job.result()`` or recovered from a remote service by job ID,
            or the pub results of several jobs combined.
        job_ids: Identifier of the job that ran each circuit, or one identifier
            for all of them.

    Returns:
        Each fragment's counts and the embedding they were sampled from.

    Raises:
        ValueError: If the number of results or job identifiers differs from the
            number of circuits.
    """
    ncircuits = len(circuits.circuits)
    if len(results) != ncircuits:
        raise ValueError(f"{len(results)} pub results for {ncircuits} circuits")
    if job_ids is None or isinstance(job_ids, str):
        job_ids = [job_ids] * ncircuits
    if len(job_ids) != ncircuits:
        raise ValueError(f"{len(job_ids)} job identifiers for {ncircuits} circuits")
    counts = [
        results[index].data[register].get_counts()
        for index, register in circuits.output_registers
    ]
    return FragmentSamples(circuits.embedding, counts, job_ids)


def _glue_circuits(circuits: Sequence[QuantumCircuit]) -> QuantumCircuit:
    """Place circuits side by side on one register and measure each separately.

    Args:
        circuits: Fragment circuits, in fragment order.

    Returns:
        One circuit whose qubits are the fragment circuits' qubits concatenated in
        order, with one classical register per fragment; ``circuits[i]`` is
        measured into ``cregs[i]``.
    """
    widths = [qc.num_qubits for qc in circuits]
    cregs = [ClassicalRegister(n) for n in widths]
    glued = QuantumCircuit(QuantumRegister(sum(widths)), *cregs)
    start = 0
    for qc, creg, n in zip(circuits, cregs, widths):
        qubits = range(start, start + n)
        glued.append(qc, qubits)
        glued.measure(qubits, creg)
        start += n
    return glued
