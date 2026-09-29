"""Samplers: run a glued circuit and return one counts dict per fragment.

A sampler is any callable ``sampler(circuit) -> list[dict[str, int]]`` where
``circuit`` comes from :func:`lassqd.circuits.glue_circuits` and the list is in
fragment order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from qiskit import QuantumCircuit
from qiskit.compiler import transpile

from lassqd.circuits import cut_counts

if TYPE_CHECKING:
    from qiskit_ibm_runtime import QiskitRuntimeService

Sampler = Callable[[QuantumCircuit], list[dict[str, int]]]


def aer_sampler(shots: int = 100_000, method: str = "matrix_product_state") -> Sampler:
    """Sample classically with Qiskit Aer.

    Args:
        shots: Number of shots per job.
        method: ``AerSimulator`` simulation method.

    Returns:
        A sampler that simulates the glued circuit and returns one counts dict per
        fragment.
    """
    from qiskit_aer import AerSimulator

    simulator = AerSimulator(method=method)

    def sample(circuit: QuantumCircuit) -> list[dict[str, int]]:
        counts = simulator.run(transpile(circuit, simulator), shots=shots).result().get_counts()
        return cut_counts(counts)

    return sample


def ibm_runtime_sampler(
    service: QiskitRuntimeService,
    backend_name: str,
    *,
    shots: int,
    initial_layout: Sequence[int] | None = None,
    optimization_level: int = 3,
    dynamical_decoupling: bool = True,
    twirling: bool = True,
    max_execution_time: int = 10800,
) -> Sampler:
    """Sample on IBM Quantum hardware with Qiskit Runtime's SamplerV2.

    Each call of the returned sampler blocks until the job finishes; the job id is
    printed so a lost result can be retrieved with ``service.job(job_id)``.

    Args:
        service: Qiskit Runtime service to run on.
        backend_name: Name of the backend, e.g. ``"ibm_sherbrooke"``.
        shots: Number of shots per job.
        initial_layout: Physical qubit for each circuit qubit, passed to the preset
            pass manager.
        optimization_level: Preset pass manager optimization level.
        dynamical_decoupling: Whether to enable ``XpXm`` dynamical decoupling.
        twirling: Whether to enable gate and measurement twirling.
        max_execution_time: Job execution time limit, in seconds.

    Returns:
        A sampler that transpiles and runs the glued circuit in a Runtime session
        and returns one counts dict per fragment.
    """
    import ffsim
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2, Session

    def sample(circuit: QuantumCircuit) -> list[dict[str, int]]:
        backend = service.backend(backend_name)
        pass_manager = generate_preset_pass_manager(
            backend=backend,
            optimization_level=optimization_level,
            initial_layout=initial_layout,
        )
        pass_manager.pre_init = ffsim.qiskit.PRE_INIT
        isa_circuit = pass_manager.run(circuit)
        with Session(backend=backend) as session:
            sampler = SamplerV2(mode=session)
            sampler.options.default_shots = shots
            sampler.options.max_execution_time = max_execution_time
            if dynamical_decoupling:
                sampler.options.dynamical_decoupling.enable = True
                sampler.options.dynamical_decoupling.sequence_type = "XpXm"
            if twirling:
                sampler.options.twirling.enable_gates = True
                sampler.options.twirling.enable_measure = True
            job = sampler.run([isa_circuit])
            print(f"submitted job {job.job_id()}", flush=True)
            data = job.result()[0].data
        return [data[creg.name].get_counts() for creg in circuit.cregs]

    return sample
