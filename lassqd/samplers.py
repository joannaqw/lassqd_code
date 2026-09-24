"""Samplers: run a glued circuit and return one counts dict per fragment.

A sampler is any callable ``sampler(circuit) -> list[dict[str, int]]`` where
``circuit`` comes from :func:`lassqd.circuits.glue_circuits` and the list is in
fragment order.
"""

from qiskit.compiler import transpile

from lassqd.circuits import cut_counts


def aer_sampler(shots=100_000, method="matrix_product_state"):
    """Sample classically with Qiskit Aer."""
    from qiskit_aer import AerSimulator

    simulator = AerSimulator(method=method)

    def sample(circuit):
        counts = simulator.run(transpile(circuit, simulator), shots=shots).result().get_counts()
        return cut_counts(counts)

    return sample


def ibm_runtime_sampler(
    service,
    backend_name,
    *,
    shots,
    initial_layout=None,
    optimization_level=3,
    dynamical_decoupling=True,
    twirling=True,
    max_execution_time=10800,
):
    """Sample on IBM Quantum hardware with Qiskit Runtime's SamplerV2.

    ``service`` is a ``qiskit_ibm_runtime.QiskitRuntimeService``. The call blocks
    until the job finishes; the job id is printed so a lost result can be
    retrieved with ``service.job(job_id)``.
    """
    import ffsim
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2, Session

    def sample(circuit):
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
