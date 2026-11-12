"""The QPU gateway: compiled circuits in, counts out.

Sampling runs in the flow process rather than as a scheduler job, because both
QPU paths QCSC offers are already Prefect objects living there --
``submit_sampler_job_task``, ``fetch_qiskit_job_result_task`` and
``QrmiRuntime`` -- and routing them through a worker would mean reimplementing
submit and fetch and losing QCSC's caching, retries and artifacts. It also keeps
a scheduler allocation from sitting idle while a provider job is queued.

The gateway needs Qiskit and h5py plus ``lassqd.circuits``, never mrh, PySCF or
qiskit-addon-sqd. ``tests/test_imports.py`` guarantees that property of
``FragmentCircuits``.

``QCSCSamplerV2.run`` and ``QCSCPrimitiveJob.result`` are coroutines, which is
why synchronous ``run_lassqd`` cannot drive them and this example calls the
stage functions instead. ``job.job_id()`` is synchronous on both paths.
"""

from __future__ import annotations

import inspect

from lassqd.circuits import FragmentCircuits, collect_samples, submit_circuits
from lassqd.records import Embedding, FragmentSamples
from qiskit.primitives import BaseSamplerV2

import layout
import manifest
import settings

STAGE = "sample"


def aer_seed(cycle: int) -> int:
    """An integer ``seed_simulator`` for macrocycle ``cycle``.

    Aer's ``SamplerV2`` passes its ``seed`` straight to ``seed_simulator``,
    which must be an integer, so the macrocycle's stream is drawn from rather
    than handed over.
    """
    return int(settings.sampler_rng(cycle).integers(2**31 - 1))


def make_sampler(params: dict, cycle: int) -> BaseSamplerV2:
    """The sampler for one macrocycle; see the README's table of QPU paths.

    Args:
        params: The run's ``params.json``.
        cycle: Macrocycle index, which fixes the sampler's random stream.

    Returns:
        A SamplerV2-style sampler. The ``runtime`` path returns QCSC's
        ``QCSCSamplerV2``, whose ``run`` is a coroutine.

    Raises:
        ValueError: If ``params["sampler"]`` is unknown.
        NotImplementedError: For the QRMI path, which this cluster cannot run.
    """
    kind = params["sampler"]
    shots = params["shots"]
    if kind == "runtime":
        from qcsc_prefect.integrations.qiskit import QCSCSamplerV2

        return QCSCSamplerV2(runtime_block_name=params["runtime_block_name"])
    if kind == "aer":
        from qiskit_aer.primitives import SamplerV2

        return SamplerV2(default_shots=shots, seed=aer_seed(cycle))
    if kind == "ffsim":
        from ffsim.qiskit import FfsimSampler

        return FfsimSampler(
            seed=settings.sampler_rng(cycle),
            global_depolarizing=params.get("global_depolarizing", 0.0),
        )
    if kind == "qrmi":
        raise NotImplementedError(
            "the QRMI path needs a qrmi binary, a qrmi_config.json and a QRMI "
            "SPANK resource, none of which the cluster this example was "
            "validated on has; see the README"
        )
    raise ValueError(f"unknown sampler {kind!r}")


async def sample(
    circuits: FragmentCircuits, params: dict, cycle: int
) -> FragmentSamples:
    """Submit ``circuits`` in one job and split the pub results per fragment.

    Awaits whatever the sampler returns, so one body serves the synchronous
    simulators and QCSC's coroutine-based submit and fetch.
    """
    sampler = make_sampler(params, cycle)
    job = submit_circuits(sampler, circuits, shots=params["shots"])
    if inspect.isawaitable(job):
        job = await job
    job_id = job.job_id()
    result = job.result()
    if inspect.isawaitable(result):
        result = await result
    # job_ids is recorded with the counts, so the provider job that produced
    # them is fixed before any downstream stage consumes them.
    return collect_samples(circuits, result, job_ids=job_id)


def reuse_samples(run: layout.RunLayout, cycle: int) -> FragmentSamples | None:
    """This macrocycle's stored counts, if a valid sample manifest names them.

    Sampling is the one stage a restart must not repeat: two sampler jobs give
    different counts, so the provider job recorded with the counts is the only
    one whose results are ever used. The classical stages are deterministic
    functions of their inputs, so rerunning them costs only time.

    The stored counts are accepted only if they were sampled from *this*
    macrocycle's embedding, which their fragments' digests show.

    Returns:
        The samples, or None if there are none to reuse.
    """
    path = run.samples(cycle)
    if not path.exists():
        return None
    try:
        manifest.verify(run.manifest(cycle, STAGE), key=run.logical_key(cycle, STAGE))
    except (FileNotFoundError, ValueError):
        return None
    samples = FragmentSamples.load(path)
    stored = [f.digest() for f in samples.embedding.fragments]
    captured = [f.digest() for f in Embedding.load(run.embedding(cycle)).fragments]
    if stored != captured:
        return None
    return samples


async def sample_cycle(
    run: layout.RunLayout, cycle: int, params: dict
) -> FragmentSamples:
    """Load the macrocycle's circuits, sample them, and write ``samples.h5``.

    Returns:
        The samples, which are also on disk.
    """
    circuits = FragmentCircuits.load(run.circuits(cycle))
    samples = await sample(circuits, params, cycle)
    samples.save(run.samples(cycle))
    shots = [int(sum(counts.values())) for counts in samples.counts]
    distinct = [len(counts) for counts in samples.counts]
    manifest.write(
        run.manifest(cycle, STAGE),
        key=run.logical_key(cycle, STAGE),
        stage=STAGE,
        cycle=cycle,
        files=[run.samples(cycle)],
        extra={
            "sampler": params["sampler"],
            "job_ids": list(samples.job_ids),
            "shots": shots,
            "distinct_bitstrings": distinct,
        },
    )
    print(
        f"sampled {run.logical_key(cycle, STAGE)} with {params['sampler']}: "
        f"job {samples.job_ids[0]}, {shots} shots, {distinct} distinct bitstrings"
    )
    return samples
