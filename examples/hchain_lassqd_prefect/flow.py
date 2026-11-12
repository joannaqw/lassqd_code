"""The Prefect flow: one task per stage, with HDF5 records as the only handoff.

Per macrocycle the tasks are capture, verify, prepare, sample, one solve per
fragment, and replay: one task per stage whose inputs and outputs are stored
records, with one sampler job and ``F`` solve tasks. Every task but sampling
submits a scheduler job through
QCSC's ``run_job_from_blocks``; sampling runs here, in the flow process, because
that is where QCSC's QPU paths live (see ``sampling.py``).

``run_job_from_blocks`` is a plain coroutine function, not a Prefect task, so
each stage needs its own ``@task`` wrapper. :func:`run_stage` is the shared body.

Two things the flow must get right:

* ``run_slurm_job`` and ``run_local_job`` **return** a result object carrying
  ``exit_status`` and the final scheduler state; they do not raise on failure.
  A task that does not inspect it treats a failed job as a successful one.
* the scheduler state says a process ended, not that it produced anything, so
  every stage's manifest is verified afterwards.

Prefect draws a dependency edge when a ``PrefectFuture`` is passed as a task
argument. Passing the list of solve futures to :func:`replay_task` therefore
gives ``F`` edges and makes replay the barrier it has to be: it cannot run
until every fragment of the macrocycle is solved. Across macrocycles the
serialization is the ``outcome.result()`` at the end of each iteration.
"""

from __future__ import annotations

import json
from typing import Any

from prefect import flow, get_run_logger, task
from qcsc_prefect_executor.from_blocks import run_job_from_blocks

import block_names
import layout
import manifest
import sampling
import settings
from flow_params import FlowParameters


async def run_stage(
    params: FlowParameters,
    *,
    stage: str,
    cycle: int,
    fragment: int | None = None,
    user_args: list[str],
    overrides: dict[str, Any] | None = None,
) -> str:
    """Submit one logical job, wait for it, and verify its manifest.

    Args:
        params: The run's settings.
        stage: One of ``block_names.JOB_STAGES``.
        cycle: Macrocycle index.
        fragment: Fragment index for a per-fragment stage, else None.
        user_args: Arguments appended after the command block's worker script.
        overrides: Per-call ``execution_profile_overrides``. A collective solve
            would pass ``{"num_nodes": P_f, "launcher": "srun",
            "mpi_options": ["--mpi=pmix_v3"]}`` here.

    Returns:
        The stage's logical key.

    Raises:
        RuntimeError: If the job reported a nonzero exit status.
        FileNotFoundError, ValueError: If its manifest is missing or invalid.
    """
    logger = get_run_logger()
    run = layout.RunLayout(params.run_dir)
    work = run.job_dir(cycle, stage, fragment)
    work.mkdir(parents=True, exist_ok=True)
    key = run.logical_key(cycle, stage, fragment)
    # Publication is single-shot, so a rerun of this logical job clears the
    # previous attempt's manifest here rather than in the worker.
    manifest.clear(run.manifest(cycle, stage, fragment))
    names = block_names.stage_blocks(stage, params.block_prefix)
    logger.info("submitting %s", key)
    result = await run_job_from_blocks(
        command_block_name=names["command"],
        execution_profile_block_name=names["execution_profile"],
        hpc_profile_block_name=params.hpc_profile_block_name,
        work_dir=work,
        script_filename=run.script_stem(cycle, stage, fragment),
        user_args=user_args,
        watch_poll_interval=params.watch_poll_interval,
        timeout_seconds=params.timeouts.get(stage),
        metrics_artifact_key=f"lassqd-{stage}",
        execution_profile_overrides=overrides,
    )
    exit_status = int(getattr(result, "exit_status", 0) or 0)
    job_id = getattr(result, "job_id", "?")
    state = getattr(result, "state", None)
    if exit_status != 0:
        raise RuntimeError(
            f"{key}: job {job_id} exited with {exit_status} (scheduler state "
            f"{state!r}); see {work}/output.out and {work}/output.err"
        )
    published = manifest.verify(run.manifest(cycle, stage, fragment), key=key)
    logger.info(
        "%s finished as job %s in %.1f s",
        key,
        job_id,
        published.get("seconds", float("nan")),
    )
    return key


@task
async def capture_task(params: FlowParameters, cycle: int) -> str:
    """Capture the macrocycle's fragment Hamiltonians and sampling bases."""
    run = layout.RunLayout(params.run_dir)
    user_args = ["--run-dir", str(run.root), "--cycle", str(cycle)]
    if cycle == 0:
        user_args.append("--initial")
    else:
        user_args += ["--state", str(run.next_state(cycle - 1))]
    return await run_stage(params, stage="capture", cycle=cycle, user_args=user_args)


@task
async def verify_problem_task(params: FlowParameters, captured: str) -> dict:
    """Check that the node that captured ran this flow's problem definition.

    ``problem.json`` records the fragmentation, the seed and the digests of the
    modules that define the numerical problem, so a worker built from a
    different revision of the example is detected here rather than by a
    puzzling energy later.

    Raises:
        ValueError: If the record disagrees with this process's settings.
    """
    run = layout.RunLayout(params.run_dir)
    record = json.loads(run.problem().read_text())
    expected = {
        "nfrags": params.nfrags,
        "seed": settings.SEED,
        "module_digests": settings.module_digests(),
    }
    differing = {
        key: (record.get(key), value)
        for key, value in expected.items()
        if record.get(key) != value
    }
    if differing:
        raise ValueError(
            f"{run.problem()} disagrees with this flow process after {captured}: "
            + "; ".join(
                f"{key} is {found!r} on file but {wanted!r} here"
                for key, (found, wanted) in differing.items()
            )
        )
    return record


@task
async def prepare_task(params: FlowParameters, cycle: int, problem_record: dict) -> str:
    """Build and compile the fragment circuits."""
    run = layout.RunLayout(params.run_dir)
    return await run_stage(
        params,
        stage="prepare",
        cycle=cycle,
        user_args=["--run-dir", str(run.root), "--cycle", str(cycle)],
    )


@task
async def sample_task(params: FlowParameters, cycle: int, prepared: str) -> list[str]:
    """Sample the circuits through the QPU gateway, here in the flow process.

    A restart reuses this macrocycle's stored counts when a valid sample
    manifest names them and they came from this embedding; see
    :func:`sampling.reuse_samples`.

    Returns:
        The provider job identifiers recorded with the counts.
    """
    logger = get_run_logger()
    run = layout.RunLayout(params.run_dir)
    stored = sampling.reuse_samples(run, cycle)
    if stored is not None:
        logger.info(
            "reusing the stored counts of %s from job %s",
            run.logical_key(cycle, sampling.STAGE),
            stored.job_ids[0],
        )
        return list(stored.job_ids)
    manifest.clear(run.manifest(cycle, sampling.STAGE))
    samples = await sampling.sample_cycle(run, cycle, params.model_dump(mode="json"))
    manifest.verify(
        run.manifest(cycle, sampling.STAGE),
        key=run.logical_key(cycle, sampling.STAGE),
    )
    return list(samples.job_ids)


# Tagged so a Prefect concurrency limit -- or a global limit whose slots stand
# for nodes, with each solve holding P_f of them -- can be attached without a
# code change.
@task(tags=["lassqd-solve"])
async def solve_task(
    params: FlowParameters, cycle: int, fragment: int, job_ids: list[str]
) -> str:
    """Solve one fragment with SQD on its stored counts."""
    run = layout.RunLayout(params.run_dir)
    user_args = [
        "--run-dir",
        str(run.root),
        "--cycle",
        str(cycle),
        "--fragment",
        str(fragment),
    ]
    if cycle > 0:
        user_args += ["--sqd-state", str(run.sqd_state(cycle - 1, fragment))]
    return await run_stage(
        params,
        stage="solve",
        cycle=cycle,
        fragment=fragment,
        user_args=user_args,
        # A collective solve sets num_nodes=P_f, launcher="srun" and
        # mpi_options here; see the README's "Not implemented" section.
        overrides=None,
    )


@task
async def replay_task(params: FlowParameters, cycle: int, solved: list[str]) -> dict:
    """Replay the fragment results into mrh and commit the orbital step.

    Every fragment's solve manifest is re-verified first: the solve tasks
    checked them when they finished, but replay is about to consume the files
    they name, and a missing or changed result must fail here rather than be
    replayed stale.
    """
    run = layout.RunLayout(params.run_dir)
    for fragment in range(params.nfrags):
        manifest.verify(
            run.manifest(cycle, "solve", fragment),
            key=run.logical_key(cycle, "solve", fragment),
        )
    await run_stage(
        params,
        stage="replay",
        cycle=cycle,
        user_args=["--run-dir", str(run.root), "--cycle", str(cycle)],
    )
    return json.loads(run.cycle_json(cycle).read_text())


def committed_cycles(run: layout.RunLayout) -> list[dict]:
    """The ``cycle.json`` of every macrocycle this run directory has committed.

    ``cycle.json`` is the replay worker's last record, written after
    ``next_state.h5`` is renamed into place, so its presence means the whole
    macrocycle is committed.
    """
    history = []
    cycle = 0
    while run.cycle_json(cycle).exists():
        history.append(json.loads(run.cycle_json(cycle).read_text()))
        cycle += 1
    return history


@flow(name="lassqd-distributed")
def lassqd_distributed(params: FlowParameters) -> dict:
    """Run macrocycles until ``|dE| < conv_tol`` or ``max_cycles``.

    Args:
        params: The run's settings, which are written into the run directory as
            ``params.json`` before the first stage; every worker reads them
            from there.

    Returns:
        The run's summary, also written as ``summary.json``.
    """
    logger = get_run_logger()
    run = layout.RunLayout(params.run_dir)
    run.root.mkdir(parents=True, exist_ok=True)
    params.write(run)
    logger.info(
        "LASSQD on %s: %s, sampler %r, up to %d macrocycles",
        params.hpc_target,
        run.root,
        params.sampler,
        params.max_cycles,
    )

    # The initial restart contract: resume at the first macrocycle that has no
    # committed cycle.json, from the LASState the previous replay committed.
    history = committed_cycles(run)
    energies = [record["e_tot"] for record in history]
    converged = bool(
        history
        and history[-1]["de"] is not None
        and abs(history[-1]["de"]) < params.conv_tol
    )
    if history:
        logger.info(
            "resuming after %d committed macrocycles, E = %.10f Ha",
            len(history),
            energies[-1],
        )
    for cycle in range(len(history), params.max_cycles):
        if converged:
            break
        captured = capture_task.submit(params, cycle)
        problem_record = verify_problem_task.submit(params, captured)
        prepared = prepare_task.submit(params, cycle, problem_record)
        job_ids = sample_task.submit(params, cycle, prepared)
        solved = [
            solve_task.submit(params, cycle, fragment, job_ids)
            for fragment in range(params.nfrags)
        ]
        outcome = replay_task.submit(params, cycle, solved)
        committed = outcome.result()
        energies.append(committed["e_tot"])
        change = committed["de"]
        logger.info(
            "macrocycle %d: E = %.10f Ha, dE = %s",
            cycle,
            committed["e_tot"],
            "--" if change is None else f"{change:+.3e}",
        )
        if change is not None and abs(change) < params.conv_tol:
            converged = True
            break

    summary = {
        "converged": converged,
        "cycles": len(energies),
        "e_tot": energies[-1],
        "e_hist": energies,
        "hpc_target": params.hpc_target,
        "sampler": params.sampler,
        "sci_solver": params.sci_solver,
    }
    run.summary().write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    logger.info(
        "%s after %d macrocycles: E = %.10f Hartree",
        "converged" if converged else "not converged",
        len(energies),
        energies[-1],
    )
    return summary
