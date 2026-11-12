"""The serial reference: the same stages, in one process or as subprocesses.

::

    python serial.py --run-dir R --max-cycles 3 --sampler aer
    python serial.py --run-dir R --max-cycles 3 --sampler aer --subprocess
    python serial.py --classical-reference

This is the reference the orchestrated run is compared against, and with
``--subprocess`` the reference for the process boundary on its own. It drives
the same workers, the same ``layout.RunLayout`` and the same ``settings``
helpers as the Prefect flow, with nothing between the stages, so a difference
against the flow is attributable to the orchestration alone.

``run_lassqd`` cannot be the reference: it holds one sampler object whose stream
advances across macrocycles and across the pubs of one job, and no record
persists it, so a workflow that samples afresh could never be bitwise equal to
it. Here both sides derive macrocycle ``c``'s stream as
``settings.sampler_rng(c)`` instead.

With ``--subprocess`` each stage runs as ``python <worker>.py ...`` with the
exact argument list the flow passes to QCSC. Sampling always runs in this
process, as it does in the flow.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

import layout
import manifest
import sampling
import settings
from flow_params import FlowParameters

HERE = Path(__file__).resolve().parent

WORKERS = {
    "capture": "capture_worker",
    "prepare": "circuits_worker",
    "solve": "solve_worker",
    "replay": "replay_worker",
}


def run_worker(stage: str, argv: list[str], *, as_subprocess: bool) -> None:
    """Run one stage's worker, in this process or as a subprocess.

    Raises:
        RuntimeError: If the worker reports a nonzero exit status.
    """
    module_name = WORKERS[stage]
    if as_subprocess:
        command = [sys.executable, str(HERE / f"{module_name}.py"), *argv]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"{stage} worker exited with {completed.returncode}: "
                f"{' '.join(command)}"
            )
        return
    worker = importlib.import_module(module_name)
    status = worker.main(argv)
    if status:
        raise RuntimeError(f"{stage} worker returned {status}")


def stage_argv(
    run: layout.RunLayout, stage: str, cycle: int, fragment: int | None = None
):
    """The argument list the flow passes to this stage's worker.

    Keeping one definition means ``serial.py --subprocess`` and the flow cannot
    disagree about the command line.
    """
    argv = ["--run-dir", str(run.root), "--cycle", str(cycle)]
    if stage == "capture":
        if cycle == 0:
            return argv + ["--initial"]
        return argv + ["--state", str(run.next_state(cycle - 1))]
    if stage == "prepare" or stage == "replay":
        return argv
    if stage == "solve":
        argv += ["--fragment", str(fragment)]
        if cycle > 0:
            argv += ["--sqd-state", str(run.sqd_state(cycle - 1, fragment))]
        return argv
    raise ValueError(f"{stage!r} has no worker")


def inline_circuits_and_samples(
    run: layout.RunLayout, cycle: int, params: dict
) -> None:
    """Build and sample the circuits in memory, writing only ``samples.h5``.

    This is the one path that can use native ffsim gates and ``FfsimSampler``,
    and so reproduce ``docs/lassqd.ipynb``'s configuration: as soon as the
    prepare and sample stages are separate processes the circuits must go
    through QPY, which loads native ffsim gates as opaque gates. Nothing else in
    the example may skip the ``FragmentCircuits`` record.
    """
    from lassqd import build_circuits
    from lassqd.records import Embedding

    import problem

    embedding = Embedding.load(run.embedding(cycle))
    circuits = build_circuits(
        embedding,
        problem.prepare_fragment,
        pass_manager=problem.pass_manager(
            params["transpile_target"], params.get("transpile_config")
        ),
        groups=settings.GROUPS,
    )
    samples = asyncio.run(sampling.sample(circuits, params, cycle))
    samples.save(run.samples(cycle))
    manifest.write(
        run.manifest(cycle, sampling.STAGE),
        key=run.logical_key(cycle, sampling.STAGE),
        stage=sampling.STAGE,
        cycle=cycle,
        files=[run.samples(cycle)],
        extra={
            "sampler": params["sampler"],
            "job_ids": list(samples.job_ids),
            "inline_circuits": True,
            "shots": [int(sum(c.values())) for c in samples.counts],
            "distinct_bitstrings": [len(c) for c in samples.counts],
        },
    )
    print(
        f"sampled {run.logical_key(cycle, sampling.STAGE)} from in-memory circuits "
        f"with {params['sampler']}"
    )


def run_cycle(
    run: layout.RunLayout,
    cycle: int,
    params: dict,
    *,
    as_subprocess: bool,
    inline: bool,
) -> dict:
    """Run one macrocycle's five stages and return its committed ``cycle.json``.

    Each stage's manifest is cleared immediately before that stage runs, in the
    same order as the flow does it, so that a resumed macrocycle can still reuse
    the counts it already has.
    """

    def stage(name: str, fragment: int | None = None) -> None:
        manifest.clear(run.manifest(cycle, name, fragment))
        run_worker(
            name,
            stage_argv(run, name, cycle, fragment),
            as_subprocess=as_subprocess,
        )

    stage("capture")
    if inline:
        manifest.clear(run.manifest(cycle, sampling.STAGE))
        inline_circuits_and_samples(run, cycle, params)
    else:
        stage("prepare")
        # Sampling is the one stage a restart must not repeat; see
        # sampling.reuse_samples.
        if sampling.reuse_samples(run, cycle) is None:
            manifest.clear(run.manifest(cycle, sampling.STAGE))
            asyncio.run(sampling.sample_cycle(run, cycle, params))
        else:
            print(f"reusing the stored counts of {run.logical_key(cycle, 'sample')}")
    for fragment in range(params["nfrags"]):
        stage("solve", fragment)
    stage("replay")
    # --inline-circuits publishes no prepare manifest: it writes no circuits
    # record, so there is no logical prepare job.
    global_stages = (
        ("capture", "sample", "replay")
        if inline
        else (
            "capture",
            "prepare",
            "sample",
            "replay",
        )
    )
    for stage in global_stages:
        manifest.verify(run.manifest(cycle, stage), key=run.logical_key(cycle, stage))
    for fragment in range(params["nfrags"]):
        manifest.verify(
            run.manifest(cycle, "solve", fragment),
            key=run.logical_key(cycle, "solve", fragment),
        )
    return json.loads(run.cycle_json(cycle).read_text())


def committed_cycles(run: layout.RunLayout) -> list[dict]:
    """The ``cycle.json`` of every macrocycle this run directory has committed."""
    history = []
    cycle = 0
    while run.cycle_json(cycle).exists():
        history.append(json.loads(run.cycle_json(cycle).read_text()))
        cycle += 1
    return history


def classical_reference() -> int:
    """Print classical LASSCF with mrh's exact fragment solvers."""
    import problem

    built = problem.build_problem()
    print(f"RHF total energy:              {built.mf.e_tot:.10f} Hartree")
    print(
        f"Classical LASSCF total energy: {problem.classical_reference(built):.10f} Hartree"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir")
    parser.add_argument("--params", help="a parameters file, as exec.py takes")
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--conv-tol", type=float)
    parser.add_argument("--shots", type=int)
    parser.add_argument("--sampler", choices=("aer", "ffsim", "runtime"))
    parser.add_argument("--sci-solver", choices=("fulqrum", "pyscf"))
    parser.add_argument(
        "--global-depolarizing",
        type=float,
        help="FfsimSampler only: the measurement noise docs/lassqd.ipynb uses",
    )
    parser.add_argument("--transpile-target")
    parser.add_argument(
        "--subprocess",
        action="store_true",
        help="run each stage as its own process, with the flow's argument list",
    )
    parser.add_argument(
        "--inline-circuits",
        action="store_true",
        help="keep the circuits in memory, the only way to use FfsimSampler",
    )
    parser.add_argument(
        "--classical-reference",
        action="store_true",
        help="print classical LASSCF from the same initial orbitals and exit",
    )
    args = parser.parse_args(argv)

    if args.classical_reference:
        return classical_reference()
    if not args.run_dir and not args.params:
        parser.error("either --run-dir or --params is required")

    overrides = {
        "run_dir": args.run_dir,
        "max_cycles": args.max_cycles,
        "conv_tol": args.conv_tol,
        "shots": args.shots,
        "sampler": args.sampler,
        "sci_solver": args.sci_solver,
        "transpile_target": args.transpile_target,
        "global_depolarizing": args.global_depolarizing,
    }
    if args.params:
        parameters = FlowParameters.load(args.params, **overrides)
    else:
        defaults = {
            "sampler": "aer",
            "transpile_target": "unmapped",
            # serial.py never launches a job, so the target only labels the run.
            "hpc_target": "local",
        }
        defaults.update({k: v for k, v in overrides.items() if v is not None})
        parameters = FlowParameters(**defaults)

    run = layout.RunLayout(parameters.run_dir)
    run.root.mkdir(parents=True, exist_ok=True)
    params = parameters.write(run)
    mode = "subprocesses" if args.subprocess else "one process"
    print(
        f"serial LASSQD in {mode}: {run.root}, sampler {params['sampler']!r}, "
        f"up to {params['max_cycles']} macrocycles"
    )

    start = time.perf_counter()
    # The same restart contract as the flow: resume at the first macrocycle
    # without a committed cycle.json.
    history = committed_cycles(run)
    energies = [record["e_tot"] for record in history]
    converged = bool(
        history
        and history[-1]["de"] is not None
        and abs(history[-1]["de"]) < params["conv_tol"]
    )
    if history:
        print(f"resuming after {len(history)} committed macrocycles")
    for cycle in range(len(history), params["max_cycles"]):
        if converged:
            break
        committed = run_cycle(
            run,
            cycle,
            params,
            as_subprocess=args.subprocess,
            inline=args.inline_circuits,
        )
        energies.append(committed["e_tot"])
        if committed["de"] is not None and abs(committed["de"]) < params["conv_tol"]:
            converged = True
            break
    elapsed = time.perf_counter() - start

    summary = {
        "converged": converged,
        "cycles": len(energies),
        "e_tot": energies[-1],
        "e_hist": energies,
        "seconds": elapsed,
        "mode": "subprocess" if args.subprocess else "inline",
    }
    run.summary().write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"{'converged' if converged else 'not converged'} after {len(energies)} "
        f"macrocycles: E = {energies[-1]:.10f} Hartree in {elapsed:.1f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
