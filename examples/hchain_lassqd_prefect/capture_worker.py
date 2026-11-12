"""Capture stage: a committed ``LASState`` becomes an ``Embedding``.

::

    python capture_worker.py --run-dir R --cycle C --initial
    python capture_worker.py --run-dir R --cycle C --state <previous next_state.h5>

Macrocycle 0 builds mrh's initial-guess RDMs at the localized orbitals; a later
macrocycle starts from the ``LASState`` the previous replay committed. Either
way the state is written into this macrocycle's directory as ``state.h5``, so
every cycle records the input it ran on.

This worker must not import ``qiskit_addon_sqd``; see ``problem.py``.
"""

from __future__ import annotations

import argparse
import sys
import time

from lassqd import capture_embedding, initial_state
from lassqd.records import LASState

import layout
import manifest
import problem

STAGE = "capture"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cycle", type=int, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--initial",
        action="store_true",
        help="start from the localized orbitals and mrh's initial-guess RDMs",
    )
    source.add_argument("--state", help="the LASState to start this macrocycle from")
    args = parser.parse_args(argv)

    run = layout.RunLayout(args.run_dir)
    cycle = args.cycle
    if args.initial and cycle != 0:
        parser.error(f"--initial is for macrocycle 0, not {cycle}")
    run.cycle_dir(cycle).mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    built = problem.las_from_run(run)
    problem.write_problem_record(run, built)
    if args.initial:
        state = initial_state(built.las, built.mo_initial)
    else:
        state = LASState.load(args.state)
    state.save(run.state(cycle))

    embedding = capture_embedding(built.las, state)
    embedding.save(run.embedding(cycle))
    elapsed = time.perf_counter() - start

    published = manifest.write(
        run.manifest(cycle, STAGE),
        key=run.logical_key(cycle, STAGE),
        stage=STAGE,
        cycle=cycle,
        files=[run.state(cycle), run.embedding(cycle), run.problem(), run.mean_field()],
        extra={
            "seconds": elapsed,
            "fragment_digests": [f.digest() for f in embedding.fragments],
            "rohf_converged": [f.rohf_converged for f in embedding.fragments],
            "spin_asymmetry": [f.spin_asymmetry for f in embedding.fragments],
        },
    )
    print(
        f"captured {len(embedding.fragments)} fragments for {published['key']} "
        f"in {elapsed:.1f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
