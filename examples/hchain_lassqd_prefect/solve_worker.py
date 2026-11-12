"""Solve stage: one fragment's counts become a ``FragmentResult``.

::

    python solve_worker.py --run-dir R --cycle C --fragment F [--sqd-state PATH]

The body is written in the three-call form of ``FragmentSQD`` from the start,
because that is where a collective MPI solve attaches: ``prepare`` and
``finalize`` are rank-0 work and ``SQDInputs.run`` is the call every rank of a
collective solve enters. With one rank it is bitwise identical to
``FragmentSQD.step``.

Fragment ``f`` reads only ``samples.h5`` and its own ``sqd_state_f{f}.h5`` and
writes only ``f``-indexed files, so the fragments of one macrocycle are
order-independent by construction.

This is the only worker that imports ``qiskit_addon_sqd``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from lassqd.records import FragmentSamples, SQDState

import layout
import manifest
import settings
import sqd_settings

STAGE = "solve"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--fragment", type=int, required=True)
    parser.add_argument(
        "--sqd-state",
        help="the previous macrocycle's SQDState; omit to start a new stream",
    )
    args = parser.parse_args(argv)

    run = layout.RunLayout(args.run_dir)
    cycle, ifrag = args.cycle, args.fragment
    params = run.read_params()
    options = settings.sqd_options(params.get("sqd"))

    start = time.perf_counter()
    samples = FragmentSamples.load(run.samples(cycle))
    embedding = samples.embedding
    fragment = embedding.fragments[ifrag]
    counts = samples.counts[ifrag]

    # Before SQD, not inside it: an empty disjoint batch fails in the batch
    # solver, and in a collective solve on whichever rank owns that batch.
    postselected = sqd_settings.check_batches(
        counts, fragment.norb, fragment.nelec, options["num_batches"]
    )

    dimensions: list[list[int]] = []

    def record_dimensions(results) -> None:
        dimensions.append([result.sci_state.amplitudes.size for result in results])

    solver = sqd_settings.make_solver(
        ifrag,
        options,
        sci_solver_name=params.get("sci_solver", "fulqrum"),
        callback=record_dimensions,
    )
    state = SQDState.load(args.sqd_state) if args.sqd_state else solver.state

    # One rank here. A collective adapter replaces sqd_settings.sci_solver and
    # guards the rank-0 work below; see the README's "Not implemented" section.
    rank = 0
    inputs = solver.prepare(fragment, counts, state, ao_ovlp=embedding.ao_ovlp)
    best = inputs.run()
    if rank != 0:
        return 0

    result, next_state = solver.finalize(fragment, inputs, best)
    result.save(run.result(cycle, ifrag))
    next_state.save(run.sqd_state(cycle, ifrag))
    elapsed = time.perf_counter() - start

    carryover = next_state.carryover_strings
    log = {
        "key": run.logical_key(cycle, STAGE, ifrag),
        "seconds": elapsed,
        "e_tot": result.e_tot,
        "norb": fragment.norb,
        "nelec": list(fragment.nelec),
        "shots": int(sum(counts.values())),
        "distinct_bitstrings": len(counts),
        "postselected_bitstrings": postselected,
        "iteration_dimensions": dimensions,
        "subspace": [len(result.ci_strs_a), len(result.ci_strs_b)],
        "carryover": None if carryover is None else [len(s) for s in carryover],
        "fragment_digest": result.fragment_digest,
        "sci_solver": params.get("sci_solver", "fulqrum"),
        "sqd": options,
    }
    run.solve_log(cycle, ifrag).write_text(
        json.dumps(log, indent=2, sort_keys=True) + "\n"
    )

    manifest.write(
        run.manifest(cycle, STAGE, ifrag),
        key=log["key"],
        stage=STAGE,
        cycle=cycle,
        fragment=ifrag,
        files=[
            run.result(cycle, ifrag),
            run.sqd_state(cycle, ifrag),
            run.solve_log(cycle, ifrag),
        ],
        extra={"seconds": elapsed, "e_tot": result.e_tot},
    )
    print(
        f"solved {log['key']}: E = {result.e_tot:.10f} Ha, subspace "
        f"{log['subspace']}, {postselected} postselected bitstrings, "
        f"in {elapsed:.1f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
