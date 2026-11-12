"""Replay stage: the fragment results become one LASSCF orbital step.

::

    python replay_worker.py --run-dir R --cycle C

``update_orbitals`` re-enters mrh at the captured embedding's orbitals and
starting RDMs with callbacks that check each requested Hamiltonian against the
captured one and return the stored results. mrh then computes the coupled LAS
energy, takes one orbital step and canonicalizes.

The commit is ordered so an interrupted write never looks complete:
``next_state.h5`` is written under a temporary name and renamed into place, and
``cycle.json`` -- which is what the flow reads to decide whether the macrocycle
finished -- is written last.

This worker must not import ``qiskit_addon_sqd``; see ``problem.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

from lassqd import update_orbitals
from lassqd.records import Embedding, FragmentResult

import layout
import manifest
import problem

STAGE = "replay"
ENERGY_FIELDS = ("cycle", "e_tot", "de")


def record_energy(path, cycle: int, e_tot: float, de: float | None) -> None:
    """Append this macrocycle's energy, replacing any row for it or a later one.

    Rewriting rather than appending keeps a rerun of a macrocycle idempotent.
    """
    rows = []
    if path.exists():
        with open(path, newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if int(row["cycle"]) < cycle]
    rows.append(
        {
            "cycle": str(cycle),
            "e_tot": repr(float(e_tot)),
            "de": "" if de is None else repr(float(de)),
        }
    )
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ENERGY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def previous_energy(run, cycle: int) -> float | None:
    """The energy committed by macrocycle ``cycle - 1``, if there is one."""
    if cycle == 0:
        return None
    path = run.cycle_json(cycle - 1)
    if not path.exists():
        return None
    return float(json.loads(path.read_text())["e_tot"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cycle", type=int, required=True)
    args = parser.parse_args(argv)

    run = layout.RunLayout(args.run_dir)
    cycle = args.cycle
    params = run.read_params()
    nfrags = params["nfrags"]

    start = time.perf_counter()
    embedding = Embedding.load(run.embedding(cycle))
    results = [FragmentResult.load(run.result(cycle, i)) for i in range(nfrags)]
    built = problem.las_from_run(run)
    step = update_orbitals(built.las, embedding, results)
    elapsed = time.perf_counter() - start

    temporary = run.next_state(cycle).with_suffix(".h5.tmp")
    step.state.save(temporary)
    os.replace(temporary, run.next_state(cycle))

    previous = previous_energy(run, cycle)
    de = None if previous is None else step.e_tot - previous
    record_energy(run.energies(), cycle, step.e_tot, de)

    committed = {
        "cycle": cycle,
        "key": run.logical_key(cycle, STAGE),
        "e_tot": float(step.e_tot),
        "de": None if de is None else float(de),
        "fragment_energies": [float(r.e_tot) for r in results],
        "seconds": elapsed,
    }
    # Written last: its presence means the macrocycle is committed.
    run.cycle_json(cycle).write_text(
        json.dumps(committed, indent=2, sort_keys=True) + "\n"
    )

    manifest.write(
        run.manifest(cycle, STAGE),
        key=committed["key"],
        stage=STAGE,
        cycle=cycle,
        files=[run.next_state(cycle), run.cycle_json(cycle), run.energies()],
        extra={"seconds": elapsed, "e_tot": committed["e_tot"], "de": committed["de"]},
    )
    change = "--" if de is None else f"{de:+.3e}"
    print(
        f"committed {committed['key']}: E = {step.e_tot:.10f} Ha, dE = {change}, "
        f"in {elapsed:.1f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
