"""Prepare stage: an ``Embedding`` becomes compiled ``FragmentCircuits``.

::

    python circuits_worker.py --run-dir R --cycle C

The compilation target comes from the run's ``params.json``. The abstract
circuit of ``problem.prepare_fragment`` -- native ffsim gates and all -- goes
straight into one pass manager: ``build_circuits`` measures it and runs
``problem.pass_manager(...)`` on it, and the transpiled result is what the QPY
record holds.

Because QPY loads native ffsim gates as opaque gates, which no sampler can run,
the worker asserts afterwards that no operation is still an ffsim one. That
assertion is the check; there is no manual decomposition step.

This worker must not import ``qiskit_addon_sqd``; see ``problem.py``.
"""

from __future__ import annotations

import argparse
import sys
import time

from lassqd import build_circuits
from lassqd.records import Embedding

import layout
import manifest
import problem
import settings

STAGE = "prepare"


def ffsim_operations(circuits) -> list[str]:
    """Names of operations that would reload from QPY as opaque gates."""
    found = set()
    for circuit in circuits:
        for instruction in circuit.data:
            operation = instruction.operation
            if (type(operation).__module__ or "").startswith("ffsim."):
                found.add(f"{operation.name} ({type(operation).__name__})")
    return sorted(found)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cycle", type=int, required=True)
    args = parser.parse_args(argv)

    run = layout.RunLayout(args.run_dir)
    cycle = args.cycle
    params = run.read_params()
    target = params["transpile_target"]
    if target == "native":
        parser.error(
            "transpile_target 'native' keeps the circuits' ffsim gates, which QPY "
            "stores as opaque gates that no sampler can run; it is available only "
            "to serial.py --inline-circuits, where the circuits stay in memory"
        )

    start = time.perf_counter()
    embedding = Embedding.load(run.embedding(cycle))
    circuits = build_circuits(
        embedding,
        problem.prepare_fragment,
        pass_manager=problem.pass_manager(target, params.get("transpile_config")),
        groups=settings.GROUPS,
    )
    remaining = ffsim_operations(circuits.circuits)
    if remaining:
        raise AssertionError(
            f"compiling for {target!r} left ffsim operations in the circuits, which "
            f"QPY stores as opaque gates: {', '.join(remaining)}"
        )
    circuits.save(run.circuits(cycle))
    elapsed = time.perf_counter() - start

    depths = [circuit.depth() for circuit in circuits.circuits]
    published = manifest.write(
        run.manifest(cycle, STAGE),
        key=run.logical_key(cycle, STAGE),
        stage=STAGE,
        cycle=cycle,
        files=[run.circuits(cycle)],
        extra={
            "seconds": elapsed,
            "transpile_target": target,
            "num_qubits": [circuit.num_qubits for circuit in circuits.circuits],
            "depths": depths,
            "two_qubit_gates": [
                sum(1 for i in circuit.data if len(i.qubits) == 2)
                for circuit in circuits.circuits
            ],
            "output_registers": circuits.output_registers,
        },
    )
    print(
        f"compiled {len(circuits.circuits)} circuits for {published['key']} "
        f"onto {target!r}, depths {depths}, in {elapsed:.1f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
