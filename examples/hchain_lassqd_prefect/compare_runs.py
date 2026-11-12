"""Walk two run directories in execution order and report the first divergence.

::

    python compare_runs.py RUN_A RUN_B [--bitwise] [--atol 1e-10] [--cycles N]

This is how two runs of the example are compared; see the README's
"Verification". It stops at the first record outside tolerance, which is what
locates where two runs started to differ: a later difference is usually a
consequence of an earlier one.

The quantities compared, in the order the stages produce them:

* ``params.json`` -- only the fields that change what is computed, so two runs
  that differ in their execution target still compare;
* ``problem.json`` -- exactly, including the digests of the modules that define
  the problem;
* ``state.h5`` and ``embedding.h5`` -- the orbitals, the fragment RDMs and each
  fragment's Hamiltonian digest;
* ``samples.h5`` -- the counts as dictionaries **with equal key order**, because
  SQD draws samples by position;
* ``result_f*.h5`` -- energy, RDMs, amplitudes, CI strings and the fragment
  digest the result was solved for;
* ``sqd_state_f*.h5`` -- the random-generator state and the carryover strings
  with their basis, which is the strongest determinism probe here;
* ``next_state.h5``, ``cycle.json`` and ``energies.csv``.

Job manifests and solve logs are not compared: they carry attempt identifiers,
host names and timings, which differ by design.

This module imports ``lassqd.records``, NumPy and the standard library, so it
runs without mrh, PySCF, Qiskit or qiskit-addon-sqd.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

import numpy as np

import layout

# Fields of params.json that change what a run computes. The rest -- the run
# directory, the execution target, block names, poll intervals -- do not.
NUMERICAL_PARAMS = (
    "conv_tol",
    "global_depolarizing",
    "nfrags",
    "sampler",
    "sci_solver",
    "shots",
    "sqd",
    "transpile_config",
    "transpile_target",
)


def _arrays(name: str, left, right, *, bitwise: bool, atol: float) -> str | None:
    """None if the arrays agree, else what differs."""
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        return f"{name}: shapes {left.shape} and {right.shape}"
    if left.dtype != right.dtype:
        return f"{name}: dtypes {left.dtype} and {right.dtype}"
    if left.size == 0:
        return None
    if np.array_equal(left, right):
        return None
    if bitwise or not np.issubdtype(left.dtype, np.floating):
        differing = int(np.count_nonzero(left != right))
        return (
            f"{name}: {differing} of {left.size} elements differ, largest "
            f"difference {float(np.max(np.abs(left - right))):.3e}"
        )
    deviation = float(np.max(np.abs(left - right)))
    if deviation > atol:
        return f"{name}: largest difference {deviation:.3e} exceeds atol {atol:.3e}"
    return None


def _values(name: str, left, right, *, bitwise: bool, atol: float) -> str | None:
    """Compare two scalars, strings, mappings or sequences structurally."""
    if isinstance(left, float) or isinstance(right, float):
        if left == right:
            return None
        if bitwise:
            return f"{name}: {left!r} and {right!r}"
        if abs(float(left) - float(right)) > atol:
            return (
                f"{name}: {left!r} and {right!r} differ by "
                f"{abs(float(left) - float(right)):.3e}"
            )
        return None
    if isinstance(left, dict) and isinstance(right, dict):
        if list(left) != list(right):
            return f"{name}: keys {list(left)} and {list(right)}"
        for key in left:
            message = _values(
                f"{name}[{key!r}]", left[key], right[key], bitwise=bitwise, atol=atol
            )
            if message:
                return message
        return None
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return f"{name}: lengths {len(left)} and {len(right)}"
        for index, (one, other) in enumerate(zip(left, right)):
            message = _values(
                f"{name}[{index}]", one, other, bitwise=bitwise, atol=atol
            )
            if message:
                return message
        return None
    if left != right:
        return f"{name}: {left!r} and {right!r}"
    return None


def _las_state(name, left, right, *, bitwise, atol):
    yield _arrays(
        f"{name}.mo_coeff", left.mo_coeff, right.mo_coeff, bitwise=bitwise, atol=atol
    )
    for index, (one, other) in enumerate(zip(left.casdm1frs, right.casdm1frs)):
        yield _arrays(
            f"{name}.casdm1frs[{index}]", one, other, bitwise=bitwise, atol=atol
        )
    for index, (one, other) in enumerate(zip(left.casdm2fr, right.casdm2fr)):
        yield _arrays(
            f"{name}.casdm2fr[{index}]", one, other, bitwise=bitwise, atol=atol
        )


def _samples(name, left, right, *, bitwise, atol):
    for index, (one, other) in enumerate(zip(left.counts, right.counts)):
        # Key order matters: SQD draws its samples by position.
        if list(one) != list(other):
            shared = set(one) == set(other)
            yield (
                f"{name}.counts[{index}]: the same {len(one)} bitstrings in a "
                "different order"
                if shared
                else f"{name}.counts[{index}]: {len(one)} and {len(other)} bitstrings"
            )
            return
        yield _values(f"{name}.counts[{index}]", one, other, bitwise=True, atol=atol)
    for index, (one, other) in enumerate(
        zip(left.embedding.fragments, right.embedding.fragments)
    ):
        yield _values(
            f"{name}.embedding.fragments[{index}].digest",
            one.digest(),
            other.digest(),
            bitwise=True,
            atol=atol,
        )


def _result(name, left, right, *, bitwise, atol):
    yield _values(
        f"{name}.fragment_digest",
        left.fragment_digest,
        right.fragment_digest,
        bitwise=True,
        atol=atol,
    )
    yield _values(f"{name}.e_tot", left.e_tot, right.e_tot, bitwise=bitwise, atol=atol)
    for field in ("ci_strs_a", "ci_strs_b"):
        yield _arrays(
            f"{name}.{field}",
            getattr(left, field),
            getattr(right, field),
            bitwise=True,
            atol=atol,
        )
    for field in ("dm1s", "dm2", "amplitudes"):
        yield _arrays(
            f"{name}.{field}",
            getattr(left, field),
            getattr(right, field),
            bitwise=bitwise,
            atol=atol,
        )


def _sqd_state(name, left, right, *, bitwise, atol):
    yield _values(
        f"{name}.rng_state", left.rng_state, right.rng_state, bitwise=True, atol=atol
    )
    if (left.carryover_strings is None) != (right.carryover_strings is None):
        yield f"{name}.carryover_strings: one is absent"
        return
    if left.carryover_strings is not None:
        for spin, (one, other) in enumerate(
            zip(left.carryover_strings, right.carryover_strings)
        ):
            yield _arrays(
                f"{name}.carryover_strings[{spin}]", one, other, bitwise=True, atol=atol
            )
        yield _arrays(
            f"{name}.carryover_orbitals",
            left.carryover_orbitals,
            right.carryover_orbitals,
            bitwise=bitwise,
            atol=atol,
        )


def _read_energies(path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def checks(
    left: layout.RunLayout, right: layout.RunLayout, *, bitwise: bool, atol: float
):
    """Yield either None or a description, for every compared quantity in order."""
    from lassqd.records import (
        Embedding,
        FragmentResult,
        FragmentSamples,
        LASState,
        SQDState,
    )

    left_params = left.read_params()
    right_params = right.read_params()
    for field in NUMERICAL_PARAMS:
        yield _values(
            f"params.json[{field!r}]",
            left_params.get(field),
            right_params.get(field),
            bitwise=True,
            atol=atol,
        )
    yield _values(
        "problem.json",
        json.loads(left.problem().read_text()),
        json.loads(right.problem().read_text()),
        bitwise=True,
        atol=atol,
    )

    nfrags = left_params["nfrags"]
    for cycle in range(_common_cycles(left, right)):
        tag = f"c{cycle:03d}"
        yield from _las_state(
            f"{tag}/state.h5",
            LASState.load(left.state(cycle)),
            LASState.load(right.state(cycle)),
            bitwise=bitwise,
            atol=atol,
        )
        for index, (one, other) in enumerate(
            zip(
                Embedding.load(left.embedding(cycle)).fragments,
                Embedding.load(right.embedding(cycle)).fragments,
            )
        ):
            yield _values(
                f"{tag}/embedding.h5 fragment {index} digest",
                one.digest(),
                other.digest(),
                bitwise=True,
                atol=atol,
            )
        yield from _samples(
            f"{tag}/samples.h5",
            FragmentSamples.load(left.samples(cycle)),
            FragmentSamples.load(right.samples(cycle)),
            bitwise=bitwise,
            atol=atol,
        )
        for fragment in range(nfrags):
            yield from _result(
                f"{tag}/result_f{fragment:02d}.h5",
                FragmentResult.load(left.result(cycle, fragment)),
                FragmentResult.load(right.result(cycle, fragment)),
                bitwise=bitwise,
                atol=atol,
            )
            yield from _sqd_state(
                f"{tag}/sqd_state_f{fragment:02d}.h5",
                SQDState.load(left.sqd_state(cycle, fragment)),
                SQDState.load(right.sqd_state(cycle, fragment)),
                bitwise=bitwise,
                atol=atol,
            )
        yield from _las_state(
            f"{tag}/next_state.h5",
            LASState.load(left.next_state(cycle)),
            LASState.load(right.next_state(cycle)),
            bitwise=bitwise,
            atol=atol,
        )
        yield _values(
            f"{tag}/cycle.json",
            _without(json.loads(left.cycle_json(cycle).read_text()), "seconds"),
            _without(json.loads(right.cycle_json(cycle).read_text()), "seconds"),
            bitwise=bitwise,
            atol=atol,
        )
    yield _values(
        "energies.csv",
        _read_energies(left.energies()),
        _read_energies(right.energies()),
        bitwise=True,
        atol=atol,
    )


def _without(record: dict, *fields: str) -> dict:
    return {key: value for key, value in record.items() if key not in fields}


def _common_cycles(left: layout.RunLayout, right: layout.RunLayout) -> int:
    return min(_committed_cycles(left), _committed_cycles(right))


def _committed_cycles(run: layout.RunLayout) -> int:
    cycle = 0
    while run.cycle_json(cycle).exists():
        cycle += 1
    return cycle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument(
        "--bitwise",
        action="store_true",
        help="require exact equality of the floating-point quantities too",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-10,
        help="tolerance on energies, RDMs and amplitudes without --bitwise",
    )
    parser.add_argument(
        "--cycles", type=int, help="compare at most this many macrocycles"
    )
    args = parser.parse_args(argv)

    left = layout.RunLayout(args.left)
    right = layout.RunLayout(args.right)
    common = _common_cycles(left, right)
    if args.cycles is not None:
        common = min(common, args.cycles)
    if common == 0:
        print("neither run has committed a macrocycle", file=sys.stderr)
        return 2

    compared = 0
    for message in checks(left, right, bitwise=args.bitwise, atol=args.atol):
        compared += 1
        if message:
            print(f"first divergence after {compared} comparisons:\n  {message}")
            return 1
    mode = "bitwise" if args.bitwise else f"atol {args.atol:.1e}"
    print(
        f"{compared} comparisons agree ({mode}) over {common} macrocycles:\n"
        f"  {left.root}\n  {right.root}"
    )
    extra = {
        "left": _committed_cycles(left),
        "right": _committed_cycles(right),
    }
    if extra["left"] != extra["right"]:
        print(
            f"note: {extra['left']} and {extra['right']} committed macrocycles; "
            f"compared the first {common}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
