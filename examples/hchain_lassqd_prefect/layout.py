"""Where every record of a run lives, and the logical key of every job.

:class:`RunLayout` is the single source of these paths. The flow process and
every worker build it from the run directory alone, so a worker is told only
``--run-dir`` and ``--cycle`` and derives the rest.

A run directory holds run-level files and one directory per macrocycle::

    <run_dir>/params.json problem.json mf.chk energies.csv summary.json
    <run_dir>/c000/ state.h5 embedding.h5 circuits.h5 samples.h5
                    sqd_state_f00.h5 result_f00.h5 solve_f00.json ...
                    next_state.h5 cycle.json
    <run_dir>/c000/jobs/{capture,prepare,solve-f00,solve-f01,replay}/

Records live in the macrocycle directory; the job directories hold only the
rendered job script, its logs and the stage's manifest. Each logical job needs
its own directory because QCSC's Slurm template hardcodes
``#SBATCH --output={{work_dir}}/output.out``.

This module imports the standard library only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# The stages of one macrocycle, in execution order.
STAGES = ("capture", "prepare", "sample", "solve", "replay")


class RunLayout:
    """Paths and logical keys of one run.

    Args:
        run_dir: Root of the run directory. It is resolved, so the layout's
            paths do not depend on the working directory a worker is launched
            in.
    """

    def __init__(self, run_dir: str | os.PathLike) -> None:
        self.root = Path(run_dir).expanduser().resolve()

    def __repr__(self) -> str:
        return f"RunLayout({str(self.root)!r})"

    # --- Run-level files. ---

    def params(self) -> Path:
        """The run's settings, written once by ``exec.py`` or ``serial.py``."""
        return self.root / "params.json"

    def problem(self) -> Path:
        """The molecular problem and the digests of the modules that define it."""
        return self.root / "problem.json"

    def mean_field(self) -> Path:
        """PySCF checkpoint of the RHF reference, written by cycle 0's capture."""
        return self.root / "mf.chk"

    def energies(self) -> Path:
        """One row per completed macrocycle, appended by the replay stage."""
        return self.root / "energies.csv"

    def summary(self) -> Path:
        """The run's outcome, written when the macrocycle loop stops."""
        return self.root / "summary.json"

    def read_params(self) -> dict:
        """The run's settings, as every worker reads them.

        Workers take only their identity on the command line and everything
        numerical from this file, so the flow and ``serial.py`` cannot drift
        apart in what they configure. ``flow_params.FlowParameters`` writes it.
        """
        return json.loads(self.params().read_text())

    # --- Macrocycle records. ---

    def cycle_dir(self, cycle: int) -> Path:
        return self.root / f"c{cycle:03d}"

    def state(self, cycle: int) -> Path:
        """The ``LASState`` this macrocycle starts from."""
        return self.cycle_dir(cycle) / "state.h5"

    def embedding(self, cycle: int) -> Path:
        """The ``Embedding`` captured at :meth:`state`."""
        return self.cycle_dir(cycle) / "embedding.h5"

    def circuits(self, cycle: int) -> Path:
        """The compiled ``FragmentCircuits``."""
        return self.cycle_dir(cycle) / "circuits.h5"

    def samples(self, cycle: int) -> Path:
        """The ``FragmentSamples``: the counts and the embedding they came from."""
        return self.cycle_dir(cycle) / "samples.h5"

    def sqd_state(self, cycle: int, fragment: int) -> Path:
        """Fragment ``fragment``'s ``SQDState`` after this macrocycle's solve."""
        return self.cycle_dir(cycle) / f"sqd_state_f{fragment:02d}.h5"

    def result(self, cycle: int, fragment: int) -> Path:
        """Fragment ``fragment``'s ``FragmentResult``."""
        return self.cycle_dir(cycle) / f"result_f{fragment:02d}.h5"

    def solve_log(self, cycle: int, fragment: int) -> Path:
        """What the solve worker observed: subspace sizes, timings, counts."""
        return self.cycle_dir(cycle) / f"solve_f{fragment:02d}.json"

    def next_state(self, cycle: int) -> Path:
        """The ``LASState`` after the orbital step: the next macrocycle's input."""
        return self.cycle_dir(cycle) / "next_state.h5"

    def cycle_json(self, cycle: int) -> Path:
        """The macrocycle's committed outcome, written after :meth:`next_state`."""
        return self.cycle_dir(cycle) / "cycle.json"

    # --- Job directories. ---

    def jobs_dir(self, cycle: int) -> Path:
        return self.cycle_dir(cycle) / "jobs"

    def job_dir(self, cycle: int, stage: str, fragment: int | None = None) -> Path:
        """Working directory of one logical job: its script, logs and manifest."""
        return self.jobs_dir(cycle) / self.job_name(stage, fragment)

    def manifest(self, cycle: int, stage: str, fragment: int | None = None) -> Path:
        """The stage's result manifest: the only evidence of its output."""
        return self.job_dir(cycle, stage, fragment) / "manifest.json"

    # --- Names and keys. ---

    @staticmethod
    def job_name(stage: str, fragment: int | None = None) -> str:
        """``"solve-f01"`` for a per-fragment stage, ``"replay"`` for a global one."""
        _check_stage(stage)
        return stage if fragment is None else f"{stage}-f{fragment:02d}"

    def logical_key(self, cycle: int, stage: str, fragment: int | None = None) -> str:
        """The readable logical key of one job, such as ``"c007/solve/f01"``.

        The run directory stands in for the calculation identifier and the
        parent-snapshot digest a fuller key would carry; see the README's
        "Not implemented" section.
        """
        _check_stage(stage)
        tail = "all" if fragment is None else f"f{fragment:02d}"
        return f"c{cycle:03d}/{stage}/{tail}"

    def script_stem(self, cycle: int, stage: str, fragment: int | None = None) -> str:
        """Job-script stem for the logical key, such as ``"c007-solve-f01"``.

        ``run_job_from_blocks`` appends the scheduler's suffix, so the Slurm
        script of a solve is ``c007-solve-f01.slurm``.
        """
        return f"c{cycle:03d}-{self.job_name(stage, fragment)}"


def _check_stage(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}, expected one of {STAGES}")
