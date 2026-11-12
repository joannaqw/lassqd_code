"""The settings of one run, validated once and then written to ``params.json``.

``exec.py`` and ``serial.py`` build a :class:`FlowParameters`, write it into the
run directory, and from then on the flow and every worker read that file. The
workers read it with :meth:`layout.RunLayout.read_params`, which needs nothing
but the standard library; this module is where the fields are defined and
checked.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

import block_names
import settings

SAMPLERS = ("runtime", "aer", "qrmi", "ffsim")


class FlowParameters(BaseModel):
    """Everything a run needs that is not in ``settings.py`` or ``problem.py``."""

    model_config = ConfigDict(extra="forbid")

    run_dir: str = Field(description="Root of the run directory.")

    # --- The macrocycle loop. ---
    max_cycles: int = Field(default=settings.MAX_CYCLES, gt=0)
    conv_tol: float = Field(default=settings.CONV_TOL, gt=0)
    nfrags: int = Field(default=settings.NFRAGS, gt=0)

    # --- Circuits and sampling. ---
    shots: int = Field(default=settings.SHOTS, gt=0)
    sampler: Literal["runtime", "aer", "qrmi", "ffsim"] = "runtime"
    runtime_block_name: str | None = Field(
        default=None, description="Saved QiskitRuntimeConfig block; the runtime path."
    )
    transpile_target: str = Field(
        default="unmapped",
        description="'backend:<name>', 'unmapped' or 'native'; see problem.pass_manager.",
    )
    transpile_config: dict = Field(default_factory=dict)
    global_depolarizing: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="FfsimSampler only: the noise the notebook uses.",
    )

    # --- The fragment solves. ---
    sci_solver: Literal["fulqrum", "pyscf"] = "fulqrum"
    sqd: dict = Field(
        default_factory=dict, description="Overrides of settings.SQD_DEFAULTS."
    )

    # --- QCSC and Prefect, unused by serial.py. ---
    hpc_target: Literal["slurm", "local"] = "slurm"
    hpc_profile_block_name: str | None = None
    block_prefix: str = "lassqd"
    watch_poll_interval: float = Field(default=10.0, gt=0)
    timeouts: dict[str, float | None] = Field(
        default_factory=dict, description="Per-stage timeout_seconds for QCSC."
    )

    @model_validator(mode="after")
    def _check(self) -> FlowParameters:
        self.run_dir = str(Path(self.run_dir).expanduser())
        if self.hpc_profile_block_name is None:
            self.hpc_profile_block_name = block_names.hpc_profile(
                self.hpc_target, self.block_prefix
            )
        if self.sampler == "runtime" and not self.runtime_block_name:
            raise ValueError(
                "sampler 'runtime' needs runtime_block_name, the name of a saved "
                "QiskitRuntimeConfig block"
            )
        if self.sampler == "qrmi":
            raise ValueError(
                "the QRMI path is not implemented here: QCSC's QrmiRuntime needs "
                "a qrmi binary, a qrmi_config.json and a QRMI SPANK resource, "
                "none of which the cluster this example was validated on has; "
                "see the README"
            )
        if self.sampler == "ffsim" and self.transpile_target != "native":
            raise ValueError(
                "FfsimSampler can only run native ffsim gates, so it needs "
                "transpile_target 'native' and serial.py --inline-circuits"
            )
        if self.transpile_target == "native" and self.sampler != "ffsim":
            raise ValueError(
                "transpile_target 'native' is for the ffsim sampler only; QPY "
                "stores native ffsim gates as opaque gates"
            )
        # Rejects an unknown key here rather than in a worker on a compute node.
        settings.sqd_options(self.sqd)
        unknown = sorted(set(self.timeouts) - set(block_names.JOB_STAGES))
        if unknown:
            raise ValueError(f"timeouts for unknown stages: {', '.join(unknown)}")
        return self

    @classmethod
    def load(cls, path: str | os.PathLike, **overrides) -> FlowParameters:
        """Read a parameters file, such as ``params.local.json``."""
        document = json.loads(Path(path).read_text())
        document.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**document)

    def write(self, run) -> dict:
        """Write ``params.json`` into the run directory and return its contents.

        Raises:
            ValueError: If the run directory already holds different settings;
                restarting a run must not silently change what it computes.
        """
        record = self.model_dump(mode="json")
        path = run.params()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            stored = json.loads(path.read_text())
            if stored != record:
                differing = sorted(
                    key
                    for key in set(stored) | set(record)
                    if stored.get(key) != record.get(key)
                )
                raise ValueError(
                    f"{path} holds different settings; these differ: "
                    f"{', '.join(differing)}"
                )
            return stored
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
        return record
