"""Default Prefect block document names for this example's stages.

The flow, ``create_blocks.py`` and ``blocks.scc.toml`` all derive their names
here, so renaming a block is one edit. ``block_prefix`` in ``params.json``
separates several configurations in one Prefect workspace.

Each job stage has a ``CommandBlock`` (what to run) and an
``ExecutionProfileBlock`` (how to run it), and the run has one
``HPCProfileBlock`` (where to run it).

This module imports nothing.
"""

from __future__ import annotations

# The stages launched as scheduler jobs. Sampling is not one: the gateway runs
# in the flow process; see the README.
JOB_STAGES = ("capture", "prepare", "solve", "replay")


def command_name(stage: str, prefix: str = "lassqd") -> str:
    """The logical command name, which the two blocks of a stage must agree on."""
    return f"{prefix}-{stage}"


def command(stage: str, prefix: str = "lassqd") -> str:
    return f"cmd-{prefix}-{stage}"


def execution_profile(stage: str, prefix: str = "lassqd") -> str:
    return f"exec-{prefix}-{stage}"


def hpc_profile(hpc_target: str, prefix: str = "lassqd") -> str:
    return f"hpc-{prefix}-{hpc_target}"


def runtime_config(prefix: str = "lassqd") -> str:
    return f"runtime-{prefix}"


def stage_blocks(stage: str, prefix: str = "lassqd") -> dict[str, str]:
    """The block names ``run_job_from_blocks`` needs for one stage."""
    if stage not in JOB_STAGES:
        raise ValueError(f"{stage!r} does not run as a job; expected {JOB_STAGES}")
    return {
        "command": command(stage, prefix),
        "execution_profile": execution_profile(stage, prefix),
    }
