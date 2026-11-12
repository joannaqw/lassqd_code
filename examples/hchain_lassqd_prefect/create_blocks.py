"""Turn a site configuration into saved Prefect blocks.

::

    python create_blocks.py --config blocks.scc.toml --dry-run
    python create_blocks.py --config blocks.scc.toml
    python create_blocks.py --config blocks.scc.toml --hpc-target local

For each stage that runs as a scheduler job this saves a ``CommandBlock`` (what
to run) and an ``ExecutionProfileBlock`` (how to run it), plus one
``HPCProfileBlock`` for the chosen target and, when the configuration names a
backend, a ``QiskitRuntimeConfig`` for the Qiskit Runtime sampling path.

Two decisions make one set of execution profiles serve both
``hpc_target="slurm"`` and ``hpc_target="local"``:

* the executable is the example's own venv Python, named in the HPC profile's
  ``executable_map``, and the worker script is the command block's first default
  argument. QCSC's Slurm template preflights the executable with ``[ ! -x ]``,
  which a venv Python passes, so no ``pre_commands`` are needed -- and local
  execution *rejects* ``modules``, ``pre_commands`` and ``gpus_per_node`` while
  honouring ``environments``. Everything environmental therefore goes in
  ``environments``.
* ``ompthreads`` renders only ``#SBATCH --cpus-per-task``, and the job script
  exports nothing but the profile's ``environments``. The thread variables are
  derived from ``ompthreads`` here so the two cannot drift apart; equal thread
  counts are what make the README's bitwise comparisons meaningful.

Python puts a script's own directory on ``sys.path[0]``, so ``import problem``
resolves in a worker even though the job runs with ``cwd`` set to its work
directory.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import block_names

HERE = Path(__file__).resolve().parent

WORKER_SCRIPTS = {
    "capture": "capture_worker.py",
    "prepare": "circuits_worker.py",
    "solve": "solve_worker.py",
    "replay": "replay_worker.py",
}

STAGE_DEFAULTS = {
    "capture": {"walltime": "00:30:00", "ompthreads": 8},
    "prepare": {"walltime": "00:30:00", "ompthreads": 8},
    "solve": {"walltime": "02:00:00", "ompthreads": 16},
    "replay": {"walltime": "00:30:00", "ompthreads": 8},
}


def stage_environment(
    ompthreads: int, extra: dict[str, str] | None = None
) -> dict[str, str]:
    """The environment a worker needs, with the thread counts from ``ompthreads``.

    ``PYTHONPATH`` and ``PYSCF_EXT_PATH`` are cleared so that no other copy of
    PySCF or mrh on the node can shadow the venv's, as
    ``examples/fefe_lassqd_pdft/run.slurm`` does. ``HDF5_USE_FILE_LOCKING`` is
    off because the records live on a shared parallel filesystem.
    """
    environment = {
        "OMP_NUM_THREADS": str(ompthreads),
        "OPENBLAS_NUM_THREADS": str(ompthreads),
        "MKL_NUM_THREADS": str(ompthreads),
        "PYTHONPATH": "",
        "PYSCF_EXT_PATH": "",
        "PYTHONUNBUFFERED": "1",
        "HDF5_USE_FILE_LOCKING": "FALSE",
    }
    environment.update({str(k): str(v) for k, v in (extra or {}).items()})
    return environment


def read_config(path: str | Path) -> dict:
    """Read the site configuration and fill in the defaults."""
    config = tomllib.loads(Path(path).read_text())
    unknown = sorted(
        set(config)
        - {
            "hpc_target",
            "block_prefix",
            "python",
            "worker_dir",
            "partition",
            "account",
            "memory",
            "runtime_backend",
            "runtime_channel",
            "runtime_instance",
            "environments",
            "stages",
        }
    )
    if unknown:
        raise ValueError(f"{path}: unknown settings {', '.join(unknown)}")
    config.setdefault("hpc_target", "slurm")
    config.setdefault("block_prefix", "lassqd")
    config.setdefault("worker_dir", str(HERE))
    config.setdefault(
        "python", str(Path(config["worker_dir"]) / ".venv" / "bin" / "python")
    )
    config.setdefault("partition", "cpu-debug")
    config.setdefault("account", "")
    config.setdefault("memory", None)
    config.setdefault("environments", {})
    stages = config.setdefault("stages", {})
    for stage in block_names.JOB_STAGES:
        settings = dict(STAGE_DEFAULTS[stage])
        settings.update(stages.get(stage, {}))
        stages[stage] = settings
    unknown_stages = sorted(set(stages) - set(block_names.JOB_STAGES))
    if unknown_stages:
        raise ValueError(f"{path}: unknown stages {', '.join(unknown_stages)}")
    return config


def build_blocks(config: dict, hpc_target: str) -> list[tuple[str, object]]:
    """The blocks to save, as ``(document name, block)`` in save order."""
    from qcsc_prefect_blocks.common.blocks import (
        CommandBlock,
        ExecutionProfileBlock,
        HPCProfileBlock,
    )

    prefix = config["block_prefix"]
    worker_dir = Path(config["worker_dir"]).resolve()
    blocks: list[tuple[str, object]] = []
    for stage in block_names.JOB_STAGES:
        stage_config = config["stages"][stage]
        command_name = block_names.command_name(stage, prefix)
        script = worker_dir / WORKER_SCRIPTS[stage]
        if not script.exists():
            raise FileNotFoundError(f"no worker script at {script}")
        blocks.append(
            (
                block_names.command(stage, prefix),
                CommandBlock(
                    command_name=command_name,
                    executable_key="python",
                    description=f"LASSQD {stage} worker ({script.name})",
                    default_args=[str(script)],
                ),
            )
        )
        blocks.append(
            (
                block_names.execution_profile(stage, prefix),
                ExecutionProfileBlock(
                    profile_name=f"{command_name}-cpu",
                    command_name=command_name,
                    resource_class="cpu",
                    num_nodes=1,
                    # P_f = 1: one task, no launcher. A collective solve raises
                    # num_nodes and sets launcher="srun"; see the README.
                    mpiprocs=1,
                    ompthreads=stage_config["ompthreads"],
                    gpus_per_node=None,
                    walltime=stage_config["walltime"],
                    launcher="single",
                    mpi_options=[],
                    modules=[],
                    pre_commands=[],
                    environments=stage_environment(
                        stage_config["ompthreads"], config["environments"]
                    ),
                ),
            )
        )
    blocks.append(
        (
            block_names.hpc_profile(hpc_target, prefix),
            HPCProfileBlock(
                hpc_target=hpc_target,
                queue_cpu=config["partition"],
                # scc needs no --account, and an empty string renders none.
                project_cpu=config["account"],
                executable_map={"python": config["python"]},
                slurm_qpu=None,
                slurm_memory=config["memory"],
                # Profile-wide, so it would apply to every stage. Leave it
                # unset and let mpiprocs set the task count.
                slurm_ntasks=None,
                slurm_extra_directives=[],
            ),
        )
    )
    if config.get("runtime_backend"):
        from qcsc_prefect.integrations.qiskit import QiskitRuntimeConfig

        blocks.append(
            (
                block_names.runtime_config(prefix),
                QiskitRuntimeConfig(
                    backend_name=config["runtime_backend"],
                    channel=config.get("runtime_channel"),
                    instance=config.get("runtime_instance"),
                ),
            )
        )
    return blocks


def describe(name: str, block: object) -> str:
    fields = block.model_dump(mode="json")
    rendered = "\n".join(f"    {key} = {value!r}" for key, value in fields.items())
    return f"  {type(block).__name__} '{name}'\n{rendered}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="a TOML site configuration")
    parser.add_argument(
        "--hpc-target",
        choices=("slurm", "local"),
        help="override the configuration's hpc_target",
    )
    parser.add_argument(
        "--ompthreads",
        type=int,
        help=(
            "use this thread count for every stage, overriding the "
            "configuration. The README's bitwise comparisons need 1: "
            "PySCF's OpenMP reductions make RHF and CCSD differ in the last bits "
            "between processes at more than one thread, which this example's "
            "sampler amplifies into different counts."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the blocks instead of saving them; needs no Prefect API",
    )
    args = parser.parse_args(argv)

    config = read_config(args.config)
    if args.ompthreads:
        for stage in config["stages"].values():
            stage["ompthreads"] = args.ompthreads
    hpc_target = args.hpc_target or config["hpc_target"]
    blocks = build_blocks(config, hpc_target)
    print(f"{len(blocks)} blocks for hpc_target={hpc_target!r} from {args.config}")
    for name, block in blocks:
        print(describe(name, block))
    if args.dry_run:
        print("--dry-run: nothing saved")
        return 0
    for name, block in blocks:
        # Saving a block document registers its block type, so the qcsc block
        # types need no separate registration step.
        block.save(name, overwrite=True)
        print(f"saved {type(block).__name__} '{name}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
