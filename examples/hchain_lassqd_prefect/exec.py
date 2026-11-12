"""Run the flow once from a parameters file, without creating a deployment.

::

    python exec.py params.local.json
    python exec.py params.slurm.json --run-dir ~/runs/other --max-cycles 1

Save the blocks first with ``create_blocks.py``, and point ``PREFECT_API_URL``
at a running Prefect server. Use ``deploy.py`` to serve the flow instead.
"""

from __future__ import annotations

import argparse
import sys

from flow import lassqd_distributed
from flow_params import FlowParameters


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("params", help="a parameters file, such as params.local.json")
    parser.add_argument("--run-dir", help="override the file's run_dir")
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--sampler", choices=("runtime", "aer"))
    parser.add_argument("--hpc-target", choices=("slurm", "local"))
    args = parser.parse_args(argv)

    parameters = FlowParameters.load(
        args.params,
        run_dir=args.run_dir,
        max_cycles=args.max_cycles,
        sampler=args.sampler,
        hpc_target=args.hpc_target,
    )
    summary = lassqd_distributed(parameters)
    print(
        f"{'converged' if summary['converged'] else 'not converged'} after "
        f"{summary['cycles']} macrocycles: E = {summary['e_tot']:.10f} Hartree"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
