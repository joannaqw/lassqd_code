"""Serve the flow as a Prefect deployment.

::

    python deploy.py --params params.slurm.json

``flow.serve`` registers a deployment and then polls for runs of it, so leave
the process running. Start a run from the Prefect UI, or with::

    prefect deployment run 'lassqd-distributed/hchain-lassqd-prefect'

The deployment's default parameters come from the parameters file, and a run can
override them. ``exec.py`` runs the flow once in the foreground instead, which
is what the README's verification steps use.
"""

from __future__ import annotations

import argparse
import sys

from flow import lassqd_distributed
from flow_params import FlowParameters


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", required=True, help="default flow parameters")
    parser.add_argument("--name", default="hchain-lassqd-prefect")
    parser.add_argument("--run-dir", help="override the file's run_dir")
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="concurrent flow runs this server process will take",
    )
    args = parser.parse_args(argv)

    parameters = FlowParameters.load(args.params, run_dir=args.run_dir)
    lassqd_distributed.serve(
        name=args.name,
        parameters={"params": parameters.model_dump(mode="json")},
        limit=args.limit,
        description=(
            "Distributed LASSQD for an H20 chain: one Prefect task per stage, "
            "HDF5 records as the only stage boundary, QCSC launching the jobs."
        ),
        tags=["lassqd", parameters.hpc_target],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
