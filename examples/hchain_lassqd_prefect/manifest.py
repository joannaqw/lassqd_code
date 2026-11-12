"""The result manifest of one stage: the only evidence of its output.

Each worker writes its records and then publishes a manifest naming them with
their size and SHA-256. The flow verifies that manifest after QCSC reports the
job finished, because a scheduler state says the process ended, not that it
produced anything. Publication uses :func:`os.link`, which fails if a manifest
is already there, so at most one attempt's result is ever accepted.

The manifest records sizes and digests only. Semantic checks on the arrays --
RDM traces, the 2-RDM partial trace, an energy recomputed from the stored
Hamiltonian -- would attach in :func:`verify`; see the README's
"Not implemented" section. This module imports the standard library only.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import socket
from pathlib import Path

FORMAT = "lassqd-example-manifest-1"
_CHUNK = 1 << 20


def file_digest(path: str | os.PathLike) -> tuple[int, str]:
    """``(size, sha256)`` of ``path``, read in chunks."""
    sha = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            sha.update(chunk)
            size += len(chunk)
    return size, sha.hexdigest()


def attempt_id() -> str:
    """Identifier of this process's attempt at a logical job.

    The scheduler job identifier is included when the job runs under one, so a
    manifest can be traced back to a Slurm job without a separate claim record.
    """
    scheduler = os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID") or "-"
    return f"{socket.gethostname()}-{os.getpid()}-{scheduler}"


def write(
    path: str | os.PathLike,
    *,
    key: str,
    stage: str,
    cycle: int,
    fragment: int | None = None,
    files: list[str | os.PathLike],
    extra: dict | None = None,
) -> dict:
    """Publish the manifest of one logical job as its final write.

    Every referenced file is hashed first, so a manifest never names a file that
    was still being written. The manifest is written under a temporary name in
    the same directory, flushed, and then linked into place; linking fails if a
    manifest already exists, which is what makes publication single-shot.

    Args:
        path: Where to publish, normally ``RunLayout.manifest(...)``.
        key: The stage's logical key, from ``RunLayout.logical_key``.
        stage: Stage name.
        cycle: Macrocycle index.
        fragment: Fragment index for a per-fragment stage, else None.
        files: The records the stage produced. They are recorded relative to the
            manifest's directory, so moving a run directory keeps it verifiable.
        extra: Further fields to record, such as the stage's own diagnostics.

    Returns:
        The published manifest.

    Raises:
        FileExistsError: If a manifest for this logical job is already there.
    """
    path = Path(path)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name in files:
        name = Path(name)
        size, digest = file_digest(name)
        entries[os.path.relpath(name, directory)] = {"size": size, "sha256": digest}
    manifest = {
        "format": FORMAT,
        "key": key,
        "stage": stage,
        "cycle": cycle,
        "fragment": fragment,
        "attempt": attempt_id(),
        "published": datetime.datetime.now(datetime.UTC).isoformat(),
        "files": entries,
    }
    if extra:
        manifest.update(extra)
    temporary = directory / f"{path.name}.{os.getpid()}.tmp"
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    with open(temporary, "w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        # Link rather than replace: the first attempt to publish wins.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def clear(path: str | os.PathLike) -> None:
    """Remove a previous attempt's manifest so a fresh attempt can publish.

    Publication is single-shot, so the orchestrator -- not the worker -- decides
    that a logical job is to be run again, and clears the manifest first.
    Concurrent attempts are not reconciled here; see the README's
    "Not implemented" section.
    """
    Path(path).unlink(missing_ok=True)


def verify(path: str | os.PathLike, *, key: str | None = None) -> dict:
    """Check a published manifest and the files it names.

    Args:
        path: The manifest, normally ``RunLayout.manifest(...)``.
        key: Logical key the manifest must carry, if it is to be checked.

    Returns:
        The manifest.

    Raises:
        FileNotFoundError: If the manifest or one of its files is missing.
        ValueError: If the format or key differs, or a file's size or SHA-256
            does not match what the manifest records.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no result manifest at {path}: the job wrote no output, or failed "
            "before publishing one"
        )
    manifest = json.loads(path.read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{path}: manifest format {manifest.get('format')!r}")
    if key is not None and manifest.get("key") != key:
        raise ValueError(
            f"{path}: manifest is for logical job {manifest.get('key')!r}, "
            f"expected {key!r}"
        )
    for name, recorded in manifest["files"].items():
        referenced = path.parent / name
        if not referenced.exists():
            raise FileNotFoundError(f"{path}: references a missing file {referenced}")
        size, digest = file_digest(referenced)
        if (size, digest) != (recorded["size"], recorded["sha256"]):
            raise ValueError(
                f"{path}: {referenced} has size {size} and SHA-256 {digest}, but "
                f"the manifest records {recorded['size']} and {recorded['sha256']}"
            )
    return manifest
