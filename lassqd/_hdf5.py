"""HDF5 helpers for LASSQD records.

Each record file has a ``record`` attribute naming its type and a
``format_version`` attribute, checked on load. Records nest by writing into
subgroups.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import h5py
import numpy as np


def create(path: str | os.PathLike, record: str, version: int) -> h5py.File:
    """Create (or overwrite) a record file and write its header."""
    f = h5py.File(path, "w")
    write_header(f, record, version)
    return f


def open_record(path: str | os.PathLike, record: str, version: int) -> h5py.File:
    """Open a record file for reading after checking its header."""
    f = h5py.File(path, "r")
    try:
        check_header(f, record, version)
    except Exception:
        f.close()
        raise
    return f


def write_header(group: h5py.Group, record: str, version: int) -> None:
    group.attrs["record"] = record
    group.attrs["format_version"] = version


def check_header(group: h5py.Group, record: str, version: int) -> None:
    """Raise ValueError unless ``group`` holds ``record`` at ``version``."""
    found = group.attrs.get("record")
    if found != record:
        what = "no record type" if found is None else f"a {found} record"
        raise ValueError(f"{group.file.filename}: expected {record}, found {what}")
    found_version = group.attrs.get("format_version")
    if found_version != version:
        raise ValueError(
            f"{group.file.filename}: {record} format version {found_version!r} "
            f"is not supported (expected {version})"
        )


def write_array_list(
    group: h5py.Group, name: str, arrays: Sequence[np.ndarray]
) -> None:
    """Write ``arrays`` as datasets ``"0"``, ``"1"``, ... of subgroup ``name``."""
    sub = group.create_group(name)
    for i, array in enumerate(arrays):
        sub.create_dataset(str(i), data=array)


def read_array_list(group: h5py.Group, name: str) -> list[np.ndarray]:
    """Read the datasets written by :func:`write_array_list`, in index order."""
    sub = group[name]
    return [sub[key][()] for key in sorted(sub.keys(), key=int)]
