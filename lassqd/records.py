"""Records passed between the LASSQD stages.

The records import only NumPy and h5py, so they can be read and written without
mrh, PySCF or Qiskit installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import h5py
import numpy as np

from lassqd import _hdf5


def _frozen(array: np.ndarray) -> np.ndarray:
    array = np.array(array)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class LASState:
    """A LAS wave function: orbitals and the fragment RDMs in their active space.

    Inputs are copied into read-only arrays. RDMs given without a root axis get
    one, so every RDM has a leading root axis.

    Attributes:
        mo_coeff: Molecular orbital coefficients.
        casdm1frs: Per-fragment spin-separated 1-RDMs, shape ``(nroots, 2, n_i, n_i)``.
        casdm2fr: Per-fragment spin-summed 2-RDMs, shape ``(nroots, n_i, n_i, n_i, n_i)``.
    """

    FORMAT_VERSION = 1

    mo_coeff: np.ndarray
    casdm1frs: tuple[np.ndarray, ...]
    casdm2fr: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if len(self.casdm1frs) != len(self.casdm2fr):
            raise ValueError(
                f"{len(self.casdm1frs)} fragment 1-RDMs but {len(self.casdm2fr)} 2-RDMs"
            )
        dm1s = tuple(
            _frozen(np.reshape(dm, (-1,) + np.shape(dm)[-3:])) for dm in self.casdm1frs
        )
        dm2s = tuple(
            _frozen(np.reshape(dm, (-1,) + np.shape(dm)[-4:])) for dm in self.casdm2fr
        )
        object.__setattr__(self, "mo_coeff", _frozen(self.mo_coeff))
        object.__setattr__(self, "casdm1frs", dm1s)
        object.__setattr__(self, "casdm2fr", dm2s)

    @property
    def nfrags(self) -> int:
        return len(self.casdm1frs)

    def save(self, path: str | os.PathLike) -> None:
        """Write the state to an HDF5 file, overwriting it if it exists."""
        with _hdf5.create(path, "LASState", self.FORMAT_VERSION) as f:
            self._write(f)

    @classmethod
    def load(cls, path: str | os.PathLike) -> LASState:
        """Read a state written by :meth:`save`."""
        with _hdf5.open_record(path, "LASState", cls.FORMAT_VERSION) as f:
            return cls._read(f)

    def _write(self, group: h5py.Group) -> None:
        _hdf5.write_array_list(group, "casdm1frs", self.casdm1frs)
        _hdf5.write_array_list(group, "casdm2frs", self.casdm2fr)
        group.create_dataset("mo_coeff", data=self.mo_coeff)

    @classmethod
    def _read(cls, group: h5py.Group) -> LASState:
        return cls(
            group["mo_coeff"][()],
            tuple(_hdf5.read_array_list(group, "casdm1frs")),
            tuple(_hdf5.read_array_list(group, "casdm2frs")),
        )
