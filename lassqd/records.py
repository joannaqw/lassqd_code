"""Records passed between the LASSQD stages.

The records import only NumPy and h5py, so they can be read and written without
mrh, PySCF or Qiskit installed.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass

import h5py
import numpy as np

from lassqd import _hdf5


def _frozen(array: np.ndarray) -> np.ndarray:
    array = np.array(array)
    array.setflags(write=False)
    return array


def to_mo(
    mo_coeff: np.ndarray, h1: np.ndarray, h2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate one- and two-electron integrals into the ``mo_coeff`` basis.

    Args:
        mo_coeff: Orbital coefficients, shape ``(norb, norb)``; column ``i`` is
            orbital ``i``.
        h1: One-electron integrals, shape ``(norb, norb)``.
        h2: Two-electron integrals in chemists' notation, shape ``(norb,) * 4``.

    Returns:
        ``(h1_mo, h2_mo)``, the integrals in the ``mo_coeff`` basis.
    """
    C = mo_coeff
    h1_mo = np.einsum("pi,pr,rj->ij", C, h1, C, optimize=True)
    h2_mo = np.einsum("pi,rj,prqs,qk,sl->ijkl", C, C, h2, C, C, optimize=True)
    return h1_mo, h2_mo


def from_mo(
    mo_coeff: np.ndarray, dm1s: np.ndarray, dm2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate spin-separated 1-RDMs and a spin-summed 2-RDM out of the ``mo_coeff`` basis.

    ``mo_coeff`` is orthonormal, so its inverse is its transpose.

    Args:
        mo_coeff: Orthonormal orbital coefficients, shape ``(norb, norb)``.
        dm1s: Alpha and beta 1-RDMs in the ``mo_coeff`` basis, shape
            ``(2, norb, norb)``.
        dm2: Spin-summed 2-RDM in the ``mo_coeff`` basis, shape ``(norb,) * 4``.

    Returns:
        ``(dm1s, dm2)`` in the original basis, with the same shapes.
    """
    C = mo_coeff
    dm1s = C @ dm1s @ C.T
    dm2 = np.einsum("ip,jr,prqs,kq,ls->ijkl", C, C, dm2, C, C, optimize=True)
    return dm1s, dm2


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


@dataclass(frozen=True, eq=False)
class Fragment:
    """One fragment's Hamiltonian at a fixed embedding, and its sampling basis.

    The sampling basis is the fragment's ROHF orbitals. Circuits prepare states
    in it, with alpha orbitals on qubits ``0..norb-1`` and beta orbitals on
    ``norb..2*norb-1``, and SQD interprets measured occupations in it.

    Build fragments with :meth:`from_hamiltonian`, or for a LAS calculation with
    :func:`lassqd.las.capture_embedding`. Arrays are read-only copies.

    Attributes:
        index: Position of the fragment in its LAS calculation.
        norb: Number of fragment orbitals.
        nelec: ``(neleca, nelecb)``.
        h0: Constant energy term.
        h1s_las: Spin-separated one-electron integrals in the LAS basis, shape
            ``(2, norb, norb)``.
        h2_las: Two-electron integrals in the LAS basis, in chemists' notation,
            shape ``(norb,) * 4``.
        las_orbitals: The fragment's LAS orbitals in the AO basis, shape
            ``(nao, norb)``; the identity if the LAS basis is fixed.
        basis: ROHF orbital coefficients in the LAS basis, shape ``(norb, norb)``.
        mo_occ: ROHF occupations.
        mo_energy: ROHF orbital energies.
        rohf_converged: Whether the ROHF calculation converged.
        h1: Alpha one-electron integrals in the sampling basis. Circuits and SQD
            use only the alpha integrals, also for a spin-dependent embedding.
        h2: Two-electron integrals in the sampling basis.
    """

    index: int
    norb: int
    nelec: tuple[int, int]
    h0: float
    h1s_las: np.ndarray
    h2_las: np.ndarray
    las_orbitals: np.ndarray
    basis: np.ndarray
    mo_occ: np.ndarray
    mo_energy: np.ndarray
    rohf_converged: bool
    h1: np.ndarray
    h2: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "h1s_las",
            "h2_las",
            "las_orbitals",
            "basis",
            "mo_occ",
            "mo_energy",
            "h1",
            "h2",
        ):
            object.__setattr__(self, name, _frozen(getattr(self, name)))
        object.__setattr__(self, "index", int(self.index))
        object.__setattr__(self, "norb", int(self.norb))
        object.__setattr__(self, "nelec", tuple(int(n) for n in self.nelec))
        object.__setattr__(self, "h0", float(self.h0))
        object.__setattr__(self, "rohf_converged", bool(self.rohf_converged))
        if self.h1s_las.shape != (2, self.norb, self.norb):
            raise ValueError(
                f"h1s_las has shape {self.h1s_las.shape}, expected "
                f"{(2, self.norb, self.norb)}"
            )

    @classmethod
    def from_hamiltonian(
        cls,
        norb: int,
        nelec: tuple[int, int],
        h0: float,
        h1s: np.ndarray,
        h2: np.ndarray,
        *,
        index: int = 0,
        las_orbitals: np.ndarray | None = None,
    ) -> Fragment:
        """Find the fragment's ROHF orbitals and rotate its Hamiltonian into them.

        Args:
            norb: Number of fragment orbitals.
            nelec: ``(neleca, nelecb)``.
            h0: Constant energy term.
            h1s: One-electron integrals in the LAS basis, spin-separated with
                shape ``(2, norb, norb)`` or spin-free with shape ``(norb, norb)``.
            h2: Two-electron integrals in the LAS basis, in chemists' notation,
                shape ``(norb,) * 4``.
            index: Position of the fragment in its LAS calculation.
            las_orbitals: The fragment's LAS orbitals in the AO basis. None
                assumes a fixed LAS basis and uses the identity.

        Returns:
            The fragment, with ROHF run once on the alpha Hamiltonian.
        """
        from lassqd.fragment import fragment_rohf

        h1s = np.asarray(h1s)
        if h1s.ndim == 2:
            h1s = np.stack([h1s, h1s])
        if las_orbitals is None:
            las_orbitals = np.eye(norb)
        mf = fragment_rohf(h1s[0], h2, norb, nelec)
        h1, h2_mo = to_mo(mf.mo_coeff, h1s[0], h2)
        return cls(
            index=index,
            norb=norb,
            nelec=nelec,
            h0=h0,
            h1s_las=h1s,
            h2_las=h2,
            las_orbitals=las_orbitals,
            basis=mf.mo_coeff,
            mo_occ=mf.mo_occ,
            mo_energy=mf.mo_energy,
            rohf_converged=mf.converged,
            h1=h1,
            h2=h2_mo,
        )

    @property
    def sampling_orbitals(self) -> np.ndarray:
        """The sampling-basis orbitals in the AO basis, shape ``(nao, norb)``."""
        return self.las_orbitals @ self.basis

    @property
    def spin_asymmetry(self) -> float:
        """Largest ``|h1_alpha - h1_beta|`` element, which circuits and SQD ignore."""
        return float(np.max(np.abs(self.h1s_las[0] - self.h1s_las[1])))

    def rdms_to_las(
        self, dm1s: np.ndarray, dm2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rotate sampling-basis RDMs to the LAS basis; see :func:`from_mo`."""
        return from_mo(self.basis, dm1s, dm2)

    def digest(self) -> str:
        """SHA-256 of the fragment problem: sizes, ``h0`` and LAS-basis integrals."""
        return _digest(
            np.array([self.norb, *self.nelec]),
            np.array(self.h0),
            self.h1s_las,
            self.h2_las,
        )


def _digest(*arrays: np.ndarray) -> str:
    sha = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        sha.update(f"{array.dtype.str}{array.shape}".encode())
        sha.update(array.tobytes())
    return sha.hexdigest()


@dataclass(frozen=True, eq=False)
class Embedding:
    """The fragment problems of one LAS state, from :func:`lassqd.las.capture_embedding`.

    Attributes:
        state: Orbitals and fragment RDMs the fragment Hamiltonians were built from.
        fragments: One fragment per LAS fragment, in fragment order.
        ao_ovlp: AO overlap matrix, used to compare orbitals between embeddings.
    """

    state: LASState
    fragments: tuple[Fragment, ...]
    ao_ovlp: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "fragments", tuple(self.fragments))
        object.__setattr__(self, "ao_ovlp", _frozen(self.ao_ovlp))
        if len(self.fragments) != self.state.nfrags:
            raise ValueError(
                f"{len(self.fragments)} fragments for a state with "
                f"{self.state.nfrags} fragments"
            )
        for i, fragment in enumerate(self.fragments):
            if fragment.index != i:
                raise ValueError(f"fragment {i} has index {fragment.index}")


@dataclass(frozen=True, eq=False)
class FragmentSamples:
    """One cycle's per-fragment measurements, from :func:`lassqd.collect_samples`.

    Attributes:
        embedding: The fragment problems the circuits were built from. The
            counts' occupations refer to each fragment's sampling basis.
        counts: Each fragment's counts, in fragment order. The right half of each
            key holds the alpha occupations.
        job_ids: Identifier of the sampler job that ran each circuit, in circuit
            order, where known.
    """

    embedding: Embedding
    counts: tuple[Mapping[str, int], ...]
    job_ids: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts", tuple(dict(c) for c in self.counts))
        object.__setattr__(self, "job_ids", tuple(self.job_ids))
        if len(self.counts) != len(self.embedding.fragments):
            raise ValueError(
                f"{len(self.counts)} counts for "
                f"{len(self.embedding.fragments)} fragments"
            )
