"""Records passed between the LASSQD stages.

The records import only NumPy and h5py, so they can be read and written without
mrh, PySCF or Qiskit installed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from lassqd import _hdf5

if TYPE_CHECKING:
    from qiskit_addon_sqd.fermion import SCIState


class _Record:
    """``save``/``load`` for a record type with ``_write``/``_read`` methods.

    Files and nested groups carry the record's class name and
    ``FORMAT_VERSION``, which are checked on reading.
    """

    FORMAT_VERSION = 1

    def save(self, path: str | os.PathLike) -> None:
        """Write the record to an HDF5 file, overwriting it if it exists."""
        with _hdf5.create(path, type(self).__name__, self.FORMAT_VERSION) as f:
            self._write(f)

    @classmethod
    def load(cls, path: str | os.PathLike):
        """Read a record written by :meth:`save`."""
        with _hdf5.open_record(path, cls.__name__, cls.FORMAT_VERSION) as f:
            return cls._read(f)

    def _write_child(self, group: h5py.Group, name: str, record: _Record) -> None:
        sub = group.create_group(name)
        _hdf5.write_header(sub, type(record).__name__, record.FORMAT_VERSION)
        record._write(sub)

    @staticmethod
    def _read_child(group: h5py.Group, name: str, cls: type[_Record]):
        sub = group[name]
        _hdf5.check_header(sub, cls.__name__, cls.FORMAT_VERSION)
        return cls._read(sub)

    def _write(self, group: h5py.Group) -> None:
        raise NotImplementedError

    @classmethod
    def _read(cls, group: h5py.Group):
        raise NotImplementedError


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
class LASState(_Record):
    """A LAS wave function: orbitals and the fragment RDMs in their active space.

    Inputs are copied into read-only arrays. RDMs given without a root axis get
    one, so every RDM has a leading root axis.

    Attributes:
        mo_coeff: Molecular orbital coefficients.
        casdm1frs: Per-fragment spin-separated 1-RDMs, shape ``(nroots, 2, n_i, n_i)``.
        casdm2fr: Per-fragment spin-summed 2-RDMs, shape ``(nroots, n_i, n_i, n_i, n_i)``.
    """

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
class Fragment(_Record):
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
        for name in self._ARRAYS:
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

    def rohf(self):
        """The ROHF solution in the sampling basis, as a PySCF object.

        See :func:`lassqd.fragment.sampling_basis_rohf`; for example,
        ``pyscf.cc.CCSD(fragment.rohf())`` runs CCSD in the sampling basis.
        """
        from lassqd.fragment import sampling_basis_rohf

        return sampling_basis_rohf(self)

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

    _ARRAYS = (
        "h1s_las",
        "h2_las",
        "las_orbitals",
        "basis",
        "mo_occ",
        "mo_energy",
        "h1",
        "h2",
    )

    def _write(self, group: h5py.Group) -> None:
        group.attrs["index"] = self.index
        group.attrs["norb"] = self.norb
        group.attrs["nelec"] = self.nelec
        group.attrs["h0"] = self.h0
        group.attrs["rohf_converged"] = self.rohf_converged
        for name in self._ARRAYS:
            group.create_dataset(name, data=getattr(self, name))

    @classmethod
    def _read(cls, group: h5py.Group) -> Fragment:
        return cls(
            index=group.attrs["index"],
            norb=group.attrs["norb"],
            nelec=tuple(group.attrs["nelec"]),
            h0=group.attrs["h0"],
            rohf_converged=group.attrs["rohf_converged"],
            **{name: group[name][()] for name in cls._ARRAYS},
        )


def _digest(*arrays: np.ndarray) -> str:
    sha = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        sha.update(f"{array.dtype.str}{array.shape}".encode())
        sha.update(array.tobytes())
    return sha.hexdigest()


@dataclass(frozen=True, eq=False)
class Embedding(_Record):
    """The fragment problems of one LAS state; see :func:`lassqd.las.capture_embedding`.

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

    def _write(self, group: h5py.Group) -> None:
        self._write_child(group, "state", self.state)
        fragments = group.create_group("fragments")
        for fragment in self.fragments:
            self._write_child(fragments, str(fragment.index), fragment)
        group.create_dataset("ao_ovlp", data=self.ao_ovlp)

    @classmethod
    def _read(cls, group: h5py.Group) -> Embedding:
        fragments = group["fragments"]
        return cls(
            cls._read_child(group, "state", LASState),
            [
                cls._read_child(fragments, key, Fragment)
                for key in sorted(fragments.keys(), key=int)
            ],
            group["ao_ovlp"][()],
        )


@dataclass(frozen=True, eq=False)
class FragmentSamples(_Record):
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

    def _write(self, group: h5py.Group) -> None:
        self._write_child(group, "embedding", self.embedding)
        # Keys keep their order: SQD draws samples by position.
        counts = group.create_group("counts")
        for i, fragment_counts in enumerate(self.counts):
            sub = counts.create_group(str(i))
            sub.create_dataset(
                "keys",
                data=np.array(list(fragment_counts), dtype=h5py.string_dtype()),
            )
            sub.create_dataset(
                "values", data=np.array(list(fragment_counts.values()), dtype=np.int64)
            )
        group.attrs["job_ids"] = json.dumps(list(self.job_ids))

    @classmethod
    def _read(cls, group: h5py.Group) -> FragmentSamples:
        counts = group["counts"]
        return cls(
            cls._read_child(group, "embedding", Embedding),
            [
                dict(
                    zip(
                        counts[key]["keys"].asstr()[()].tolist(),
                        counts[key]["values"][()].tolist(),
                    )
                )
                for key in sorted(counts.keys(), key=int)
            ],
            json.loads(group.attrs["job_ids"]),
        )


@dataclass(frozen=True, eq=False)
class SQDState(_Record):
    """What a :class:`lassqd.sqd.FragmentSQD` carries from one solve to the next.

    Attributes:
        rng_state: ``bit_generator.state`` of the random generator for the next
            solve.
        carryover_strings: Alpha and beta strings to include in the next solve,
            in the sampling basis of :attr:`carryover_orbitals`; None before the
            first solve with carryover.
        carryover_orbitals: AO coefficients of that sampling basis, shape
            ``(nao, norb)``, used to map the strings onto the next solve's basis.
    """

    rng_state: dict[str, Any]
    carryover_strings: tuple[np.ndarray, np.ndarray] | None = None
    carryover_orbitals: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rng_state", copy.deepcopy(self.rng_state))
        if (self.carryover_strings is None) != (self.carryover_orbitals is None):
            raise ValueError(
                "carryover_strings and carryover_orbitals must be given together"
            )
        if self.carryover_strings is not None:
            strings_a, strings_b = self.carryover_strings
            object.__setattr__(
                self, "carryover_strings", (_frozen(strings_a), _frozen(strings_b))
            )
            object.__setattr__(
                self, "carryover_orbitals", _frozen(self.carryover_orbitals)
            )

    @classmethod
    def from_seed(cls, seed: int | np.random.Generator | None = None) -> SQDState:
        """The initial state for ``seed``; a generator is copied, not shared."""
        if not isinstance(seed, np.random.Generator):
            seed = np.random.default_rng(seed)
        return cls(seed.bit_generator.state)

    def rng(self) -> np.random.Generator:
        """A new generator in the stored state."""
        bit_generator = getattr(np.random, self.rng_state["bit_generator"])()
        bit_generator.state = self.rng_state
        return np.random.Generator(bit_generator)

    def _write(self, group: h5py.Group) -> None:
        group.attrs["rng_state"] = _dump_rng_state(self.rng_state)
        if self.carryover_strings is not None:
            group.create_dataset("carryover_a", data=self.carryover_strings[0])
            group.create_dataset("carryover_b", data=self.carryover_strings[1])
            group.create_dataset("carryover_orbitals", data=self.carryover_orbitals)

    @classmethod
    def _read(cls, group: h5py.Group) -> SQDState:
        carryover_strings = carryover_orbitals = None
        if "carryover_orbitals" in group:
            carryover_strings = (group["carryover_a"][()], group["carryover_b"][()])
            carryover_orbitals = group["carryover_orbitals"][()]
        return cls(
            _load_rng_state(group.attrs["rng_state"]),
            carryover_strings,
            carryover_orbitals,
        )


def _dump_rng_state(state: dict[str, Any]) -> str:
    """JSON for a ``bit_generator.state``, exact for its integers and arrays."""

    def encode(value):
        if isinstance(value, np.ndarray):
            return {"__ndarray__": value.tolist(), "dtype": value.dtype.str}
        raise TypeError(f"cannot store {type(value).__name__} in a generator state")

    return json.dumps(state, default=encode)


def _load_rng_state(text: str) -> dict[str, Any]:
    def decode(value):
        if "__ndarray__" in value:
            return np.array(value["__ndarray__"], dtype=value["dtype"])
        return value

    return json.loads(text, object_hook=decode)


@dataclass(frozen=True, eq=False)
class FragmentResult(_Record):
    """One fragment's SQD solution, from :meth:`lassqd.sqd.FragmentSQD.solve`.

    Attributes:
        e_tot: Fragment energy, including ``h0``.
        dm1s: Spin-separated 1-RDMs in the LAS basis, shape ``(2, norb, norb)``.
        dm2: Spin-summed 2-RDM in the LAS basis, shape ``(norb,) * 4``.
        amplitudes: Amplitudes of the lowest-energy SCI state, shape
            ``(len(ci_strs_a), len(ci_strs_b))``, in the sampling basis.
        ci_strs_a: Alpha determinant strings of that state.
        ci_strs_b: Beta determinant strings of that state.
        norb: Number of fragment orbitals.
        nelec: ``(neleca, nelecb)``.
        fragment_digest: :meth:`Fragment.digest` of the solved fragment.
    """

    e_tot: float
    dm1s: np.ndarray
    dm2: np.ndarray
    amplitudes: np.ndarray
    ci_strs_a: np.ndarray
    ci_strs_b: np.ndarray
    norb: int
    nelec: tuple[int, int]
    fragment_digest: str

    def __post_init__(self) -> None:
        for name in self._ARRAYS:
            object.__setattr__(self, name, _frozen(getattr(self, name)))
        object.__setattr__(self, "e_tot", float(self.e_tot))
        object.__setattr__(self, "norb", int(self.norb))
        object.__setattr__(self, "nelec", tuple(int(n) for n in self.nelec))

    @property
    def sci_state(self) -> SCIState:
        """The SCI state as a ``qiskit_addon_sqd.fermion.SCIState``."""
        from qiskit_addon_sqd.fermion import SCIState

        return SCIState(
            amplitudes=self.amplitudes,
            ci_strs_a=self.ci_strs_a,
            ci_strs_b=self.ci_strs_b,
            norb=self.norb,
            nelec=self.nelec,
        )

    _ARRAYS = ("dm1s", "dm2", "amplitudes", "ci_strs_a", "ci_strs_b")

    def _write(self, group: h5py.Group) -> None:
        group.attrs["e_tot"] = self.e_tot
        group.attrs["norb"] = self.norb
        group.attrs["nelec"] = self.nelec
        group.attrs["fragment_digest"] = self.fragment_digest
        for name in self._ARRAYS:
            group.create_dataset(name, data=getattr(self, name))

    @classmethod
    def _read(cls, group: h5py.Group) -> FragmentResult:
        return cls(
            e_tot=group.attrs["e_tot"],
            norb=group.attrs["norb"],
            nelec=tuple(group.attrs["nelec"]),
            fragment_digest=group.attrs["fragment_digest"],
            **{name: group[name][()] for name in cls._ARRAYS},
        )
