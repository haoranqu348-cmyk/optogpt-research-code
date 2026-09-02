"""Immutable canonical records and stable structure hashing.

This module intentionally uses only the Python standard library so record and
hash operations do not pull GPU or model dependencies into orchestration code.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Iterable, Mapping, Sequence

ALLOWED_MATERIALS = frozenset(
    {"Al2O3", "AlN", "HfO2", "MgF2", "MgO", "Si3N4", "SiO2", "Ta2O5", "TiO2", "ZnO"}
)
MIN_LAYERS = 1
MAX_LAYERS = 20
THICKNESS_MIN_NM = 10
THICKNESS_MAX_NM = 300
THICKNESS_STEP_NM = 10
N_WAVELENGTHS = 71
SPECTRUM_LAYOUT = ("Rs", "Ts", "Rp", "Tp")
STRUCTURE_HASH_SCHEMA = "optogpt.active_learning.structure.v1"


def _require_nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _canonical_thickness(value: Any, context: str = "thickness_nm") -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context} must be a finite number in nanometers, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{context} must be an integer number of nanometers, got {value!r}")
    thickness = int(numeric)
    if not THICKNESS_MIN_NM <= thickness <= THICKNESS_MAX_NM:
        raise ValueError(
            f"{context}={thickness} is outside [{THICKNESS_MIN_NM}, {THICKNESS_MAX_NM}] nm"
        )
    if (thickness - THICKNESS_MIN_NM) % THICKNESS_STEP_NM:
        raise ValueError(
            f"{context}={thickness} is not on the {THICKNESS_STEP_NM} nm grid "
            f"starting at {THICKNESS_MIN_NM} nm"
        )
    return thickness


@dataclass(frozen=True, slots=True)
class Layer:
    material: str
    thickness_nm: int

    def __post_init__(self) -> None:
        material = _require_nonempty_text(self.material, "material")
        if material not in ALLOWED_MATERIALS:
            allowed = ", ".join(sorted(ALLOWED_MATERIALS))
            raise ValueError(f"material {material!r} is not allowed; expected one of: {allowed}")
        object.__setattr__(self, "thickness_nm", _canonical_thickness(self.thickness_nm))

    def to_dict(self) -> dict[str, Any]:
        return {"material": self.material, "thickness_nm": self.thickness_nm}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Layer":
        return cls(material=value["material"], thickness_nm=value["thickness_nm"])


@dataclass(frozen=True, slots=True)
class MultilayerStructure:
    layers: tuple[Layer, ...]

    def __post_init__(self) -> None:
        layers = tuple(self.layers)
        if not MIN_LAYERS <= len(layers) <= MAX_LAYERS:
            raise ValueError(
                f"layer count {len(layers)} is outside [{MIN_LAYERS}, {MAX_LAYERS}]"
            )
        for index, layer in enumerate(layers):
            if not isinstance(layer, Layer):
                raise ValueError(f"layers[{index}] must be a Layer, got {type(layer).__name__}")
        object.__setattr__(self, "layers", layers)

    @classmethod
    def from_materials(
        cls, materials: Sequence[str], thicknesses_nm: Sequence[Real]
    ) -> "MultilayerStructure":
        if len(materials) != len(thicknesses_nm):
            raise ValueError(
                f"materials/thickness length mismatch: {len(materials)} != {len(thicknesses_nm)}"
            )
        return cls(tuple(Layer(material, thickness) for material, thickness in zip(materials, thicknesses_nm)))

    @classmethod
    def from_legacy_tokens(cls, tokens: Iterable[str]) -> "MultilayerStructure":
        return parse_legacy_tokens(tokens)

    @property
    def structure_hash(self) -> str:
        payload = {
            "layers": [[layer.material, layer.thickness_nm] for layer in self.layers],
            "schema": STRUCTURE_HASH_SCHEMA,
        }
        canonical_bytes = json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
        return hashlib.sha256(canonical_bytes).hexdigest()

    def to_legacy_tokens(self) -> tuple[str, ...]:
        return tuple(f"{layer.material}_{layer.thickness_nm}" for layer in self.layers)

    def to_dict(self) -> dict[str, Any]:
        return {"layers": [layer.to_dict() for layer in self.layers]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MultilayerStructure":
        return cls(tuple(Layer.from_dict(layer) for layer in value["layers"]))


def parse_legacy_tokens(tokens: Iterable[str]) -> MultilayerStructure:
    """Adapt existing ``material_thickness`` token lists to the canonical record."""
    clean_tokens = tuple(tokens)
    forbidden = {"BOS", "EOS", "PAD", "UNK", ""}
    layers = []
    for index, token in enumerate(clean_tokens):
        if not isinstance(token, str):
            raise ValueError(f"tokens[{index}] must be a string, got {type(token).__name__}")
        if token in forbidden:
            raise ValueError(f"tokens[{index}] contains forbidden special token {token!r}")
        try:
            material, thickness_text = token.rsplit("_", 1)
        except ValueError as exc:
            raise ValueError(
                f"tokens[{index}]={token!r} must use 'material_thickness' format"
            ) from exc
        try:
            thickness = int(thickness_text)
        except ValueError as exc:
            raise ValueError(f"tokens[{index}]={token!r} has a non-integer thickness") from exc
        if thickness_text != str(thickness):
            raise ValueError(f"tokens[{index}]={token!r} is not in canonical decimal form")
        layers.append(Layer(material, thickness))
    return MultilayerStructure(tuple(layers))


def _spectrum_component(values: Iterable[Real], name: str) -> tuple[float, ...]:
    result = tuple(values)
    if len(result) != N_WAVELENGTHS:
        raise ValueError(f"{name} has {len(result)} values; expected exactly {N_WAVELENGTHS}")
    canonical = []
    for index, value in enumerate(result):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name}[{index}] must be a finite real number, got {value!r}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}] must be finite, got {value!r}")
        canonical.append(number)
    return tuple(canonical)


@dataclass(frozen=True, slots=True)
class JointSpectrum:
    """Joint label ordered as 71-value components ``Rs, Ts, Rp, Tp``."""

    rs: tuple[float, ...]
    ts: tuple[float, ...]
    rp: tuple[float, ...]
    tp: tuple[float, ...]

    def __post_init__(self) -> None:
        for field_name, layout_name in zip(("rs", "ts", "rp", "tp"), SPECTRUM_LAYOUT):
            object.__setattr__(self, field_name, _spectrum_component(getattr(self, field_name), layout_name))

    @classmethod
    def from_flat(cls, values: Sequence[Real]) -> "JointSpectrum":
        expected = len(SPECTRUM_LAYOUT) * N_WAVELENGTHS
        if len(values) != expected:
            raise ValueError(
                f"flat joint spectrum has {len(values)} values; expected {expected} in "
                f"{list(SPECTRUM_LAYOUT)} order"
            )
        parts = [tuple(values[i : i + N_WAVELENGTHS]) for i in range(0, expected, N_WAVELENGTHS)]
        return cls(*parts)

    def to_flat(self) -> tuple[float, ...]:
        return self.rs + self.ts + self.rp + self.tp

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout": list(SPECTRUM_LAYOUT),
            "rs": list(self.rs),
            "ts": list(self.ts),
            "rp": list(self.rp),
            "tp": list(self.tp),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JointSpectrum":
        layout = tuple(value.get("layout", ()))
        if layout != SPECTRUM_LAYOUT:
            raise ValueError(f"spectrum layout {layout!r} does not match required {SPECTRUM_LAYOUT!r}")
        return cls(tuple(value["rs"]), tuple(value["ts"]), tuple(value["rp"]), tuple(value["tp"]))


@dataclass(frozen=True, slots=True)
class CandidateProvenance:
    generator: str
    round_index: int
    candidate_id: str | None = None
    parent_hashes: tuple[str, ...] = ()
    parameters: tuple[tuple[str, str | int | float | bool | None], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "generator", _require_nonempty_text(self.generator, "generator"))
        if isinstance(self.round_index, bool) or not isinstance(self.round_index, Integral) or self.round_index < 0:
            raise ValueError(f"round_index must be a non-negative integer, got {self.round_index!r}")
        object.__setattr__(self, "round_index", int(self.round_index))
        if self.candidate_id is not None:
            object.__setattr__(self, "candidate_id", _require_nonempty_text(self.candidate_id, "candidate_id"))
        parent_hashes = tuple(self.parent_hashes)
        for index, value in enumerate(parent_hashes):
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"parent_hashes[{index}] must be a lowercase SHA-256 hex digest")
        object.__setattr__(self, "parent_hashes", parent_hashes)

        parameters = tuple(tuple(item) for item in self.parameters)
        if any(len(item) != 2 for item in parameters):
            raise ValueError("parameters must contain (name, scalar_value) pairs")
        keys = [item[0] for item in parameters]
        for key in keys:
            _require_nonempty_text(key, "parameter name")
        if len(set(keys)) != len(keys):
            raise ValueError("parameters must contain unique (name, scalar_value) pairs")
        for key, value in parameters:
            if not isinstance(value, (str, int, float, bool, type(None))) or (
                isinstance(value, float) and not math.isfinite(value)
            ):
                raise ValueError(f"parameter {key!r} must be a finite JSON scalar, got {value!r}")
        object.__setattr__(self, "parameters", tuple(sorted(parameters)))

    @classmethod
    def from_parameters(
        cls,
        generator: str,
        round_index: int,
        parameters: Mapping[str, str | int | float | bool | None],
        **kwargs: Any,
    ) -> "CandidateProvenance":
        return cls(generator, round_index, parameters=tuple(parameters.items()), **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generator": self.generator,
            "round_index": self.round_index,
            "candidate_id": self.candidate_id,
            "parent_hashes": list(self.parent_hashes),
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateProvenance":
        return cls.from_parameters(
            generator=value["generator"],
            round_index=value["round_index"],
            candidate_id=value.get("candidate_id"),
            parent_hashes=tuple(value.get("parent_hashes", ())),
            parameters=value.get("parameters", {}),
        )


@dataclass(frozen=True, slots=True)
class LabeledSample:
    structure: MultilayerStructure
    spectrum: JointSpectrum
    provenance: CandidateProvenance
    structure_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.structure, MultilayerStructure):
            raise ValueError("structure must be a MultilayerStructure")
        if not isinstance(self.spectrum, JointSpectrum):
            raise ValueError("spectrum must be a JointSpectrum")
        if not isinstance(self.provenance, CandidateProvenance):
            raise ValueError("provenance must be a CandidateProvenance")
        object.__setattr__(self, "structure_hash", self.structure.structure_hash)

    def to_dict(self) -> dict[str, Any]:
        return {
            "structure_hash": self.structure_hash,
            "structure": self.structure.to_dict(),
            "spectrum": self.spectrum.to_dict(),
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LabeledSample":
        sample = cls(
            structure=MultilayerStructure.from_dict(value["structure"]),
            spectrum=JointSpectrum.from_dict(value["spectrum"]),
            provenance=CandidateProvenance.from_dict(value["provenance"]),
        )
        recorded_hash = value.get("structure_hash")
        if recorded_hash != sample.structure_hash:
            raise ValueError(
                f"manifest structure_hash {recorded_hash!r} does not match canonical hash "
                f"{sample.structure_hash!r}"
            )
        return sample
