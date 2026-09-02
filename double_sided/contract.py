"""Canonical two-sided structure and leak-resistant hash contract."""

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Sequence, Tuple

import numpy as np


SIDE_SEP = "SIDE_SEP"
SPECIAL_TOKENS = ("UNK", "PAD", "BOS", "EOS", SIDE_SEP)


@dataclass(frozen=True)
class Layer:
    material: str
    thickness_nm: float

    def __post_init__(self):
        if not self.material or self.material in SPECIAL_TOKENS or "_" in self.material:
            raise ValueError(f"Invalid material: {self.material!r}")
        if not np.isfinite(self.thickness_nm) or self.thickness_nm <= 0:
            raise ValueError("Layer thickness must be finite and positive")


def merge_adjacent(layers: Iterable[Layer]) -> Tuple[Layer, ...]:
    merged = []
    for layer in layers:
        if merged and merged[-1].material == layer.material:
            previous = merged[-1]
            merged[-1] = Layer(previous.material, previous.thickness_nm + layer.thickness_nm)
        else:
            merged.append(layer)
    return tuple(merged)


@dataclass(frozen=True)
class DoubleSidedStructure:
    front: Tuple[Layer, ...]
    back: Tuple[Layer, ...]

    def __post_init__(self):
        object.__setattr__(self, "front", tuple(self.front))
        object.__setattr__(self, "back", tuple(self.back))
        if not self.front or not self.back:
            raise ValueError("Front and back coatings must both be non-empty")

    @classmethod
    def from_tokens(cls, tokens, allowed_materials, max_layers_per_side):
        values = list(tokens)
        if values.count(SIDE_SEP) != 1:
            raise ValueError("SIDE_SEP must occur exactly once")
        if not values or values[0] != "BOS" or values[-1] != "EOS":
            raise ValueError("Sequence must begin with BOS and end with EOS")
        if any(token in ("UNK", "PAD") for token in values):
            raise ValueError("UNK and PAD are not physical structure tokens")
        separator = values.index(SIDE_SEP)
        front_tokens, back_tokens = values[1:separator], values[separator + 1:-1]
        if not front_tokens or not back_tokens:
            raise ValueError("Neither coating side may be empty")
        if max(len(front_tokens), len(back_tokens)) > max_layers_per_side:
            raise ValueError("Per-side technical layer limit exceeded")
        allowed = set(allowed_materials)

        def parse(side):
            parsed = []
            for token in side:
                if token in SPECIAL_TOKENS or "_" not in token:
                    raise ValueError(f"Invalid layer token: {token!r}")
                material, thickness = token.rsplit("_", 1)
                if material not in allowed:
                    raise ValueError(f"Material {material!r} is not enabled")
                try:
                    value = float(thickness)
                except ValueError as exc:
                    raise ValueError(f"Invalid thickness token: {token!r}") from exc
                parsed.append(Layer(material, value))
            return tuple(parsed)

        return cls(parse(front_tokens), parse(back_tokens))

    def merged(self):
        return DoubleSidedStructure(merge_adjacent(self.front), merge_adjacent(self.back))

    @property
    def physical_layer_counts(self):
        merged = self.merged()
        return len(merged.front), len(merged.back)

    def to_tokens(self, thickness_format="g"):
        def token(layer):
            return f"{layer.material}_{format(layer.thickness_nm, thickness_format)}"
        return ["BOS", *map(token, self.front), SIDE_SEP, *map(token, self.back), "EOS"]

    def canonical_payload(self, merge=True, decimals=9):
        structure = self.merged() if merge else self

        def side(layers):
            return [(layer.material, round(float(layer.thickness_nm), decimals)) for layer in layers]

        return {"front": side(structure.front), "back": side(structure.back)}

    def physical_hash(self):
        raw = json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("ascii")).hexdigest()

    def split_group_hash(self):
        """Group exact, merged-equivalent, and front/back mirror structures together."""
        direct = self.canonical_payload()
        mirrored = {
            "front": list(reversed(direct["back"])),
            "back": list(reversed(direct["front"])),
        }
        variants = [
            json.dumps(direct, sort_keys=True, separators=(",", ":")),
            json.dumps(mirrored, sort_keys=True, separators=(",", ":")),
        ]
        return hashlib.sha256(min(variants).encode("ascii")).hexdigest()


def assign_split(group_hash, ratios=(0.8, 0.1, 0.1)):
    if len(group_hash) != 64 or not np.isclose(sum(ratios), 1.0):
        raise ValueError("Invalid hash or split ratios")
    value = int(group_hash[:16], 16) / float(16 ** 16)
    if value < ratios[0]:
        return "train"
    if value < ratios[0] + ratios[1]:
        return "dev"
    return "test"
