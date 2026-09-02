"""Lightweight JSON and JSONL serialization for canonical labeled samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .records import LabeledSample

MANIFEST_SCHEMA = "optogpt.active_learning.labeled_samples.v1"


def dumps_manifest(samples: Iterable[LabeledSample], *, indent: int | None = 2) -> str:
    payload = {"schema": MANIFEST_SCHEMA, "samples": [sample.to_dict() for sample in samples]}
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=indent, allow_nan=False)


def loads_manifest(text: str) -> tuple[LabeledSample, ...]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON manifest: {exc}") from exc
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported manifest schema {payload.get('schema')!r}; expected {MANIFEST_SCHEMA!r}")
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("manifest 'samples' must be a list")
    return tuple(LabeledSample.from_dict(sample) for sample in samples)


def dump_manifest(samples: Iterable[LabeledSample], path: str | Path) -> None:
    Path(path).write_text(dumps_manifest(samples) + "\n", encoding="utf-8", newline="\n")


def load_manifest(path: str | Path) -> tuple[LabeledSample, ...]:
    return loads_manifest(Path(path).read_text(encoding="utf-8"))


def write_jsonl_manifest(samples: Iterable[LabeledSample], path: str | Path) -> None:
    lines = []
    for sample in samples:
        record = {"schema": MANIFEST_SCHEMA, "sample": sample.to_dict()}
        lines.append(json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False))
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")


def read_jsonl_manifest(path: str | Path) -> tuple[LabeledSample, ...]:
    samples = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
        if record.get("schema") != MANIFEST_SCHEMA:
            raise ValueError(f"line {line_number}: unsupported schema {record.get('schema')!r}")
        try:
            samples.append(LabeledSample.from_dict(record["sample"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"line {line_number}: invalid sample: {exc}") from exc
    return tuple(samples)
