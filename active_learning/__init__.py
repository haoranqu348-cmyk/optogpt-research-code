"""Canonical data contracts for the active-learning pipeline."""

from .deduplication import DeduplicationResult, deduplicate_batch
from .manifests import (
    dump_manifest,
    dumps_manifest,
    load_manifest,
    loads_manifest,
    read_jsonl_manifest,
    write_jsonl_manifest,
)
from .records import (
    ALLOWED_MATERIALS,
    MAX_LAYERS,
    MIN_LAYERS,
    N_WAVELENGTHS,
    SPECTRUM_LAYOUT,
    THICKNESS_MAX_NM,
    THICKNESS_MIN_NM,
    THICKNESS_STEP_NM,
    CandidateProvenance,
    JointSpectrum,
    LabeledSample,
    Layer,
    MultilayerStructure,
    parse_legacy_tokens,
)

__all__ = [
    "ALLOWED_MATERIALS",
    "MAX_LAYERS",
    "MIN_LAYERS",
    "N_WAVELENGTHS",
    "SPECTRUM_LAYOUT",
    "THICKNESS_MAX_NM",
    "THICKNESS_MIN_NM",
    "THICKNESS_STEP_NM",
    "CandidateProvenance",
    "DeduplicationResult",
    "JointSpectrum",
    "LabeledSample",
    "Layer",
    "MultilayerStructure",
    "deduplicate_batch",
    "dump_manifest",
    "dumps_manifest",
    "load_manifest",
    "loads_manifest",
    "parse_legacy_tokens",
    "read_jsonl_manifest",
    "write_jsonl_manifest",
]
