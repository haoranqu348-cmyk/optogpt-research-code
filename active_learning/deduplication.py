"""First-seen batch deduplication and labeled-set exclusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .records import LabeledSample, MultilayerStructure

HashableRecord = MultilayerStructure | LabeledSample


def _record_hash(record: HashableRecord, index: int) -> str:
    if isinstance(record, MultilayerStructure):
        return record.structure_hash
    if isinstance(record, LabeledSample):
        return record.structure_hash
    raise ValueError(
        f"batch[{index}] must be a MultilayerStructure or LabeledSample, "
        f"got {type(record).__name__}"
    )


@dataclass(frozen=True, slots=True)
class DeduplicationResult:
    unique: tuple[HashableRecord, ...]
    duplicate_hashes: tuple[str, ...]
    excluded_hashes: tuple[str, ...]

    @property
    def removed_count(self) -> int:
        return len(self.duplicate_hashes) + len(self.excluded_hashes)


def deduplicate_batch(
    batch: Iterable[HashableRecord], existing_labeled_hashes: Iterable[str] = ()
) -> DeduplicationResult:
    """Keep first occurrence, separating within-batch duplicates from exclusions."""
    existing = set(existing_labeled_hashes)
    for value in existing:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(
                f"existing labeled hash must be a lowercase SHA-256 hex digest, got {value!r}"
            )

    unique = []
    duplicates = []
    excluded = []
    batch_seen = set()
    for index, record in enumerate(batch):
        structure_hash = _record_hash(record, index)
        if structure_hash in existing:
            excluded.append(structure_hash)
        elif structure_hash in batch_seen:
            duplicates.append(structure_hash)
        else:
            unique.append(record)
            batch_seen.add(structure_hash)
    return DeduplicationResult(tuple(unique), tuple(duplicates), tuple(excluded))
