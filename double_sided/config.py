"""Configuration for staged double-sided finite-glass experiments."""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


BASE_MATERIALS = (
    "Al2O3", "AlN", "HfO2", "MgF2", "MgO",
    "Si3N4", "SiO2", "Ta2O5", "TiO2", "ZnO",
)


@dataclass(frozen=True)
class LayerStage:
    minimum: int
    maximum: int

    def __post_init__(self):
        if self.minimum < 1 or self.maximum < self.minimum:
            raise ValueError("Each layer stage must satisfy 1 <= minimum <= maximum")


@dataclass
class DoubleSidedConfig:
    wavelengths_nm: np.ndarray = field(
        default_factory=lambda: np.arange(400.0, 1101.0, 10.0)
    )
    angle_deg: float = 60.0
    substrate: str = "Glass_Substrate"
    substrate_thickness_nm: float = 500_000.0
    layer_stages: List[LayerStage] = field(default_factory=lambda: [
        LayerStage(1, 4), LayerStage(5, 8),
        LayerStage(9, 16), LayerStage(17, 32),
    ])
    technical_max_layers_per_side: int = 32
    min_thickness_nm: float = 10.0
    max_thickness_nm: float = 500.0
    token_thickness_step_nm: float = 10.0
    stage_improvement_threshold: float = 1e-3
    consecutive_stalled_stages: int = 2
    allowed_materials: Tuple[str, ...] = BASE_MATERIALS

    def validate(self, require_truth_grid=True):
        wavelengths = np.asarray(self.wavelengths_nm, dtype=float)
        if (require_truth_grid and (wavelengths.shape != (71,) or not np.array_equal(
                wavelengths, np.arange(400.0, 1101.0, 10.0)))):
            raise ValueError("The physical truth grid must be 400..1100 nm in 10 nm steps")
        if (not require_truth_grid and
                (wavelengths.ndim != 1 or len(wavelengths) < 2 or np.any(np.diff(wavelengths) <= 0))):
            raise ValueError("Search wavelengths must be a strictly increasing one-dimensional grid")
        if self.technical_max_layers_per_side < max(s.maximum for s in self.layer_stages):
            raise ValueError("technical_max_layers_per_side must cover every configured stage")
        if self.min_thickness_nm <= 0 or self.max_thickness_nm <= self.min_thickness_nm:
            raise ValueError("Invalid thickness bounds")
        if (self.token_thickness_step_nm <= 0 or
                not np.isclose(self.min_thickness_nm / self.token_thickness_step_nm,
                               round(self.min_thickness_nm / self.token_thickness_step_nm)) or
                not np.isclose(self.max_thickness_nm / self.token_thickness_step_nm,
                               round(self.max_thickness_nm / self.token_thickness_step_nm))):
            raise ValueError("Thickness bounds must align to the token thickness step")
        if not self.allowed_materials:
            raise ValueError("At least one material is required")
        return self

    @property
    def technical_max_tokens(self):
        # BOS + front + SIDE_SEP + back + EOS
        return 2 * self.technical_max_layers_per_side + 3

    def should_stop(self, stage_best_values):
        """Stop only after two consecutive stage improvements are below threshold."""
        if len(stage_best_values) < self.consecutive_stalled_stages + 1:
            return False
        improvements = [
            max(0.0, float(stage_best_values[i - 1]) - float(stage_best_values[i]))
            for i in range(1, len(stage_best_values))
        ]
        return all(
            value < self.stage_improvement_threshold
            for value in improvements[-self.consecutive_stalled_stages:]
        )
