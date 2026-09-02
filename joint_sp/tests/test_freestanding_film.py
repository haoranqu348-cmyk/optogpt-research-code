import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from joint_sp.scripts.evaluate_freestanding_film import (  # noqa: E402
    parse_layers,
    simulate_freestanding,
    summarize,
)


def test_parse_layers_preserves_air_side_order():
    assert parse_layers("MgF2:80.5,Al2O3:120") == [
        ("MgF2", 80.5),
        ("Al2O3", 120.0),
    ]


def test_parse_layers_rejects_non_dielectric_material():
    with pytest.raises(ValueError, match="allowed dielectric"):
        parse_layers("Ag:50")


def test_lossless_freestanding_stack_conserves_power():
    wavelengths = np.array([400.0, 700.0, 1100.0])
    nk_dict = {
        "MgF2": np.full(3, 1.38 + 0.0j),
        "Al2O3": np.full(3, 1.65 + 0.0j),
    }
    results = simulate_freestanding(
        [("MgF2", 100.0), ("Al2O3", 100.0)], 60.0, wavelengths, nk_dict
    )
    metrics = summarize(results)
    for pol in ("s", "p"):
        np.testing.assert_allclose(results[pol]["R"] + results[pol]["T"], 1.0, atol=1e-12)
        np.testing.assert_allclose(results[pol]["A"], 0.0, atol=1e-12)
        assert metrics[f"max_energy_error_{pol}"] < 1e-12
