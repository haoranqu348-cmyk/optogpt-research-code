import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from joint_sp.scripts.evaluate_double_sided_glass import (  # noqa: E402
    build_double_sided_stack,
    simulate_double_sided,
)


def test_double_sided_stack_mirrors_air_to_glass_layer_order():
    n_list, d_list, c_list = build_double_sided_stack(
        [(1.38, 130.0), (1.72, 210.0)], 1.5, 500000.0
    )
    assert n_list == [1.0, 1.38, 1.72, 1.5, 1.72, 1.38, 1.0]
    assert d_list == [np.inf, 130.0, 210.0, 500000.0, 210.0, 130.0, np.inf]
    assert c_list == ["i", "c", "c", "i", "c", "c", "i"]


def test_lossless_double_sided_stack_conserves_power():
    wavelengths = np.asarray([400.0, 700.0, 1100.0])
    nk_dict = {
        "MgF2": np.full(3, 1.38 + 0j),
        "MgO": np.full(3, 1.72 + 0j),
        "Glass_Substrate": np.full(3, 1.50 + 0j),
    }
    results = simulate_double_sided(
        [("MgF2", 130.0), ("MgO", 210.0)], 60.0, wavelengths, nk_dict, 500000.0
    )
    for pol in ("s", "p"):
        np.testing.assert_allclose(results[pol]["R"] + results[pol]["T"], 1.0, atol=1e-12)
