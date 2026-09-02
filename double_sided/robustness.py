"""Nominal and worst-case robustness for two-sided finite-glass designs."""

from dataclasses import replace

import numpy as np

from .contract import DoubleSidedStructure, Layer
from .physics import simulate_c, summarize


def perturb_side(structure, side, index, delta_nm):
    front, back = list(structure.front), list(structure.back)
    target = front if side == "front" else back
    layer = target[index]
    target[index] = Layer(layer.material, max(0.1, layer.thickness_nm + delta_nm))
    return DoubleSidedStructure(tuple(front), tuple(back))


def scale_nk(nk_dict, materials=(), substrate=None, n_scale=1.0, k_scale=1.0):
    selected = set(materials)
    if substrate:
        selected.add(substrate)
    output = {}
    for material, values in nk_dict.items():
        values = np.asarray(values, dtype=np.complex128)
        if material in selected:
            output[material] = values.real * n_scale + 1j * values.imag * k_scale
        else:
            output[material] = values.copy()
    return output


def evaluate_robustness(structure, nk_dict, config, random_trials=100, seed=20260728):
    scenarios = []

    def add(name, candidate=structure, candidate_nk=nk_dict, candidate_config=config):
        spectrum = simulate_c(candidate, candidate_nk, candidate_config)
        scenarios.append({"scenario": name, "metrics": summarize(spectrum), "spectrum": spectrum})

    add("nominal")
    for side, layers in (("front", structure.front), ("back", structure.back)):
        for index in range(len(layers)):
            for magnitude in (1.0, 2.0, 5.0):
                for sign in (-1.0, 1.0):
                    add(f"{side}_layer_{index}_{sign*magnitude:+g}nm",
                        perturb_side(structure, side, index, sign * magnitude))

    rng = np.random.RandomState(seed)
    for trial in range(random_trials):
        front = tuple(Layer(layer.material, max(0.1, layer.thickness_nm + rng.normal(0, 2.0)))
                      for layer in structure.front)
        back = tuple(Layer(layer.material, max(0.1, layer.thickness_nm + rng.normal(0, 2.0)))
                     for layer in structure.back)
        add(f"independent_random_2nm_{trial}", DoubleSidedStructure(front, back))

    for scale in (0.99, 1.01):
        add(f"glass_n_{scale:.2f}", candidate_nk=scale_nk(
            nk_dict, substrate=config.substrate, n_scale=scale
        ))
        coating_materials = {layer.material for layer in (*structure.front, *structure.back)}
        add(f"coating_nk_{scale:.2f}", candidate_nk=scale_nk(
            nk_dict, materials=coating_materials, n_scale=scale, k_scale=scale
        ))
    for angle in (59.0, 60.0, 61.0):
        add(f"angle_{angle:g}deg", candidate_config=replace(config, angle_deg=angle))

    worst = {}
    for pol in ("s", "p"):
        reflectances = np.stack([scenario["spectrum"][pol]["R"] for scenario in scenarios])
        transmissions = np.stack([scenario["spectrum"][pol]["T"] for scenario in scenarios])
        absorptions = np.stack([scenario["spectrum"][pol]["A"] for scenario in scenarios])
        worst[pol] = {
            "R": np.max(reflectances, axis=0),
            "T": np.min(transmissions, axis=0),
            "A": np.max(absorptions, axis=0),
        }
    return scenarios, worst, summarize(worst)
