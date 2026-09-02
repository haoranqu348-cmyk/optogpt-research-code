"""Generate a small real-inc_tmm dataset that exercises the formal contract."""

import argparse
from pathlib import Path

import numpy as np

from double_sided.config import DoubleSidedConfig
from double_sided.data import sample_random_structure, sample_record, write_dataset
from optogpt.core.datasets.sim import load_materials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--max-layers", type=int, default=4)
    args = parser.parse_args()
    config = DoubleSidedConfig(technical_max_layers_per_side=max(32, args.max_layers)).validate()
    nk_dict = load_materials(
        all_mats=[config.substrate, *config.allowed_materials],
        wavelengths=config.wavelengths_nm / 1000.0,
        DATABASE=str(Path(__file__).resolve().parents[2] / "optogpt" / "nk"),
    )
    rng = np.random.RandomState(args.seed)
    records, spectra = [], []
    for index in range(args.samples):
        family = "alternating" if index % 2 else "random"
        structure = sample_random_structure(
            rng, config.allowed_materials, (1, args.max_layers),
            (config.min_thickness_nm, config.max_thickness_nm), family, nk_dict,
            config.token_thickness_step_nm,
        )
        record, values = sample_record(structure, family, nk_dict, config)
        records.append(record)
        spectra.append(values)
    print(write_dataset(records, spectra, args.output, config, args.seed))


if __name__ == "__main__":
    main()
