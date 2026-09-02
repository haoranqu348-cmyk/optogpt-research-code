# OptoGPT Research Code

This repository is a source-only, reproducible code package for multilayer thin-film inverse design. It deliberately excludes datasets, trained weights, checkpoints, generated figures, prediction outputs, and virtual environments.

## Contents

- `optogpt/`: baseline OptoGPT and OL-Transformer implementation, including optical-constant tables.
- `joint_sp/`: joint s/p-polarization data preparation, fine-tuning, validation, and constrained decoding.
- `double_sided/`: double-sided coating formulation, training, search, robustness evaluation, and Windows runners.
- `active_learning/`: canonical data records, manifests, and deduplication utilities.
- `self_improving/`: self-improving data-augmentation workflow.
- `tools/predictor/`: batch and interactive prediction utilities.
- `tools/figure_generation/`: scripts used to generate paper figures.
- `tools/manuscript_generation/`: scripts used to assemble manuscript drafts.
- `tools/formal_training_patch/`: Windows-oriented training patch scripts.

## Environment

Create the Windows-oriented Conda environment:

```bash
conda env create -f environment_windows.yml
conda activate optogpt
```

The configuration specifies Python 3.10, PyTorch 2.0.1 with CUDA 11.7, NumPy, SciPy, Pandas, Matplotlib, scikit-learn, TMM, and related analysis dependencies.

## Data and Checkpoints

The repository intentionally does not contain training data or model checkpoints. Place locally generated or externally supplied data and weights in the paths expected by each runner. Those artifacts are ignored by Git to prevent accidental publication or oversized commits.

## Validation

Run the lightweight unit tests from the repository root:

```bash
python -m unittest discover -s active_learning/tests
python -m unittest discover -s joint_sp/tests
python -m unittest discover -s double_sided/tests
```

Some end-to-end workflows additionally require local datasets and checkpoints.

## Research Background

The package consolidates local work built around OptoGPT, OL-Transformer, self-improving data augmentation, joint s/p-polarization design, and double-sided optical coating design. Preserve the citation and attribution requirements of the original research materials when reusing this code.
