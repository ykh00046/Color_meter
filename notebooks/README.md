# Notebooks Directory

This directory contains Jupyter notebooks for prototyping and experimentation.

## Structure

```
notebooks/
├── 01_prototype.ipynb               # Stage 1 prototype
├── 02_lens_detection.ipynb          # Lens detection experiments
├── 03_radial_profile.ipynb          # R-profile analysis
├── 04_zone_segmentation.ipynb       # Zone segmentation tests
├── 05_color_evaluation.ipynb        # Color evaluation experiments
└── experiments/
    ├── lighting_robustness.ipynb
    └── multi_sku_test.ipynb
```

## Usage

Start Jupyter:
```bash
jupyter notebook
```

Or use JupyterLab:
```bash
jupyter lab
```

## Guidelines

1. **Naming**: Use numbered prefixes for sequential notebooks
2. **Documentation**: Add markdown cells to explain each step
3. **Cleanup**: Clear output before committing (to reduce file size)
4. **Results**: Save important findings in `docs/`

## Recommended Workflow

1. Experiment in notebook
2. Validate results with multiple samples
3. Refactor code into `src/` modules
4. Add unit tests in `tests/`
5. Update documentation

## Git Tracking

Notebook checkpoints (`.ipynb_checkpoints/`) are ignored by Git.
Commit notebooks with outputs cleared:

```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebook.ipynb
```
