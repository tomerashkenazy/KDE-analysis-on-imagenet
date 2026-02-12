# AGENTS

This file tells agents how to work in this repo.

## Project summary
- Research code for analyzing pairwise distances on the ImageNet validation set.
- Heavy computations (50k x 50k matrices) and large outputs.

## Environment
- Python 3.8+.
- GPU recommended for distance computation.
- External data required: ImageNet validation set and optionally the ImageNet devkit (for superclass matrix).

## Setup
- Install dependencies:
  - `pip install -r requirements.txt`
- Configure paths and parameters in `config.yaml` (Hydra config).

## Common tasks
- Build distance matrices: `python val_dist_mat.py`
- PCA: `python pca_distance_matrix.py --norms l1 l2 linf --variance 0.95 --batch-size 1000`
- Nearest-neighbor eval: `python nearest_neighbor.py`
- Class stats: `python imagenet_class_stats.py`
- Distribution analysis: `python imagenet_hist.py`
- Cluster separation: `python cluster_seperation.py`
- Build superclass matrix: `python build_sup_mat.py`

## Data and outputs
- Large outputs go to `results_imagenet_stats/` and Hydra outputs in `outputs/`.
- Logs are written under `logs/`.
- Some scripts expect ImageNet labels in `imagenet_labels.txt`.

## Tests
- No formal test suite is defined. If you add one, document it here.

## Conventions
- Prefer adding scripts rather than refactoring existing ones unless asked.
- Be careful with memory usage and intermediate files; many artifacts are large.
- Use ASCII text and concise comments.
