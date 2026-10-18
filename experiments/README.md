# Experiments

All runs use the conda environment from `environment.yml` and are run from any directory:

| Script | Run | Corpus | Purpose |
|---|---|---|---|
| `run_smoke_test.sh` | `smoke_test` | 100K tokens × 1 epoch, dim 50 | Check that the pipeline runs; not reported |
| `run_demo.sh` | `demo` | 1M tokens × 3 epochs, dim 100 | Smaller-corpus comparison |
| `run_final.sh` | `final` | 5M tokens × 3 epochs, dim 100 | Main results of the report and slides |

The demo and final runs differ only in the number of tokens; window 5, 5 negatives and seed 42 are shared.

Each script trains and then evaluates. A run writes to `output/<run>/`:

- `embeddings_<run>.npz`, `loss_<run>.npz`, `loss_<run>.png`, `config.json` (from `train`)
- `evaluation.json`, `evaluation.md`, `pca_<run>.png` (from `evaluate`)

After the demo and final runs, `python experiments/make_report_assets.py` regenerates the figures and tables used by `report/` and `slides/` (`report/figures/`, `report/generated/`), so every number in them comes from these files.
