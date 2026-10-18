# Word2Vec Training From Scratch

A study fork of [NiccoloCase/word2vec-from-scratch](https://github.com/NiccoloCase/word2vec-from-scratch) by Niccolò Caselli (University of Amsterdam, 2026). The Skip-gram with Negative Sampling (SGNS) implementation is the upstream author's; this fork makes the experiments reproducible and evaluates the learned embeddings. See [What changed in this fork](#what-changed-in-this-fork) and [strategy.md](strategy.md) for the project plan.

## Goal

Minimal implementation of the Skip-gram Word2Vec model with negative sampling. The project trains word embeddings from scratch on the text8 corpus, handles data prep, training, and basic evaluation utilities with minimal dependencies (no PyTorch/TensorFlow, only NumPy).

## Dataset

- Uses the text8 corpus (Wikipedia text). It is downloaded automatically to data/text8 on first run and optionally truncated via the `--max-tokens` flag for quicker experiments.

## Project layout

```
word2vec-from-scratch/
├─ run.py                   # Typer CLI entry point
├─ environment.yml          # Conda environment spec
├─ strategy.md              # Project plan for the report and presentation
├─ data/                    # text8 download cache (not tracked)
├─ output/                  # Saved runs: embeddings, loss curves, config.json (not tracked)
├─ upstream/                # Upstream author's submission notes, gradient derivation (implementation.pdf), Slurm jobs and logs
├─ experiments/            # run_{smoke_test,demo,final}.sh, make_report_assets.py (figures/tables for report and slides)
├─ report/                 # LaTeX report (main.tex, main.pdf); generated/ and figures/ come from make_report_assets.py
├─ slides/                 # Beamer slides (main.tex, main.pdf) and timed speaker notes
├─ demo/demo.py            # Live demo: neighbours, cosine percentile, analogies on a trained run
├─ src/
│  ├─ config.py             # Paths and directory bootstrap
│  ├─ utils.py              # Tokeniser, noise dist, negative sampler, sigmoid, LR decay
│  ├─ dataset/
│  │  ├─ text8.py           # Download/load text8
│  │  ├─ vocabulary.py      # Vocab build + subsampling
│  │  └─ training.py        # Data prep pipeline
│  ├─ model/
│  │  └─ word2vec.py        # SGNS model, training loop, eval utils
│  ├─ training/
│  │  └─ train.py           # Experiment runner and logging
│  └─ evaluation/
│     └─ evaluate.py        # Fixed queries, untrained baseline, cosine percentiles, PCA
└─ eval_embeddings.ipynb    # Inspect embeddings/analogies/loss
```

## Setup

```bash
conda env create -f environment.yml
conda activate word2vec-from-scratch
```

## How to train

From the repo root:

```bash
python run.py train --run-name demo --embed-dim 100 --epochs 3 --max-window 5 --n-negatives 5 \
    --max-tokens 1000000 --report-every 100000 --seed 42
```

`python run.py train --help` lists every option, including `--min-count`, `--subsample-t`, `--start-lr` and `--lr-min`. Runs with the same arguments produce identical embeddings.

Outputs land in `output/<run_name>/`:

- `embeddings_<run_name>.npz`: `W_in`, `W_out` and the vocabulary
- `loss_<run_name>.npz` / `.png`: mean loss over each `--report-every` window, plus the mean loss of each epoch
- `config.json`: all hyperparameters, vocabulary size, tokens after subsampling, runtime, loss history, Python/NumPy versions and the git commit

The number of training steps is roughly `epochs × tokens after subsampling × (max_window + 1)`; subsampling keeps about 65% of the tokens. Training runs at about 45k steps/s on a single core of an Apple M5, so 1M tokens × 3 epochs takes a few minutes.

## Evaluation, report and slides

```bash
python run.py evaluate --run-name final          # writes evaluation.json/.md and a PCA figure to output/final/
python experiments/make_report_assets.py         # figures, tables and number macros for report/ and slides/ (needs demo and final runs)
cd report && xelatex main && bibtex main && xelatex main && xelatex main
cd slides && xelatex main && xelatex main
python demo/demo.py                              # interactive: WORD | WORD WORD | WORD - WORD + WORD
```

## Notebook usage

- Open `eval_embeddings.ipynb` after a training run. Point it to a saved `embeddings_<run_name>.npz` to explore nearest neighbours and analogies.

## Design choices (upstream)

- Skip-gram with negative sampling to keep training efficient on large vocabularies
- Unigram^{3/4} noise distribution to balance frequent/rare negatives
- Mikolov subsampling to down-weight extremely frequent tokens
- Linear learning-rate decay.
- Separate input/output embedding matrices (W_in, W_out) for standard SGNS updates; but only W_in used for similarity queries
- Uniform init for W_in and zero init for W_out

## What changed in this fork

- **CLI**: `train` is an explicit subcommand (the upstream command `python run.py train ...` failed because Typer treats a single command as the root command). The previously hard-coded `min_count`, `subsample_t`, `start_lr`, `lr_min` and `report_every` are now options, and `--seed` makes runs reproducible.
- **Run record**: each run writes a `config.json`.
- **Loss logging**: each recorded point is the mean loss since the previous report; upstream recorded a running average since the start of the epoch.
- **Negative sampling**: negatives are drawn in batches from the same unigram^{3/4} distribution instead of one `np.random.choice(p=...)` call per training pair, which rebuilt the distribution's CDF every time. Training is 4–9× faster depending on vocabulary size.
- **Analogy scores** are proper cosine similarities (the target vector is normalised); upstream scores could exceed 1.
- **Environment**: conda-forge only, plus Jupyter.
- **Repository hygiene**: generated outputs are no longer tracked, and the upstream author's own submission material is in `upstream/`. The upstream author's 5M-token, 1-epoch embeddings remain available in git history at commit `ff07bfb` (`output/train/`).
