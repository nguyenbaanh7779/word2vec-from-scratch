import json
import platform
import random
import subprocess
import time

import numpy as np

from dataset.training import get_training_data
from utils import build_noise_distribution
from model.word2vec import MyWord2Vec
from config import OUTPUT_DIR, PROJECT_ROOT

import matplotlib
matplotlib.use("Agg")  # headless backend for servers
import matplotlib.pyplot as plt
  

def get_git_state() -> dict:
    "Commit of the code used for a run, and whether tracked files had uncommitted changes"
    try:
        def git(*args): return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def train(run_name: str, embed_dim: int = 100, epochs: int = 1, max_window: int = 5, n_negatives: int = 5, max_tokens: int | None = 1_000_000,
          min_count: int = 5, subsample_t: float = 1e-3, start_lr: float = 0.025, lr_min: float = 1e-4,
          report_every: int = 500_000, seed: int = 42):
    print("Training the model...")

    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # fix both random generators: `random` drives subsampling, NumPy drives weight init, window sizes and negative samples
    random.seed(seed)
    np.random.seed(seed)

    # load training data and build the vocabulary
    tokens, voc = get_training_data(max_tokens=max_tokens, min_count=min_count, subsample_t=subsample_t)

    # for the sampling of negative examples we build a probability distribution over the vocabulary, based on word frequencies:
    freqs = list(voc.freqs.values()) 
    noise_dist = build_noise_distribution(freqs)

    # init model
    model = MyWord2Vec(
        vocab = voc,
        embed_dim = embed_dim,
        k = n_negatives,
        start_lr = start_lr
    )

    # train the model
    t0 = time.time()
    loss_history, epoch_losses = model.train(
        train_tokens = tokens,
        noise_dist = noise_dist,
        max_window = max_window,
        epochs = epochs,
        lr_min = lr_min,
        report_every = report_every
    )
    train_time = time.time() - t0

    # save the resulting embeddings
    output_file = run_dir / f"embeddings_{run_name}.npz"
    model.save_embeddings(output_file)

    # persist loss curve data and plot
    if loss_history:
        steps, losses = zip(*loss_history)
        steps_arr = np.array(steps)
        losses_arr = np.array(losses)

        loss_data_path = run_dir / f"loss_{run_name}.npz"
        np.savez(loss_data_path, steps=steps_arr, losses=losses_arr, epoch_losses=np.array(epoch_losses))

        plt.figure(figsize=(8, 4))
        plt.plot(steps_arr, losses_arr, marker=".", label=f"Avg loss over the last {report_every:,} steps")
        plt.xlabel("Global step")
        plt.ylabel("Average loss")
        plt.title(f"Training loss ({run_name})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        loss_plot_path = run_dir / f"loss_{run_name}.png"
        plt.savefig(loss_plot_path)
        plt.close()

        print(f"Saved loss data to {loss_data_path}")
        print(f"Saved loss plot to {loss_plot_path}")

    # record everything needed to reproduce and report this run
    config = {
        "run_name": run_name,
        "hyperparameters": {
            "max_tokens": max_tokens,
            "min_count": min_count,
            "subsample_t": subsample_t,
            "embed_dim": embed_dim,
            "epochs": epochs,
            "max_window": max_window,
            "n_negatives": n_negatives,
            "noise_power": 0.75,
            "start_lr": start_lr,
            "lr_min": lr_min,
            "report_every": report_every,
            "seed": seed,
        },
        "data": {
            "vocab_size": len(voc),
            "tokens_after_subsampling": len(tokens),
        },
        "results": {
            "train_time_s": round(train_time, 2),
            "epoch_losses": [float(loss) for loss in epoch_losses],
            "loss_history": [[int(step), float(loss)] for step, loss in loss_history],
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "git": get_git_state(),
        },
    }
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    print(f"Saved run config to {config_path}")
