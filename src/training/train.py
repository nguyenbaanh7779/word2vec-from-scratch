import random

import numpy as np

from dataset.training import get_training_data
from utils import build_noise_distribution
from model.word2vec import MyWord2Vec
from config import OUTPUT_DIR

import matplotlib
matplotlib.use("Agg")  # headless backend for servers
import matplotlib.pyplot as plt
  

def train(run_name: str, embed_dim: int = 100, epochs: int = 1, max_window: int = 5, n_negatives: int = 5, max_tokens: int | None = 1_000_000,
          min_count: int = 5, subsample_t: float = 1e-3, start_lr: float = 0.025, lr_min: float = 1e-4, seed: int = 42):
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
    loss_history = model.train(
        train_tokens = tokens,
        noise_dist = noise_dist,
        max_window = max_window,
        epochs = epochs,
        lr_min = lr_min
    )

    # save the resulting embeddings
    output_file = run_dir / f"embeddings_{run_name}.npz"
    model.save_embeddings(output_file)

    # persist loss curve data and plot
    if loss_history:
        steps, losses = zip(*loss_history)
        steps_arr = np.array(steps)
        losses_arr = np.array(losses)

        loss_data_path = run_dir / f"loss_{run_name}.npz"
        np.savez(loss_data_path, steps=steps_arr, losses=losses_arr)

        plt.figure(figsize=(8, 4))
        plt.plot(steps_arr, losses_arr, label="Avg loss")
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
