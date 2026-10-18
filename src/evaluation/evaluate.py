import json

import numpy as np

from model.word2vec import MyWord2Vec
from config import OUTPUT_DIR

import matplotlib
matplotlib.use("Agg")  # headless backend for servers
import matplotlib.pyplot as plt


# All queries are fixed here before looking at any trained model, and every result is reported,
# including failures: swapping a failed query for one that happens to work would be cherry-picking.

NEIGHBOUR_QUERIES = ["king", "computer", "city", "france", "january", "three", "war", "music"]

# (word_a, word_b): related pairs first, then unrelated control pairs
SIMILARITY_PAIRS = [
    ("king", "queen"),
    ("man", "woman"),
    ("computer", "software"),
    ("city", "country"),
    ("france", "germany"),
    ("january", "february"),
    ("king", "software"),
    ("computer", "january"),
    ("music", "france"),
]

# "a is to b as c is to expected", i.e. expected ≈ b - a + c
ANALOGIES = [
    ("man", "king", "woman", "queen"),
    ("france", "paris", "italy", "rome"),
    ("france", "paris", "germany", "berlin"),
    ("man", "woman", "boy", "girl"),
    ("brother", "sister", "father", "mother"),
    ("good", "better", "bad", "worse"),
    ("big", "bigger", "small", "smaller"),
    ("england", "english", "france", "french"),
]

PCA_GROUPS = {
    "countries": ["france", "germany", "italy", "spain", "england", "russia", "china", "japan"],
    "numbers": ["one", "two", "three", "four", "five", "six", "seven", "eight"],
    "months": ["january", "february", "march", "april", "june", "july", "august", "october"],
    "family/royalty": ["king", "queen", "prince", "father", "mother", "son", "daughter", "brother"],
}

N_RANDOM_PAIRS = 10_000

# The random-pair baseline is drawn from the most frequent words only. Rare words receive few updates and
# drift along a shared direction, so pairs of rare words are very similar to each other (0.93 on average in
# the 1M-token demo, against 0.61 among the 2,000 most frequent words). Since rare words make up most of the
# vocabulary, an all-vocabulary baseline is inflated and does not match the frequent words used in the queries.
N_BASELINE_WORDS = 2_000


def normalise(W: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    return W / np.where(norms == 0, 1.0, norms)


def random_pair_cosines(model: MyWord2Vec, rng: np.random.Generator, n_words: int | None = None) -> np.ndarray:
    """
    Cosine similarity of random pairs of distinct words among the n_words most frequent ones (all words if None).
    Raw cosine values are hard to read on their own (SGNS embeddings tend to share a common direction, so
    even unrelated words can score high); this distribution is the reference a word pair is compared against.
    """
    W = normalise(model.W_in)
    n = len(W) if n_words is None else min(n_words, len(W))  # the vocabulary is sorted by decreasing frequency
    i = rng.integers(0, n, N_RANDOM_PAIRS)
    j = rng.integers(0, n, N_RANDOM_PAIRS)
    keep = i != j
    return np.sum(W[i[keep]] * W[j[keep]], axis=1)


def evaluate_model(model: MyWord2Vec, seed: int) -> dict:
    vocab = model.vocab.word2idx
    W = normalise(model.W_in)
    random_cos = random_pair_cosines(model, np.random.default_rng(seed), N_BASELINE_WORDS)
    random_cos_all = random_pair_cosines(model, np.random.default_rng(seed))

    neighbours = {
        q: [(w, round(s, 3)) for w, s in model.get_most_similar(q, top_k=8)] if q in vocab else None
        for q in NEIGHBOUR_QUERIES
    }

    pairs = []
    for a, b in SIMILARITY_PAIRS:
        if a not in vocab or b not in vocab:
            pairs.append({"pair": [a, b], "cosine": None, "percentile": None})
            continue
        cos = float(W[vocab[a]] @ W[vocab[b]])
        # share of random word pairs that are less similar than this pair
        pairs.append({"pair": [a, b], "cosine": round(cos, 3), "percentile": round(100 * float(np.mean(random_cos < cos)), 1)})

    analogies = []
    for a, b, c, expected in ANALOGIES:
        if any(w not in vocab for w in (a, b, c, expected)):
            analogies.append({"query": [a, b, c], "expected": expected, "top5": None, "rank": None})
            continue
        top10 = [w for w, _ in model.analogy(a, b, c, top_k=10)]
        rank = top10.index(expected) + 1 if expected in top10 else None
        analogies.append({"query": [a, b, c], "expected": expected, "top5": top10[:5], "rank": rank})

    answerable = [x for x in analogies if x["top5"] is not None]
    return {
        "random_pair_cosine": {
            f"top_{N_BASELINE_WORDS}": {"mean": round(float(random_cos.mean()), 3), "std": round(float(random_cos.std()), 3)},
            "all_words": {"mean": round(float(random_cos_all.mean()), 3), "std": round(float(random_cos_all.std()), 3)},
        },
        "neighbours": neighbours,
        "pairs": pairs,
        "analogies": analogies,
        "analogy_summary": {
            "answerable": len(answerable),
            "hit@1": sum(x["rank"] == 1 for x in answerable),
            "hit@5": sum(x["rank"] is not None and x["rank"] <= 5 for x in answerable),
        },
    }


# Figure styling: validated categorical slots 1-3 plus a neutral ink for the 4th group (four hues cannot all be
# told apart in a scatter), and a distinct marker per group so identity never relies on colour alone
GROUP_STYLES = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^"), ("#52514e", "D")]
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style_axes(ax) -> None:
    "Recessive axes: no top/right spines, light grid behind the data, secondary ink for ticks and labels"
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK_MUTED)
    ax.yaxis.label.set_color(INK_MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_pca(models: dict[str, MyWord2Vec], out_path, panel_size: tuple[float, float] = (5.2, 4.4)) -> None:
    """
    2D PCA of the PCA_GROUPS words, one panel per model.
    Vectors are L2-normalised (as for cosine similarity) and centred on the plotted words before the SVD.
    """
    fig, axes = plt.subplots(1, len(models), figsize=(panel_size[0] * len(models), panel_size[1]))
    panels = []
    for ax, (title, model) in zip(np.atleast_1d(axes), models.items()):
        vocab = model.vocab.word2idx
        words, groups = [], []
        for group, group_words in PCA_GROUPS.items():
            for w in group_words:
                if w in vocab:
                    words.append(w)
                    groups.append(group)

        X = normalise(model.W_in[[vocab[w] for w in words]])
        X = X - X.mean(axis=0)
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        coords = X @ Vt[:2].T
        explained = S[:2] ** 2 / np.sum(S ** 2)

        for group, (color, marker) in zip(PCA_GROUPS, GROUP_STYLES):
            idx = [i for i, g in enumerate(groups) if g == group]
            ax.scatter(coords[idx, 0], coords[idx, 1], label=group, s=40, color=color, marker=marker,
                       edgecolors="white", linewidths=1.0, zorder=3)
        style_axes(ax)
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_xlabel(f"PC1 ({explained[0]:.0%} of variance)", fontsize=9)
        ax.set_ylabel(f"PC2 ({explained[1]:.0%} of variance)", fontsize=9)
        ax.legend(fontsize=7.5, frameon=False, labelcolor=INK)
        panels.append((ax, coords, words))

    fig.tight_layout()

    # label as many points as possible without overlapping text or the legend: in tight clusters only some
    # words are named, the group is still given by colour, marker and legend
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, coords, words in panels:
        placed = [ax.get_legend().get_window_extent(renderer)]
        for (x, y), w in zip(coords, words):
            label = ax.annotate(w, (x, y), fontsize=7.5, color=INK, xytext=(3, 3), textcoords="offset points")
            box = label.get_window_extent(renderer).expanded(1.05, 1.1)
            if any(box.overlaps(other) for other in placed):
                label.remove()
            else:
                placed.append(box)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def format_markdown(run_name: str, trained: dict, untrained: dict) -> str:
    def fmt_cos(x): return "OOV" if x["cosine"] is None else f"{x['cosine']:.3f} (p{x['percentile']:.0f})"
    def fmt_rank(x): return "OOV" if x["top5"] is None else (str(x["rank"]) if x["rank"] else ">10")

    lines = [f"# Evaluation: {run_name}", "",
             "Untrained = the same model at initialisation (same seed), before any training step.", ""]

    lines += ["## Nearest neighbours (cosine, W_in)", "", "| Query | Trained (top 8) | Untrained (top 3) |", "|---|---|---|"]
    for q in NEIGHBOUR_QUERIES:
        t, u = trained["neighbours"][q], untrained["neighbours"][q]
        lines.append(f"| {q} | {'OOV' if t is None else ', '.join(w for w, _ in t)} | {'OOV' if u is None else ', '.join(w for w, _ in u[:3])} |")

    def fmt_dist(d): return f"{d['mean']:.3f} ± {d['std']:.3f}"
    top = f"top_{N_BASELINE_WORDS}"
    rt, ru = trained["random_pair_cosine"], untrained["random_pair_cosine"]
    lines += ["", "## Cosine similarity", "",
              f"Random pairs among the {N_BASELINE_WORDS:,} most frequent words: trained {fmt_dist(rt[top])}, untrained {fmt_dist(ru[top])}. "
              f"pN: the pair is more similar than N% of these random pairs. "
              f"(Random pairs over the whole vocabulary: trained {fmt_dist(rt['all_words'])}, inflated by rare words.)", "",
              "| Pair | Trained | Untrained |", "|---|---|---|"]
    for t, u in zip(trained["pairs"], untrained["pairs"]):
        lines.append(f"| {t['pair'][0]} – {t['pair'][1]} | {fmt_cos(t)} | {fmt_cos(u)} |")

    st, su = trained["analogy_summary"], untrained["analogy_summary"]
    lines += ["", "## Analogies (b − a + c)", "",
              f"Hit@1: trained {st['hit@1']}/{st['answerable']}, untrained {su['hit@1']}/{su['answerable']}. "
              f"Hit@5: trained {st['hit@5']}/{st['answerable']}, untrained {su['hit@5']}/{su['answerable']}.", "",
              "| a : b :: c : ? | Expected | Trained top 5 | Rank (trained) | Rank (untrained) |", "|---|---|---|---|---|"]
    for t, u in zip(trained["analogies"], untrained["analogies"]):
        a, b, c = t["query"]
        top5 = "OOV" if t["top5"] is None else ", ".join(t["top5"])
        lines.append(f"| {a} : {b} :: {c} : ? | {t['expected']} | {top5} | {fmt_rank(t)} | {fmt_rank(u)} |")

    return "\n".join(lines) + "\n"


def load_run_models(run_name: str) -> tuple[MyWord2Vec, MyWord2Vec, int]:
    "The trained model of a run, the same model before training, and the run's seed"
    run_dir = OUTPUT_DIR / run_name
    model = MyWord2Vec.load(str(run_dir / f"embeddings_{run_name}.npz"))
    seed = json.loads((run_dir / "config.json").read_text())["hyperparameters"]["seed"]

    # rebuild the model as it was before the first training step: training seeds NumPy and then
    # initialises W_in with no other NumPy draw in between (subsampling uses the `random` module)
    np.random.seed(seed)
    untrained = MyWord2Vec(vocab=model.vocab, embed_dim=model.embed_dim, k=model.k, start_lr=model.start_lr)
    return model, untrained, seed


def evaluate(run_name: str):
    run_dir = OUTPUT_DIR / run_name
    model, untrained, seed = load_run_models(run_name)

    results = {"trained": evaluate_model(model, seed), "untrained": evaluate_model(untrained, seed)}
    (run_dir / "evaluation.json").write_text(json.dumps(results, indent=2))

    report = format_markdown(run_name, results["trained"], results["untrained"])
    (run_dir / "evaluation.md").write_text(report)

    plot_pca({f"Trained ({run_name})": model, "Untrained (initialisation)": untrained}, run_dir / f"pca_{run_name}.png")

    print(report)
    print(f"Saved evaluation to {run_dir / 'evaluation.json'}, {run_dir / 'evaluation.md'} and {run_dir / f'pca_{run_name}.png'}")
