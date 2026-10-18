"""
Live demo: query the trained embeddings of a run.

    python demo/demo.py               # interactive
    python demo/demo.py --script      # the fixed sequence shown on the demo slide

Interactive input:
    king                    nearest neighbours
    king queen              cosine similarity, with its percentile among random pairs of frequent words
    king - man + woman      analogy (nearest words to king - man + woman)
    quit
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

import numpy as np

from config import OUTPUT_DIR
from model.word2vec import MyWord2Vec
from evaluation.evaluate import N_BASELINE_WORDS, normalise, random_pair_cosines

# Every query comes from the fixed lists in src/evaluation/evaluate.py; the failed analogy is shown on purpose
DEMO_SCRIPT = [
    "january",
    "france",
    "king",
    "king queen",
    "king software",
    "king - man + woman",
    "bigger - big + small",
]


class Demo:
    def __init__(self, run_name: str):
        run_dir = OUTPUT_DIR / run_name
        self.model = MyWord2Vec.load(str(run_dir / f"embeddings_{run_name}.npz"))
        self.W = normalise(self.model.W_in)
        # same random pairs as `run.py evaluate`, so the percentiles match the report
        seed = json.loads((run_dir / "config.json").read_text())["hyperparameters"]["seed"]
        self.baseline = np.sort(random_pair_cosines(self.model, np.random.default_rng(seed), N_BASELINE_WORDS))

    def check(self, *words: str) -> bool:
        missing = [w for w in words if w not in self.model.vocab]
        if missing:
            print(f"  not in vocabulary: {', '.join(missing)}")
        return not missing

    def neighbours(self, word: str) -> None:
        if self.check(word):
            for w, s in self.model.get_most_similar(word, top_k=8):
                print(f"  {w:<14} {s:.3f}")

    def cosine(self, a: str, b: str) -> None:
        if self.check(a, b):
            idx = self.model.vocab.word2idx
            cos = float(self.W[idx[a]] @ self.W[idx[b]])
            pct = 100 * np.searchsorted(self.baseline, cos) / len(self.baseline)
            print(f"  cos({a}, {b}) = {cos:.3f}: more similar than {pct:.1f}% of random pairs of frequent words")

    def analogy(self, positive: str, negative: str, other: str) -> None:
        # positive - negative + other, i.e. "negative is to positive as other is to ?"
        if self.check(positive, negative, other):
            for w, s in self.model.analogy(negative, positive, other, top_k=5):
                print(f"  {w:<14} {s:.3f}")

    def run(self, line: str) -> None:
        tokens = line.lower().split()
        if len(tokens) == 5 and tokens[1] == "-" and tokens[3] == "+":
            self.analogy(tokens[0], tokens[2], tokens[4])
        elif len(tokens) == 2:
            self.cosine(*tokens)
        elif len(tokens) == 1:
            self.neighbours(tokens[0])
        else:
            print("  use: WORD | WORD WORD | WORD - WORD + WORD")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", default="final", help="Training run in output/ whose embeddings are queried")
    parser.add_argument("--script", action="store_true", help="Run the fixed demo sequence and exit")
    args = parser.parse_args()

    demo = Demo(args.run_name)
    print(f"Loaded '{args.run_name}': {len(demo.model.vocab):,} words, {demo.model.embed_dim} dimensions")

    if args.script:
        for line in DEMO_SCRIPT:
            print(f"\n> {line}")
            demo.run(line)
        return

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line in ("quit", "exit"):
            break
        if line:
            demo.run(line)


if __name__ == "__main__":
    main()
