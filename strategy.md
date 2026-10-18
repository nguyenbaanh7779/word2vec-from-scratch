# Word2Vec From Scratch — Project Strategy

> **Project goal:** Fork `NiccoloCase/word2vec-from-scratch`, customize
> it into a small, reproducible Word2Vec study project, then produce a
> concise technical report and a ~10-minute presentation using
> LaTeX/Beamer.

> **Revision notes (2026-10-18), after reading the code and running it:**
>
> - The CLI command `python run.py train ...` failed upstream (Typer treated the single command as the root command). Fixed; `train` and `evaluate` are now subcommands (§4, §22).
> - Compute budget measured and added (§4.0). Negative sampling was the bottleneck; with batched sampling training runs at ~45k steps/s on an Apple M5, so the 5M-token final experiment takes about 25 minutes instead of ~8 hours.
> - Citations corrected: negative sampling comes from Mikolov et al. 2013b (arXiv:1310.4546), not 1301.3781 (§11).
> - Concept → code table corrected: training pairs and SGD updates live in `src/model/word2vec.py` (§9).
> - Reproducibility: `--seed`, previously hidden hyperparameters exposed, and a `config.json` per run (§4, §15).
> - Loss is now logged as a windowed average; upstream logged a running average since the start of each epoch (§6.1).
> - Evaluation: queries fixed in advance, an untrained baseline, cosine reported as a percentile of random pairs, and analogy scores normalised (§5, §6, §14).
> - Upstream material moved to `upstream/`; generated outputs untracked (§16).

## 0. Project decision

### Chosen base repository

-   **Repository:** `NiccoloCase/word2vec-from-scratch`
-   **Official repo:**
    https://github.com/NiccoloCase/word2vec-from-scratch
-   **Model:** Skip-gram with Negative Sampling (SGNS)
-   **Framework:** NumPy; no PyTorch/TensorFlow
-   **Dataset:** text8, downloaded automatically on first run
-   **Primary evaluation:** nearest neighbours, cosine similarity of
    selected word pairs (compared against random pairs; no word-similarity
    benchmark), word analogies, training loss
-   **Presentation target:** ~10 minutes

The upstream repository is intentionally minimal and already contains
the core implementation, text8 loading, training CLI, saved
embeddings/loss outputs, and an evaluation notebook. The upstream README
describes the implementation as Skip-gram Word2Vec with negative
sampling and lists unigram^3/4 negative sampling, frequent-token
subsampling, linear learning-rate decay, and separate input/output
embedding matrices.

The upstream author also left their own material: a 5M-token, 1-epoch
training run (embeddings, loss, Slurm log), a submission note and a
gradient-derivation PDF. These are kept in `upstream/` (and the
embeddings in git history at commit `ff07bfb`) and must be labelled as
upstream wherever they are used.

------------------------------------------------------------------------

# 1. What we are trying to demonstrate

The project should answer one central question:

> **How can a model learn useful numerical representations of words from
> their surrounding context?**

The story of the project should be:

``` text
Raw text
   ↓
Vocabulary + preprocessing
   ↓
Training pairs
   ↓
Skip-gram
   ↓
Negative sampling
   ↓
Gradient-based optimization
   ↓
Word embeddings
   ↓
Similarity / nearest neighbours / analogy
   ↓
Interpretation
```

The report should not become a generic survey of NLP. Keep the scope
focused on:

1.  What Word2Vec is.
2.  Why word embeddings are useful.
3.  How Skip-gram works.
4.  Why negative sampling is used.
5.  How this repository implements the algorithm.
6.  What happens when we train it.
7.  What the learned embeddings can demonstrate.
8.  Limitations and possible applications.

------------------------------------------------------------------------

# 2. Important customization principle

Do **not** simply fork the repository and submit the original
code/results.

The fork should become an educational experiment built on top of the
original implementation.

## Keep from upstream

-   Core Skip-gram implementation.
-   Negative sampling.
-   text8 loader.
-   Vocabulary construction.
-   Existing training pipeline.
-   Existing evaluation logic where useful.

## Customize

At minimum:

-   Add a project-specific configuration.
-   Add a reproducible experiment script/config.
-   Add a small results table.
-   Add loss visualization if necessary.
-   Add nearest-neighbour evaluation.
-   Add analogy evaluation.
-   Add 2D embedding visualization (PCA preferred for a simple report;
    t-SNE can be optional).
-   Add README/project documentation.
-   Add LaTeX report.
-   Add Beamer slides.
-   Add a short demo notebook.
-   Record exact experiment parameters.
-   Separate exploratory runs from the final experiment.

## Do not over-customize

Avoid:

-   Rewriting the entire algorithm unnecessarily.
-   Adding PyTorch just for the sake of using a framework.
-   Building a web application.
-   Training the entire text8 corpus unless needed.
-   Implementing CBOW unless it is explicitly required.
-   Comparing many Word2Vec libraries.
-   Turning this into a production NLP system.

The objective is **understanding + reproducibility + clear
presentation**, not maximum model performance.

------------------------------------------------------------------------

# 3. Recommended final repository structure

After forking, gradually move toward:

``` text
word2vec-from-scratch/
│
├── src/                         # upstream/core implementation
│   ├── config.py
│   ├── utils.py
│   ├── dataset/
│   ├── model/
│   ├── training/
│   └── evaluation/              # added: fixed queries + untrained baseline
│
├── data/                        # local data/cache; not committed
│
├── output/                      # generated experiment outputs; not committed
│   ├── smoke_test/
│   ├── demo/
│   └── final/
│
├── upstream/                    # upstream author's notes, derivation PDF, Slurm jobs/logs
│
├── experiments/
│   ├── README.md
│   ├── run_smoke_test.sh
│   ├── run_demo.sh
│   └── run_final.sh
│
├── notebooks/
│   ├── 01_explore_corpus.ipynb
│   ├── 02_train_and_evaluate.ipynb
│   └── 03_visualize_embeddings.ipynb
│
├── report/
│   ├── main.tex
│   ├── references.bib
│   ├── figures/
│   └── sections/
│       ├── 01_introduction.tex
│       ├── 02_background.tex
│       ├── 03_methodology.tex
│       ├── 04_experiment.tex
│       ├── 05_results.tex
│       ├── 06_applications.tex
│       └── 07_conclusion.tex
│
├── slides/
│   ├── main.tex
│   ├── references.bib
│   └── figures/
│
├── demo/
│   └── demo.py
│
├── README.md
├── strategy.md
└── environment.yml
```

The exact upstream structure can remain intact. The folders above are
the **target organization**, not a requirement to refactor everything
immediately.

------------------------------------------------------------------------

# 4. Experiment strategy

## 4.0 Compute budget

Training steps ≈ `epochs × tokens after subsampling × (max_window + 1)`.
Subsampling keeps about 54% of the tokens at 100K, 63% at 1M and 67% at 5M.

Measured throughput (single core, `embed_dim=100`):

| Code | Machine | Steps/s |
|---|---|---|
| Upstream, V = 36K | Snellius (upstream log) | ~2,100 |
| Upstream, V = 14K / 36K | Apple M5 | ~11K / ~5K |
| This fork (batched negatives) | Apple M5 | ~45K, independent of V |

| Run | Steps | Time on M5 (this fork) |
|---|---|---|
| Smoke 100K × 1 | ~0.3M | ~5 s |
| Demo 1M × 3 | ~11M | ~4–5 min |
| Final 5M × 3 | ~61M | ~25 min |
| Final 5M × 5 | ~101M | ~40 min |

## 4.1 Smoke test

Purpose: prove that the fork runs.

Suggested configuration:

``` text
max_tokens = 100_000
embed_dim = 50
epochs = 1
max_window = 2 or 5
n_negatives = 5
```

``` bash
python run.py train --run-name smoke_test --embed-dim 50 --epochs 1 \
    --max-window 5 --n-negatives 5 --max-tokens 100000 --report-every 20000
```

Expected output:

-   training starts successfully;
-   loss is produced;
-   embeddings and `config.json` are saved.

The vocabulary is small (~2.6K words), so words such as `computer` are
out of vocabulary. Do not use these results in the final report unless
necessary.

------------------------------------------------------------------------

## 4.2 Main demo experiment

Purpose: generate the results used in the presentation.

Suggested starting configuration:

``` text
max_tokens = 1_000_000
embed_dim = 100
epochs = 3
max_window = 5
n_negatives = 5
seed = 42
```

``` bash
python run.py train \
    --run-name demo \
    --embed-dim 100 \
    --epochs 3 \
    --max-window 5 \
    --n-negatives 5 \
    --max-tokens 1000000 \
    --report-every 100000 \
    --seed 42

python run.py evaluate --run-name demo
```

Hyperparameters that are not set on the command line still apply and
must appear in the report: `min_count = 5`, `subsample_t = 1e-3`,
`start_lr = 0.025`, `lr_min = 1e-4`, noise power `0.75`. All of them
are written to `output/<run>/config.json`.

------------------------------------------------------------------------

## 4.3 Final experiment

Only run this if the demo experiment is stable. With the measured
budget (§4.0) this is now feasible on a laptop.

Suggested starting point:

``` text
max_tokens = 5_000_000
embed_dim = 100
epochs = 3–5
max_window = 5
n_negatives = 5
```

The upstream author used 5M tokens for 1 epoch. Their run can serve as a
clearly labelled reference point, but it is not our result.

Do not claim that these parameters are universally optimal. They are
simply the chosen experimental configuration.

------------------------------------------------------------------------

# 5. Experimental questions

The report should answer a small number of concrete questions.

### Q1. Does training reduce the loss?

Evidence:

-   loss curve (windowed average, see §6.1);
-   initial vs final loss (`epoch_losses` in `config.json`).

### Q2. Does the model learn semantically related words?

Evidence:

``` text
most_similar("king")
most_similar("computer")
most_similar("city")
```

compared with the same queries on the untrained model (same seed, before
any update). Use words that actually exist in the vocabulary.

### Q3. Are similar words close in vector space?

Evidence:

``` text
cosine_similarity(word_a, word_b)
```

Compare several intuitive pairs **and unrelated control pairs**, and
report each value as a percentile of the cosine of random pairs drawn
from the 2,000 most frequent words. Raw values are inflated: neighbours
in the upstream embeddings all score 0.88–0.96, and in the 1M-token demo
random pairs average 0.84 over the whole vocabulary, because rare words
barely move from a shared direction (0.93 among words seen ≤ 19 times,
0.61 among the 2,000 most frequent). The query words are all frequent,
so the baseline must be too. This rare-word effect is itself worth a
sentence in the Limitations section.

Example:

``` text
king ↔ queen
computer ↔ software
city ↔ country
king ↔ software   (control)
```

Do not hard-code expected results before checking the trained vocabulary
and actual output.

### Q4. Can the embedding represent relationships?

Evidence:

``` text
king - man + woman ≈ queen
```

In code, `model.analogy(a, b, c)` returns the neighbours of
`b − a + c`, so this query is `analogy("man", "king", "woman")`.

This is an experiment, not an assumption. Upstream's 5M-token, 1-epoch
run does **not** solve it (top answers: pope, iii, constantine), so a
1M-token run very likely will not either. Report it honestly and explain
possible reasons (corpus size, training time).

### Q5. What does the embedding space look like?

Evidence:

-   PCA 2D plot, trained vs untrained;
-   optionally t-SNE as an additional visualization.

------------------------------------------------------------------------

# 6. Evaluation protocol

Use three complementary forms of evaluation. All queries are fixed in
`src/evaluation/evaluate.py` **before** looking at results, and every
result is reported, including failures.

## 6.1 Training behaviour

Plot:

``` text
step → mean loss over the last report_every steps
```

Interpretation:

-   decreasing loss indicates that the training objective is being
    optimized;
-   loss alone does **not** prove semantic quality;
-   upstream logged a running average since the start of each epoch,
    which lags and resets every epoch. Describe the curve as a windowed
    average.

## 6.2 Qualitative semantic evaluation

For selected words:

``` text
query word
    ↓
nearest neighbours
```

Create a table:

  Query    Top neighbours (trained)   Untrained   Interpretation
  -------- -------------------------- ----------- ----------------
  word 1   ...                        ...         ...
  word 2   ...                        ...         ...
  word 3   ...                        ...         ...

Use actual outputs from the final experiment.

## 6.3 Vector relationship evaluation

Use:

-   cosine similarity, reported against random pairs of frequent words;
-   analogy queries, reported as the rank of the expected answer
    (hit@1 / hit@5).

Example:

``` text
vector(king) - vector(man) + vector(woman)
```

Then retrieve the nearest vocabulary words.

Optional, if a number is wanted in the results table: accuracy on the
Google analogy set (`questions-words.txt`, restricted to in-vocabulary
questions) or Spearman correlation on WordSim-353.

------------------------------------------------------------------------

# 7. Visualization plan

Only create visualizations that help the story.

## Figure 1 --- Training loss

Purpose:

> Show that optimization is progressing.

## Figure 2 --- 2D embedding map

Recommended:

> PCA

Reason: PCA is deterministic, simple to explain, and sufficient for a
10-minute educational presentation.

Plot 20--50 selected words from 3–4 clear groups (countries, numbers,
months, family/royalty). L2-normalise the vectors and centre them before
the SVD, and show the untrained model next to the trained one.
`python run.py evaluate` produces this figure.

Avoid plotting thousands of words in the report.

## Optional Figure 3 --- t-SNE

Use only if it produces a clear and interpretable result.

Do not spend excessive time tuning t-SNE.

------------------------------------------------------------------------

# 8. Report strategy

Target length:

> **8--12 pages excluding references and appendices**

Recommended structure:

``` text
1. Introduction
2. Background
   2.1 Word representation
   2.2 Word2Vec
   2.3 CBOW vs Skip-gram
   2.4 Negative Sampling
3. Methodology
   3.1 Dataset
   3.2 Preprocessing
   3.3 Vocabulary and training pairs
   3.4 Skip-gram model
   3.5 Negative sampling
   3.6 Optimization
4. Experimental Setup
   4.1 Environment
   4.2 Hyperparameters
   4.3 Evaluation protocol
5. Results
   5.1 Training loss
   5.2 Similar words
   5.3 Cosine similarity
   5.4 Word analogy
   5.5 Embedding visualization
6. Applications
7. Limitations
8. Conclusion
References
Appendix
```

------------------------------------------------------------------------

# 9. Report writing rules

## Rule 1 --- Explain the intuition before equations

For example:

``` text
Words appearing in similar contexts tend to have similar
representations.
```

Then explain Skip-gram.

Only after the intuition is clear introduce:

``` text
P(context | target)
```

and the negative-sampling objective.

## Rule 2 --- Every important theoretical component should map to code

  Concept                       Code
  ----------------------------- ---------------------------------------------------------------
  Text8 download/load           `src/dataset/text8.py`
  Vocabulary (`min_count`)      `src/dataset/vocabulary.py` `Vocabulary._build`
  Subsampling                   `src/dataset/vocabulary.py` `encode_subsampled`
  Data pipeline                 `src/dataset/training.py` `get_training_data`
  Training pairs (dyn. window)  `src/model/word2vec.py` `MyWord2Vec.train`
  Noise distribution            `src/utils.py` `build_noise_distribution`, `NegativeSampler`
  SGNS forward/backward + SGD   `src/model/word2vec.py` `MyWord2Vec.step`
  Learning-rate decay           `src/utils.py` `learning_rate_decay`
  Experiment runner, saving     `src/training/train.py` (calls `save_embeddings`)
  Evaluation                    `src/evaluation/evaluate.py`, `eval_embeddings.ipynb`

`src/dataset/training.py` only loads text8, builds the vocabulary and
subsamples; it does not create training pairs. `src/training/train.py`
orchestrates a run; the gradient update itself is in `step()`.

This makes the report much more convincing.

## Rule 3 --- Clearly distinguish source material from your experiment

Use wording such as:

> The original repository implements ...

versus:

> In our experiment, we configured ...

Do not present the upstream author's design decisions or results as if
they were produced by you. This includes the upstream 5M-token run, the
submission note and `implementation.pdf`: cite the derivation, do not
copy it.

------------------------------------------------------------------------

# 10. Theory that must be covered

The minimum mathematical content is:

## 10.1 Word embeddings

Represent each word as:

``` text
w ∈ R^d
```

where `d` is the embedding dimension.

## 10.2 Skip-gram objective

Given a center word `w_t`, predict surrounding context words:

``` text
maximize:

Σ log P(w_c | w_t)
```

## 10.3 Negative sampling

Instead of calculating a full vocabulary softmax, train positive and
negative pairs.

For a positive pair:

``` text
(target, context) → 1
```

For negative samples:

``` text
(target, negative_word) → 0
```

A common objective can be written as:

``` text
log σ(v'_c · v_w)
+
Σ log σ(-v'_{n_i} · v_w)
```

where:

-   `v_w` is the input embedding (`W_in[centre]` in code);
-   `v'_c` is the output embedding of the positive context word
    (`W_out[context]`);
-   `v'_{n_i}` are negative-sample output vectors (`W_out[negatives]`);
-   `σ` is the sigmoid function.

The code minimises the negative of this objective written as
`-log σ(s_o) - Σ log(1 - σ(s_k))`, which is the same because
`1 - σ(x) = σ(-x)`.

## 10.4 Cosine similarity

Use:

``` text
cos(u,v) = (u · v) / (||u|| ||v||)
```

to compare word vectors.

Do not overload the report with derivations. The purpose is to explain
the mechanism clearly.

------------------------------------------------------------------------

# 11. Paper/reference strategy

The two central academic references are the original Word2Vec papers:

> Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013a).
> *Efficient Estimation of Word Representations in Vector Space.*
> arXiv:1301.3781.

Use it to support: motivation, the CBOW and Skip-gram architectures,
efficient training, and learned word representations.

> Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013b).
> *Distributed Representations of Words and Phrases and their
> Compositionality.* NeurIPS 2013. arXiv:1310.4546.

Use it to support: **negative sampling**, the unigram^3/4 noise
distribution and subsampling of frequent words. These are not in
1301.3781.

Additional references:

-   Mikolov, Yih & Zweig (2013), *Linguistic Regularities in Continuous
    Space Word Representations*, NAACL: the king − man + woman example
    and vector offsets.
-   Goldberg & Levy (2014), *word2vec Explained*, arXiv:1402.3722:
    derivation of the SGNS objective.
-   Harris (1954) / Firth (1957): distributional hypothesis.
-   PCA/t-SNE (van der Maaten & Hinton, 2008) if used.

The GitHub repository should be cited as the **implementation source**,
not as the primary academic source. Note that it has no license file, so
credit it explicitly.

------------------------------------------------------------------------

# 12. LaTeX report setup

Use:

``` text
report/main.tex
report/references.bib
report/sections/*.tex
report/figures/*
```

Recommended packages:

``` latex
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{xcolor}
```

For code snippets, keep them short.

Do not paste entire Python files into the report.

Recommended code presentation:

``` latex
\begin{lstlisting}[language=Python]
most_similar("king")
\end{lstlisting}
```

------------------------------------------------------------------------

# 13. LaTeX slide setup

Use Beamer:

``` text
slides/main.tex
```

Target:

> **9 slides / ~10 minutes**

Recommended slide plan:

### Slide 1 --- Title

30 seconds

### Slide 2 --- Motivation

Why do we need word representations?

1 minute

### Slide 3 --- What is Word2Vec?

Word → vector and distributional hypothesis.

1 minute

### Slide 4 --- Skip-gram

Center word → context words.

1 minute

### Slide 5 --- Negative Sampling

Positive pair vs negative pairs.

1 minute

### Slide 6 --- Implementation

``` text
text8
 ↓
preprocess
 ↓
training pairs
 ↓
SGNS
 ↓
embeddings
```

1 minute

### Slide 7 --- Experimental Results

Loss + nearest neighbours (trained vs untrained).

1.5 minutes

### Slide 8 --- Live Demo

Similarity + analogy + visualization.

2 minutes

### Slide 9 --- Conclusion

What Word2Vec achieved, limitations, applications.

1 minute

------------------------------------------------------------------------

# 14. Live demo strategy

The demo should be deterministic and short.

Prepare the trained embedding before presenting.

Do **not** rely on live training during the presentation. Keep
screenshots of the demo output in the slides as a fallback.

Demo flow:

``` text
1. Load trained embeddings
2. Query a word
3. Show nearest neighbours
4. Calculate cosine similarity
5. Run an analogy
6. Show 2D visualization
```

Example:

``` text
Input:
    king

Output:
    queen
    prince
    ...
```

Then:

``` text
king - man + woman
        ↓
     queen
```

If an analogy does not work well, do not fake the result. The
demo may show a query that works, but only if it was in the fixed query
list and the report states how many of the fixed analogies succeeded.
Otherwise, explicitly discuss why the small training setup may not
reproduce the relationship. High-frequency groups (countries, numbers,
months) are the most likely to give stable neighbours.

------------------------------------------------------------------------

# 15. Customization checklist

Before considering the project complete:

-   [x] Fork repository.
-   [x] Rename/customize project metadata (README credits upstream and lists changes).
-   [x] Confirm environment runs (conda-forge env, Python 3.11).
-   [x] Run smoke test.
-   [x] Run main experiment.
-   [x] Save exact hyperparameters (`config.json`, `--seed`).
-   [x] Save loss curve.
-   [x] Save embeddings.
-   [x] Nearest-neighbour, cosine-similarity and analogy queries
        (`python run.py evaluate`).
-   [x] Generate PCA visualization.
-   [x] Record final results.
-   [x] Create `report/main.tex`.
-   [x] Create `report/references.bib`.
-   [x] Create Beamer slides.
-   [x] Prepare demo script.
-   [ ] Re-run final experiment from a clean environment if possible.
-   [x] Check that all reported numbers come from your final experiment
        (`config.json` records the git commit of each run).

------------------------------------------------------------------------

# 16. Git workflow

Recommended branches:

``` text
main
 ├── experiment
 ├── report
 └── slides
```

Or, if the project is small:

``` text
main
```

with meaningful commits:

``` text
init: fork upstream word2vec implementation
feat: add experiment configuration
feat: add embedding evaluation
feat: add PCA visualization
docs: add experiment results
docs: add latex report
docs: add beamer slides
```

Do not commit:

-   downloaded text8;
-   large generated embedding files;
-   temporary notebook checkpoints;
-   unnecessary cache files.

`.gitignore` now covers `data/`, `output/`, caches, notebook checkpoints
and LaTeX build files. Upstream's embeddings were already committed, so
`.git` is ~128 MB; leave the history as it is.

------------------------------------------------------------------------

# 17. Suggested project milestones

## Milestone 1 --- Understand

Read only:

``` text
README.md
run.py
src/dataset/
src/model/word2vec.py
src/training/train.py
eval_embeddings.ipynb
```

Do not read every file immediately.

Goal:

> Explain the complete data flow in your own words.

------------------------------------------------------------------------

## Milestone 2 --- Run

Run:

``` text
100K tokens → 1 epoch
```

Goal:

> Confirm the code works.

------------------------------------------------------------------------

## Milestone 3 --- Experiment

Run:

``` text
1M tokens
100 dimensions
3 epochs
window 5
5 negatives
seed 42
```

Goal:

> Generate final candidate results.

------------------------------------------------------------------------

## Milestone 4 --- Analyze

Generate:

``` text
loss
nearest neighbours
cosine similarity
analogy
PCA
```

Goal:

> Understand what the model learned.

------------------------------------------------------------------------

## Milestone 5 --- Report

Write:

``` text
Introduction
→ Theory
→ Method
→ Experiment
→ Results
→ Limitations
→ Conclusion
```

Goal:

> Tell one coherent story.

------------------------------------------------------------------------

## Milestone 6 --- Slides

Convert the report into 9 slides.

Goal:

> Explain the whole project in 10 minutes without reading the report.

------------------------------------------------------------------------

# 18. What NOT to claim

Avoid statements such as:

> "Word2Vec understands language."

Prefer:

> "Word2Vec learns distributed representations that capture statistical
> relationships between words based on their contexts."

Avoid:

> "A lower loss means the model is better."

Prefer:

> "A decreasing training loss indicates that the model is optimizing the
> training objective; semantic quality should be assessed separately."

Avoid:

> "The model always learns king → queen."

Prefer:

> "The learned embedding may capture relationships such as king/queen
> when the training corpus and configuration provide sufficient
> evidence."

Avoid:

> "king and queen have a cosine similarity of 0.90, so they are very similar."

Prefer:

> "king–queen is more similar than N% of random word pairs" (with N measured, see §6.3).

Avoid reporting upstream results as your own.

------------------------------------------------------------------------

# 19. Expected final deliverables

The project should produce:

``` text
1. Customized GitHub repository
2. strategy.md
3. Reproducible training command/config
4. Trained embedding output
5. Loss curve
6. Similarity/nearest-neighbour results
7. Analogy results
8. PCA visualization
9. LaTeX report PDF
10. Beamer slide PDF
11. Short demo script/notebook
```

------------------------------------------------------------------------

# 20. Definition of Done

The project is complete when you can answer all of these without reading
the code:

### Concept

-   What problem does Word2Vec solve?
-   What is a word embedding?
-   Why does context matter?
-   What is Skip-gram?
-   What is Negative Sampling?

### Implementation

-   Where is text8 loaded?
-   How is vocabulary constructed?
-   How are training pairs generated?
-   Where is the SGNS model implemented?
-   Where are embeddings updated?
-   Where are embeddings saved?

### Experiment

-   What dataset did we use?
-   How many tokens?
-   What embedding dimension?
-   What window size?
-   How many negative samples?
-   How many epochs?
-   What happened to the loss?

### Results

-   What are the nearest neighbours?
-   Which word pairs have high similarity?
-   Does an analogy work?
-   What does the embedding visualization show?
-   What are the limitations?

### Presentation

You should be able to explain the entire project using this sentence:

> **Word2Vec learns a vector representation for each word by using its
> surrounding context; in our implementation, Skip-gram with Negative
> Sampling is trained on text8, and we evaluate the resulting embeddings
> using similarity, nearest neighbours, analogies, and visualization.**

------------------------------------------------------------------------

# 21. Source links

-   Upstream implementation:
    https://github.com/NiccoloCase/word2vec-from-scratch
-   Word2Vec paper (Skip-gram/CBOW): https://arxiv.org/abs/1301.3781
-   Word2Vec paper (negative sampling, subsampling):
    https://arxiv.org/abs/1310.4546

------------------------------------------------------------------------

# 22. First actions after forking

Start with exactly these steps:

``` bash
git clone <YOUR-FORK-URL>
cd word2vec-from-scratch

conda env create -f environment.yml
conda activate word2vec-from-scratch

python run.py train \
    --run-name smoke_test \
    --embed-dim 50 \
    --epochs 1 \
    --max-window 5 \
    --n-negatives 5 \
    --max-tokens 100000 \
    --report-every 20000
```

Then inspect:

``` text
output/smoke_test/
```

After the smoke test succeeds, move to the 1M-token demo experiment.

**Do not start writing the final report before you have run and
inspected the model at least once.**
