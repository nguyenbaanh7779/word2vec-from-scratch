# Speaker notes — Learning Word Embeddings from Scratch

Target: 10 minutes, 9 slides. Times are cumulative targets. Before the talk: activate the conda environment, open a terminal in the repository root, and run `python demo/demo.py --script` once so the embeddings are in the file cache.

---

## Slide 1 — Title (0:00–0:30)

Hello everyone. We are Bá Anh and Nhã Linh. Our project is about Word2Vec: we trained word embeddings from scratch, with nothing but NumPy, on a slice of Wikipedia, and then checked what the vectors actually learned.

## Slide 2 — Motivation (0:30–1:30)

Any NLP model has to turn words into numbers first. The simplest way is a one-hot vector: one dimension per word in the vocabulary. In our main experiment that is over 36,000 dimensions, and every pair of words is orthogonal, so *king* is exactly as far from *queen* as from *carrot*. Nothing about meaning is encoded.

Word embeddings replace this with a dense vector of, here, 100 numbers, learned from text. The idea behind them is the distributional hypothesis, summed up by Firth: you shall know a word by the company it keeps. Words that appear in similar contexts should get similar vectors. So our question is: how can a model learn useful vectors from context alone?

## Slide 3 — What is Word2Vec? (1:30–2:30)

Word2Vec is a very shallow model: just two matrices. Each word has an input vector, used when it is the centre word, and an output vector, used when it is a context word. The input vectors are the embeddings we keep.

There are two variants. CBOW predicts the centre word from its context; Skip-gram does the opposite and predicts every context word from the centre word. We use Skip-gram.

Written as a softmax, the probability of a context word needs a sum over the whole vocabulary. That sum has to be computed for every training pair, and we have tens of millions of pairs, so this is far too slow.

## Slide 4 — Skip-gram (2:30–3:30)

Here is what a training pair looks like. Take the centre word *king*. We draw a window size at random between 1 and 5 — here it is 2 — and every word inside the window, *crowned*, *the*, *and*, *queen*, forms a positive pair with *king*. Because the window size is random, close neighbours are included more often than distant ones. On average each centre word gives about six pairs.

Each positive pair also gets some negative pairs — that is the next slide.

## Slide 5 — Negative sampling (3:30–4:30)

Negative sampling turns the expensive softmax into a cheap yes/no question: is this a real pair from the text, or a random one? For each true pair, we draw K = 5 random words. The first term pulls the centre vector towards its real context word; the second pushes it away from the random words. That costs five dot products instead of thirty-six thousand.

Random words are drawn in proportion to frequency to the power three quarters, which gives rare words a bit more chance. We also subsample very frequent words like *the*, which carry little information about their neighbours.

## Slide 6 — Implementation (4:30–5:30)

This is the pipeline, with the file that implements each step. The model code comes from an open-source NumPy repository by Niccolò Caselli; we want to be clear about that. What we added is in the three columns.

First, reproducibility: a random seed, every hyperparameter on the command line, and a record file for every run. The documented training command did not even run, and we fixed that.

Second, speed: profiling showed that drawing negative samples took most of the time, because NumPy rebuilt a probability table over the whole vocabulary for every single pair. We draw them in large batches from the same distribution instead. On the one-million-token run, training went from 17 minutes to about 4, with the same losses.

Third, an evaluation protocol, which we will come back to with the results.

## Slide 7 — Results: loss and neighbours (5:30–7:00)

Our main run uses the first five million tokens of text8, about 29% of it, with 100 dimensions, a window of 5, 5 negatives and 3 epochs. It took about 25 minutes on one laptop core.

On the left is the training loss. It falls very quickly early in the first epoch and then stays flat. The average loss of the third epoch is even slightly higher than that of the second, and the original code does exactly the same, so the training loss alone does not tell us how good the embeddings are. That is why we evaluate the embeddings directly.

On the right are the nearest neighbours of eight words that we fixed before looking at any results. *January* gives other months, *France* gives European countries, *three* gives numbers, *music* gives genres, *computer* gives computing words. *King* gives specific monarchs and *throne*, but not *queen*. The untrained model, for comparison, gives random words: *marpat*, *verbally*, *brando*.

## Slide 8 — Live demo (7:00–9:00)

*(Switch to the terminal and run `python demo/demo.py`. If anything fails, the same outputs are on the slide.)*

First, *king queen*: cosine 0.70. On its own that number is hard to read, because in this model random pairs of frequent words already average about 0.32, and in the smaller model 0.61. So we report a percentile: *king* and *queen* are more similar than 99.7% of random pairs. *King* and *software*: only 5%.

Now analogies. *bigger − big + small* gives *smaller*. *king − man + woman* gives *daughter*, *caligula*… and *queen* is not even in the top ten. We show this on purpose. Of our eight fixed analogies, four are solved exactly and six are in the top five, up from zero and two with one million tokens. But this famous one fails, probably because five million tokens are not enough to learn the relation consistently.

On the right is a PCA of 32 words from four groups that we chose in advance: countries, numbers, months, family and royalty. After training they form four separate clusters; before training, the same plot shows no structure at all.

## Slide 9 — Conclusion (9:00–10:00)

To conclude. The loss shows that the objective is optimised, but not how good the embeddings are. Similar words do end up close together, as long as we compare against the right baseline. Analogies work only partly, and the word groups form clear clusters.

The limitations: a small slice of Wikipedia, rare words that are barely trained, one vector per word, so *bank* has the same vector for both of its meanings, and a small, fixed evaluation with a single seed.

Embeddings like these are used as input features for text classifiers, for similarity search and query expansion, or to explore how a corpus uses words.

In one sentence: Word2Vec learns a vector for each word from its context; we trained Skip-gram with negative sampling on text8 and evaluated the result with neighbours, similarity, analogies and visualisation. Thank you, we are happy to take questions.
