# Word2Vec from Scratch — Submission

**Code:** [github.com/NiccoloCase/word2vec-from-scratch](https://github.com/NiccoloCase/word2vec-from-scratch)

**Derivation report:** [implementation.pdf](https://github.com/NiccoloCase/word2vec-from-scratch/blob/main/implementation.pdf)

---

## Implementation

I implemented the skip-gram variant of Word2Vec with negative sampling, following Mikolov et al. (2013), entirely in NumPy. I trained on the text8 (Wikipedia text) using 5M tokens.

The implementation covers:

- Basic data preprocessing (tokenization and subsampling of frequent words)
- Model definition and forward and backward passes in NumPy
- Training loop with random window sampling and negative sampling
- SGD with linear learning rate decay
- Utilities for evaluation

The repository also includes a detailed `README` with project structure, training instructions, and design choices, and `eval_embeddings.ipynb`, a Jupyter notebook for exploring nearest neighbours and analogies on the trained embeddings.

Also, to explain the derivation of the gradients and loss function, I have included a separate `implementation.pdf` document.

---

As training was not the main focus of this assignment, I trained only for 1 epoch on 5M tokens. I used an embedding dimension of 100 (D=100), K=5 negative samples, and max window size of 5. Training ran for approximately 2.5 hours on a Snellius HPC cluster (Rome partition). The loss decreased as expected, and the nearest neighbours in the embedding space showed reasonable semantic relationships, though the analogy performance was poor, suggesting that the vector space was not fully structured yet. With more epochs and a larger embedding dimension, I would expect the performance to improve significantly.

---

This assignment aligns closely with my academic background. In my Deep Learning 1 course, I implemented an MLP both in PyTorch and in pure NumPy, and my Machine Learning 1 course emphasized deriving backpropagation manually. Although I had previously studied Word2Vec and skip-gram theoretically in my NLP course, implementing it from scratch significantly deepened my understanding of negative sampling and contrastive learning in practice.

---

Below is the core training step. The full implementation, including detailed inline comments explaining each gradient and update step, is available in the repository.

```python
def step(self,
         centre_idx: int,      # index of the centre word
         pos_idx: int,         # index of the positive context word
         neg_idxs: np.ndarray, # indices of the K negative samples
         lr: float             # learning rate for this step
     ) -> float:

    # FORWARD
    v_c = self.W_in[centre_idx]
    v_o = self.W_out[pos_idx]
    V_n = self.W_out[neg_idxs]

    s_o  = v_c @ v_o
    sig_o = sigmoid(s_o)

    s_k  = V_n @ v_c
    sig_k = sigmoid(s_k)

    loss = -np.log(sig_o + eps) - np.sum(np.log(1.0 - sig_k + eps))

    # BACKWARD
    grad_s_o = sig_o - 1.0
    grad_s_k = sig_k
    grad_vc = grad_s_o * v_o + grad_s_k @ V_n
    grad_vo = grad_s_o * v_c
    grad_Vn = np.outer(grad_s_k, v_c)

    # SGD
    self.W_in[centre_idx]  -= lr * grad_vc
    self.W_out[pos_idx]    -= lr * grad_vo
    np.add.at(self.W_out, neg_idxs, -lr * grad_Vn)

    return float(loss)
```
