from pathlib import Path

import numpy as np
from utils import sigmoid, learning_rate_decay, NegativeSampler
import time
from tqdm import tqdm
from dataset.vocabulary import Vocabulary

eps = 1e-10

class MyWord2Vec:
  
    def __init__(self,
                 vocab: Vocabulary, # vocabulary to own and serialize
                 embed_dim: int = 100, # (D) dimensionality of embeddings 
                 k: int = 5, # (K) number of negative samples for contrastive learning
                 start_lr: float = 0.025, # starting learning rate
        ): 

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embed_dim = embed_dim
        self.k = k
        self.start_lr = start_lr

        # W_in -> word embeddings  (V, D)
        self.W_in  = (np.random.rand(self.vocab_size, embed_dim) - 0.5) / embed_dim # uniform init 

        # W_out -> context embeddings (V, D)
        self.W_out = np.zeros((self.vocab_size, embed_dim), dtype=np.float64) # zero init 


    def step(self,
            centre_idx: int, # index of the centre word
            pos_idx: int, # index of the positive context word 
            neg_idxs: np.ndarray, # indices of the negative samples 
            lr: float # learning rate for this step
        ) -> float:
        
        """
        Perform a single training step for one (centre, context) pair and K negative samples.
        Runs the forward and backward passes, and updates the embeddings in-place with SGD.
        Returns the loss for this training example.
        """
            

        # notation:
        # c: centre word
        # o: positive context word
        # k: negative samples


        # FORWARD
        
        # first we get the word embedding for the target word 
        v_c = self.W_in[centre_idx] # (D)

        # for the positive and negative samples we get their context embeddings
        v_o = self.W_out[pos_idx]   # (D) 
        V_n = self.W_out[neg_idxs]  # (K, D)


        # now we can compute the scores for the positive and negative samples as thier dot product with the centre embedding

        # positive score between centre embedding and positive context embedding
        s_o  = v_c @ v_o  # (D) @ (D) -> scalar
        sig_o = sigmoid(s_o) # σ(s_o) = P(pos | centre)

        # negative scores between centre embedding and negative context embeddings
        s_k = V_n @ v_c # (K, D) @ (D) -> (K)
        sig_k = sigmoid(s_k) # σ(s_k) = P(neg | centre)


        # now we compute the loss for this training example
        # negative log-likelihood:
        # L = − y log(σ(s)) − (1−y) log(1−σ(s)) 
        # for the positive:
        # L_pos = -log σ(s_o)  (since y=1)
        # for the negatives:
        # L_neg = - sum_k { log(1 - σ(s_k)) }  (since y=0)
        # putting it together:
        # L = L_pos + L_neg = -log σ(s_o) - sum_k { log(1 - σ(s_k)) }       
        loss = -np.log(sig_o + eps) - np.sum(np.log(1.0 - sig_k + eps)) 

        # BACKWARD
        # we have to perform backpropagation to compute the gradients for the input and output embeddings


        # 1) derivative of the loss w.r.t. the positive scores (s_o)

        # we consider only the positive part of the loss: -log σ(s_o)
        # and we know that the derivative of the sigmoid function is σ'(x) = σ(x) * (1 - σ(x))
        # for the chain rule we have that the derivative of -log σ(s_o) w.r.t. s_o is:
        # -1/σ(s_o) * σ'(s_o) = -1/σ(s_o) * σ(s_o)(1 - σ(s_o)) = -(1 - σ(s_o)) = σ(s_o) - 1
        grad_s_o = sig_o - 1.0                


        # 2) derivative of the loss w.r.t. the negative scores (s_k)

        # we consider only the negative part of the loss: - sum_k { log(1 - σ(s_k)) }
        # we fix a specifc k and we compute the derivative w.r.t. s_k:
        # -1/(1 - σ(s_k)) * (-σ'(s_k)) = σ'(s_k) / (1 - σ(s_k)) = σ(s_k) * (1 - σ(s_k)) / (1 - σ(s_k)) = σ(s_k) 
        grad_s_k = sig_k     


        # 3)  derivaive of the loss w.r.t. the centre embedding v_c 

        # dL = dL_pos + dL_neg
        # dL_pos / dv_c = dL/ds_o * ds_o/dv_c 
        # we also know that s_o = v_c @ v_o so for the multiplication rule ds_o/dv_c = v_o
        grad_vc_pos = grad_s_o * v_o  # (D)

        # dL_neg / d v_c = sum_k { dL/ds_k * ds_k/dv_c }
        # s_k = V_n @ v_c => ds_k/dv_c = V_n
        grad_vc_neg = grad_s_k @ V_n  # (K) @ (K, D) -> (D)

        grad_vc = grad_vc_pos + grad_vc_neg  # (D)


        # 4) derivative of the loss w.r.t. the positive context embedding v_o:

        # we consider only the positive part of the loss
        # dL/dv_o = dL/ds_o * ds_o/dv_o
        # ds_o/dv_o = v_c
        grad_vo = grad_s_o * v_c  # (D)
        
        
        # 5) derivative of the loss w.r.t. the negative context embeddings V_n:

        # we consider only the negative part of the loss
        # for a specific k we have:
        # dL/dV_n[k] = dL/ds_k * ds_k/dV_n[k]
        # s_k = V_n @ v_c => ds_k/dV_n[k] = v_c
        grad_Vn = np.outer(grad_s_k, v_c) # (K) @ (D) -> (K, D)


        # OPTIMIZER:
        # Stochastic gradient descent update
        self.W_in[centre_idx]  -= lr * grad_vc
        self.W_out[pos_idx]    -= lr * grad_vo
        np.add.at(self.W_out, neg_idxs, -lr * grad_Vn) # fix for possible duplicate indices in neg_idxs

        return float(loss)
    
    
    def train(self,
              train_tokens: list[int], # list of token indices for the training corpus
              noise_dist: np.ndarray, # (V,) array with the noise distribution for negative sampling
              max_window: int   = 5, # max window size for context words
              epochs: int   = 5, # number of epochs to train for
              lr_min: float = 1e-4, # minimum learning rate for linear decay
              report_every: int  = 500_000
                        ) -> tuple[list[tuple[int, float]], list[float]]:
        """
        Training loop

        - For each centre word in the training corpus, we sample a random window size W, and we consider the W words to the left and W words to the right as positive context words.
        - For each (centre, context) pair we then sample K negative examples from the noise distribution
        - We then perform a training step with the centre word, one positive context word and K negative samples, and we update the embeddings in-place with stochastic gradient descent.

        Returns:
        - loss_history: (global_step, avg_loss) tuples every report_every steps, where avg_loss is the mean loss
          over the steps since the previous report (not a running average since the start of the epoch)
        - epoch_losses: the mean loss of each epoch
        """

        num_tokens = len(train_tokens)
        sampler = NegativeSampler(noise_dist)
        loss_history: list[tuple[int, float]] = []  # (global_step, avg_loss)
        epoch_losses: list[float] = []


        # for the learning rate decay we need an estimate of the total number of training steps
        # as w is drawn uniformly from 1 to max_window, the average window size is (max_window + 1) / 2
        # for each centre word we have on average 2 * (max_window + 1) / 2 = max_window + 1 context words
        approx_total_steps = epochs * num_tokens * (max_window + 1)

        print(f"Estimated training steps: {approx_total_steps:,} (epochs={epochs}, tokens={num_tokens:,}, max_window={max_window})")


        lr0 = self.start_lr
        t0 = time.time()

        global_step = 0
        total_loss = 0.0
        window_loss = 0.0  # loss accumulated since the previous report
        window_steps = 0

        for epoch in tqdm(range(epochs), desc="Epochs"):
            epoch_loss = 0.0
            epoch_steps = 0

            # for each centre word in the training corpus
            for i, centre_idx in enumerate(train_tokens):

                # draw the window size for this training example
                W = np.random.randint(1, max_window + 1)
                start = max(0, i - W)
                end   = min(num_tokens, i + W + 1)

                pos_context_idxs = train_tokens[start:i] + train_tokens[i+1:end]

                for pos_idx in pos_context_idxs:
                    # sample negatives examples 
                    neg = sampler.sample(self.k)

                    # we should re-draw negative samples until we have no overlap with the positive context word and the centre word
                    mask = (neg == pos_idx) | (neg == centre_idx)
                    while np.any(mask):
                        neg[mask] = sampler.sample(mask.sum())
                        mask = (neg == pos_idx) | (neg == centre_idx)


                    # learing rate decay
                    lr = learning_rate_decay(lr0, lr_min, global_step, approx_total_steps)

                    # forward and backward pass
                    loss = self.step(centre_idx, pos_idx, neg, lr=lr)  


                    epoch_loss += loss
                    window_loss += loss
                    global_step += 1
                    epoch_steps += 1
                    window_steps += 1

                    if global_step % report_every == 0:
                        elapsed = time.time() - t0
                        avg_loss = window_loss / window_steps
                        loss_history.append((global_step, avg_loss))
                        window_loss, window_steps = 0.0, 0
                        print(f"Step {global_step}, Avg Loss (last {report_every:,} steps): {avg_loss:.4f}, Elapsed: {elapsed:.2f}s")

            avg_loss = epoch_loss / max(1, epoch_steps)
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs} done! avg loss {avg_loss:.4f}")

            total_loss += epoch_loss

        # flush the last, partial window so that short runs still record at least one point
        if window_steps > 0:
            loss_history.append((global_step, window_loss / window_steps))

        print(f"Training completed in {time.time() - t0:.2f}s, Total Loss: {total_loss:.4f}")
        return loss_history, epoch_losses


    def save_embeddings(self, file_path):
        """
        Save model weights and the owned vocabulary to a .npz file.
        """

        if self.vocab is None:
            raise ValueError("Vocabulary missing; cannot save model.")

        out_path = Path(file_path)
        if out_path.suffix != ".npz":
            raise ValueError("file_path should end with .npz")
        
        out_file = out_path
        np.savez(
            out_file,
            W_in=self.W_in,
            W_out=self.W_out,
            vocab_state=self.vocab.to_state(),
            k=self.k,
            start_lr=self.start_lr,
            embed_dim=self.embed_dim,
        )

    @classmethod
    def load(cls, file_path: str) -> "MyWord2Vec":
        """
        Load a model and its vocabulary from disk.
        """
        in_path = Path(file_path)
        if not in_path.exists():
            raise FileNotFoundError(f"{file_path} not found")
        if in_path.suffix != ".npz":
            raise ValueError("file_path should end with .npz")

        data = np.load(in_path, allow_pickle=True)
        W_in = data["W_in"]
        W_out = data["W_out"]

        if "vocab_state" in data.files:
            vocab_state = data["vocab_state"].item()
            vocab = Vocabulary.from_state(vocab_state)
        elif "words" in data.files:
            words = data["words"].tolist()
            fallback_state = {
                "words": words,
                "freqs": [1 for _ in words],
                "counts": [1 for _ in words],
            }
            vocab = Vocabulary.from_state(fallback_state)
        else:
            raise KeyError("No vocabulary data found in saved file.")

        k = int(data["k"]) if "k" in data.files else 5
        start_lr = float(data["start_lr"]) if "start_lr" in data.files else 0.025
        embed_dim = W_in.shape[1]

        model = cls(vocab=vocab, embed_dim=embed_dim, k=k, start_lr=start_lr)
        model.W_in = W_in
        model.W_out = W_out
        return model

    @classmethod
    def load_embeddings(cls, file_path: str) -> "MyWord2Vec":
        """Alias for load to ease migration from the old API."""
        return cls.load(file_path)


    def get_most_similar(self,
                     word: str,
                     vocab: Vocabulary | None = None, # word to index mapping
                     top_k: int = 10) -> list[tuple[str, float]]:
        """
        Find the top k most similar words to the given word, based on cosine similarity of the input embeddings.
        Returns a list of (word, similarity) 
        """
        vocab = vocab or self.vocab
        if vocab is None: raise ValueError("Vocabulary missing; pass one explicitly or load a saved model.")

        if word not in vocab.word2idx: raise KeyError(f"'{word}' not in vocabulary")

        # L2-normalise the matrix 
        # in this way the cosine similarity can be computed as a dot product between the normalised embeddings
        norms = np.linalg.norm(self.W_in, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        W_norm = self.W_in / norms

        # embedding of the query word
        query = W_norm[vocab.word2idx[word]] 

        # calc the similarity between the query embedding and all the other embeddings as a dot product
        sims  = W_norm @ query # (V, D) @ (D) -> (V)
        sims[vocab.word2idx[word]] = -1.0   # exclude the query word itself from the results       


        # get the top k indices with the highest similarity
        top_ids = np.argsort(sims)[-top_k:][::-1]

        return [(vocab.idx2word[i], float(sims[i])) for i in top_ids]
    

    def analogy(self, a: str, b: str, c: str, top_k: int = 5):
        """
        Given an analogy query of the form "a is to b as c is to ?", find the top k most likely answers based on cosine similarity.
        The target vector is b - a + c, so "king - man + woman" is analogy("man", "king", "woman").
        """

        vocab = self.vocab
        for w in (a, b, c):
            if w not in vocab.word2idx:
                raise KeyError(f"'{w}' not in vocabulary")

        W = self.W_in

        norms = np.linalg.norm(W, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        W_norm = W / norms

        target = W_norm[vocab.word2idx[b]] - W_norm[vocab.word2idx[a]] + W_norm[vocab.word2idx[c]]
        # normalise the target too, so that the scores are cosine similarities in [-1, 1] (the ranking is unchanged)
        target /= max(np.linalg.norm(target), eps)

        sims = W_norm @ target

        for w in (a, b, c):
            sims[vocab.word2idx[w]] = -np.inf

        top_ids = np.argpartition(sims, -top_k)[-top_k:]
        top_ids = top_ids[np.argsort(sims[top_ids])[::-1]]

        return [(vocab.idx2word[i], float(sims[i])) for i in top_ids]


    

