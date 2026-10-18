import re
import numpy as np

def tokenise(text: str) -> list[str]:
    "Very simple but effective tokeniser: we just split on non-alphabetic characters and lowercase everything"
    tokens = re.findall(r"[a-z]+", text.lower())
    return tokens

def build_noise_distribution(freqs: list[int], power: float = 0.75) -> np.ndarray:
    """
    Unigram distribution raised to the 3/4 power.
    Used to draw negative samples, preventing very frequent words from dominating.
    """
    arr = np.array(freqs, dtype=np.float64) ** power
    return arr / arr.sum()

class NegativeSampler:
    """
    Draws negative samples from the noise distribution in large batches.
    np.random.choice with p= rebuilds the cumulative distribution over the whole vocabulary on every call (O(V)),
    which dominated the training time when called once per (centre, context) pair. Drawing many i.i.d. samples
    at once and consuming them in order yields samples from the same distribution at a fraction of the cost.
    """
    def __init__(self, noise_dist: np.ndarray, buffer_size: int = 1_000_000) -> None:
        self.noise_dist = noise_dist
        self.buffer_size = buffer_size
        self._buffer = np.empty(0, dtype=np.int64)
        self._pos = 0

    def sample(self, n: int) -> np.ndarray:
        if self._pos + n > len(self._buffer):
            self._buffer = np.random.choice(len(self.noise_dist), size=max(self.buffer_size, n), p=self.noise_dist)
            self._pos = 0
        out = self._buffer[self._pos:self._pos + n].copy()  # copy: callers modify the result in place
        self._pos += n
        return out

def sigmoid(x: np.ndarray) -> np.ndarray:
    "Sigmoid function"
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )


def learning_rate_decay(lr0: float, lr_min: float, current_step: int, total_steps: int) -> float:
    """
    Computes the learning rate using linear decay scheduling.
    
    The learning rate starts at lr0 and linearly decreases to lr_min over total_steps.

    Args:
        lr0: Initial learning rate.
        lr_min: Minimum learning rate (floor value).
        current_step: Current training step/iteration.
        total_steps: Total number of training steps.
    
    Returns:
        float: Current learning rate.

    """
    return max(lr_min, lr0 * (1.0 - current_step / total_steps))

