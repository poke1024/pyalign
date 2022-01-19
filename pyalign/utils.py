import numpy as np


class RandomSequenceGenerator:
    def __init__(self, alphabet, seed=1234):
        self._alphabet = tuple(set(alphabet))
        self._rng = np.random.default_rng(seed)

    def __call__(self, n):
        a = self._alphabet
        return ''.join([a[i] for i in self._rng.integers(0, len(a), n)])

