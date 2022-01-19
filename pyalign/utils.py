import numpy as np


def random_seq(alphabet, n, seed=1234):
    np.random.seed(seed)
    return ''.join([alphabet[i] for i in np.random.randint(0, len(alphabet), n)])

