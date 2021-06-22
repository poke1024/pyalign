import numpy as np
from .solve import Problem


class Encoder:
	def __init__(self, alphabet):
		self._alphabet = set(alphabet)
		self._ids = dict((k, i) for i, k in enumerate(self._alphabet))

	def encode(self, s):
		ids = self._ids
		return np.array([ids[x] for x in s], dtype=np.uint32)

	@property
	def alphabet(self):
		return self._alphabet


class SimilarityOperator:
	def __init__(self, sim={}, eq=1, neq=0):
		self._sim = sim  # FIXME symmetry
		self._eq = eq
		self._neq = neq

	def get(self, u, v):
		w = self._sim.get((u, v))
		if w is not None:
			return w
		if u == v:
			return self._eq
		else:
			return self._neq

	def to_matrix(self, encoder):
		n = len(encoder.alphabet)
		m = np.full((n, n), self._neq, dtype=np.float32)
		np.fill_diagonal(m, self._eq)
		for uv, w in self._sim.items():
			i, j = encoder.encode(uv)
			m[i, j] = w
			m[j, i] = w
		return m


class EncoderProblemFactory:
	def __init__(self, sim, encoder):
		self._encoder = encoder
		self._sim = sim.to_matrix(encoder)

	def new_problem(self, s, t):
		return Problem(self._sim[np.ix_(
			self._encoder.encode(s),
			self._encoder.encode(t))], s, t)


class SimpleProblemFactory:
	def __init__(self, sim):
		self._sim = sim

	def new_problem(self, s, t):
		m = np.empty((len(s), len(t)), dtype=np.float32)
		for i, x in enumerate(s):
			for j, y in enumerate(t):
				m[i, j] = self._sim.get(x, y)
		return Problem(m, s, t)

