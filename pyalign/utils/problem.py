import numpy as np

from pyalign.solve import Problem
from pyalign.utils.similarity import *


class SimpleProblem(Problem):
	def __init__(self, sim, *args, **kwargs):
		self._sim = sim
		super().__init__(*args, **kwargs)

	def build_matrix(self, out):
		for i, x in enumerate(self.s):
			for j, y in enumerate(self.t):
				out[i, j] = self._sim.get(x, y)


class SimpleProblemFactory:
	def __init__(self, sim: SimilarityOperator, dtype=np.float32):
		self._sim = sim
		self._dtype = dtype

	def new_problem(self, s, t, direction="maximize"):
		return SimpleProblem(
			self._sim, (len(s), len(t)), s, t,
			direction=direction, dtype=self._dtype)


class EncoderProblemFactory:
	def __init__(self, sim: SimilarityOperator, encoder: Encoder):
		self._encoder = encoder
		n = len(encoder.alphabet)
		self._sim = np.empty((n, n), dtype=np.float32)
		sim.build_matrix(encoder, self._sim)

	def new_problem(self, s, t, direction="maximize"):
		return Problem(self._sim[np.ix_(
			self._encoder.encode(s),
			self._encoder.encode(t))][:, :, np.newaxis], s, t, direction=direction)


class SpatialProblemFactory:
	def __init__(self, distance=None):
		if distance is None:
			from scipy.spatial.distance import euclidean
			distance = euclidean

		self._distance = distance

	def new_problem(self, s, t, direction="minimize"):
		m = np.empty((len(s), len(t)), dtype=np.float32)
		for i, x in enumerate(s):
			for j, y in enumerate(t):
				m[i, j] = self._distance(x, y)
		return Problem(m, s, t, direction=direction)
