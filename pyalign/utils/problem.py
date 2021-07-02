import numpy as np

from pyalign.solve import Problem, MatrixProblem, IndexedMatrixProblem
from pyalign.utils.similarity import *


class SimpleProblem(MatrixProblem):
	def __init__(self, sim, s, t, **kwargs):
		super().__init__((len(s), len(t)), s, t, **kwargs)
		self._sim = sim

	def build_matrix(self, out):
		for i, x in enumerate(self.s):
			for j, y in enumerate(self.t):
				out[i, j] = self._sim.get(x, y)


class SimpleProblemFactory:
	def __init__(self, sim: Operator, direction="maximize", dtype=np.float32):
		self._sim = sim
		self._direction = direction
		self._dtype = dtype

	def new_problem(self, s, t):
		return SimpleProblem(
			self._sim, s, t,
			direction=self._direction,
			dtype=self._dtype)


class EncodedProblem(IndexedMatrixProblem):
	def __init__(self, sim, encoder, *args, **kwargs):
		self._sim = sim
		self._encoder = encoder
		super().__init__(*args, **kwargs)

	def similarity_lookup_table(self):
		return self._sim

	def build_index_sequences(self, a, b):
		encoder = self._encoder
		encoder.encode(self._s, out=a)
		encoder.encode(self._t, out=b)


class EncodedProblemFactory:
	def __init__(self, sim: Operator, encoder: Encoder, direction="maximize", dtype=np.float32):
		self._encoder = encoder
		n = len(encoder.alphabet)
		self._sim = np.empty((n, n), dtype=np.float32)
		sim.build_matrix(encoder, self._sim)
		self._direction = direction
		self._dtype = dtype

	def new_problem(self, s, t):
		return EncodedProblem(
			self._sim, self._encoder, (len(s), len(t)), s, t,
			direction=self._direction, dtype=self._dtype)


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
