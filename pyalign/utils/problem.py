import numpy as np

from pyalign.solve import MatrixProblem, IndexedMatrixProblem
from pyalign.utils.similarity import *


class GeneralProblem(MatrixProblem):
	def __init__(self, sim, s, t, **kwargs):
		super().__init__((len(s), len(t)), s, t, **kwargs)
		self._sim = sim

	def build_matrix(self, out):
		for i, x in enumerate(self.s):
			for j, y in enumerate(self.t):
				out[i, j] = self._sim.get(x, y)


class GeneralProblemFactory:
	def __init__(self, sim: Operator, direction="maximize", dtype=np.float32):
		self._sim = sim
		self._direction = direction
		self._dtype = dtype

	def new_problem(self, s, t):
		return GeneralProblem(
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


class SpatialProblem(MatrixProblem):
	def __init__(self, distance, s, t, **kwargs):
		super().__init__((len(s), len(t)), s, t, **kwargs)
		self._distance = distance

	def build_matrix(self, out):
		for i, x in enumerate(self.s):
			for j, y in enumerate(self.t):
				out[i, j] = self._distance(x, y)


class SpatialProblemFactory:
	def __init__(self, distance=None, direction="minimize", dtype=np.float32):
		if distance is None:
			from scipy.spatial.distance import euclidean
			distance = euclidean

		self._distance = distance
		self._direction = direction
		self._dtype = dtype

	def new_problem(self, s, t):
		return SpatialProblem(
			self._distance, s, t, direction=self._direction, dtype=self._dtype)
