from pyalign.problem.instance import MatrixProblem, IndexedMatrixProblem
from pyalign.problem.function import *


class Problem(MatrixProblem):
	def __init__(self, f, s, t, **kwargs):
		super().__init__((len(s), len(t)), s, t, **kwargs)
		self._f = f

	def build_matrix(self, out):
		for i, x in enumerate(self.s):
			for j, y in enumerate(self.t):
				out[i, j] = self._f(x, y)


class ProblemFactory:
	def __init__(self, f: Operator, direction="maximize", dtype=np.float32):
		if f is None and direction == "minimize":
			from scipy.spatial.distance import euclidean
			f = euclidean

		self._f = f
		self._direction = direction
		self._dtype = dtype

	def new_problem(self, s, t):
		return Problem(
			self._f, s, t,
			direction=self._direction,
			dtype=self._dtype)


class AlphabetProblem(IndexedMatrixProblem):
	def __init__(self, matrix, encoder, *args, **kwargs):
		self._matrix = matrix
		self._encoder = encoder
		super().__init__(*args, **kwargs)

	def similarity_lookup_table(self):
		return self._matrix

	def build_index_sequences(self, a, b):
		encoder = self._encoder
		encoder.encode(self._s, out=a)
		encoder.encode(self._t, out=b)


class AlphabetProblemFactory:
	def __init__(self, f: Operator, encoder: Encoder, direction="maximize", dtype=np.float32):
		self._encoder = encoder
		n = len(encoder.alphabet)
		self._matrix = np.empty((n, n), dtype=np.float32)
		f.build_matrix(encoder, self._matrix)
		self._direction = direction
		self._dtype = dtype

	def new_problem(self, s, t):
		return AlphabetProblem(
			self._matrix, self._encoder, (len(s), len(t)), s, t,
			direction=self._direction, dtype=self._dtype)
