from pyalign.problem.instance import MatrixProblem, IndexedMatrixProblem
from pyalign.problem.function import *


class Problem(MatrixProblem):
	"""
	A Problem induced by some binary function \(f(x, y)\).
	"""

	def __init__(self, f, s, t, **kwargs):
		super().__init__((len(s), len(t)), s, t, **kwargs)
		self._f = f

	def build_matrix(self, out):
		for i, x in enumerate(self.s):
			for j, y in enumerate(self.t):
				out[i, j] = self._f(x, y)


class ProblemFactory:
	"""
	A factory for problems that can be built from some binary function
	\(f(x, y)\).
	"""

	def __init__(self, f: callable, direction="maximize", dtype=np.float32):
		"""

		Parameters
		----------
		f : callable
			binary function \(f(x, y)\) that returns a measure of affinity (or
			distance) between two arbitrary elements (e.g. characters) \(x\) and
			\(y\).
		direction : {'minimize', 'maximize'}
			direction of problems created by this factory
		dtype : type
			dtype of values returned by \(f\)
		"""

		if f is None and direction == "minimize":
			from scipy.spatial.distance import euclidean
			f = euclidean

		self._f = f
		self._direction = direction
		self._dtype = dtype

	def new_problem(self, s, t):
		"""
		Creates a new alignment problem for the sequences \(s\) and \(t\)

		Parameters
		----------
		s : array_like
			first sequence
		t : array_like
			second sequence

		Returns
		-------
		Problem modelling optimal alignment between \(s\) and \(t\)
		"""

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
	"""
	A factory for alignment problems involving sequences \(s, t\) that can be
	written using a small fixed alphabet \(\Omega\) such that \(∀i: s_i \in
	\Omega\), \(∀j: t_j \in \Omega\).
	"""

	def __init__(self, f: callable, encoder: Encoder, direction="maximize", dtype=np.float32):
		"""
		Parameters
		----------
		f : callable
			binary function \(f\) such that \(f(E(x), E(y))\) is a measure of
			affinity (or distance) between two items \(x, y\)
		encoder : Encoder
			Encoder \(E\) that gives \(E: \Omega → \mathbb{N}\)
		direction : {'minimize', 'maximize'}
			direction of problems created by this factory
		dtype
			dtype of values returned by \(f\)
		"""

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
