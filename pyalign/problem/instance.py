import numpy as np
import enum


class Problem:
	"""
	An alignment problem between two sequences.
	"""

	def __init__(self, shape, s=None, t=None, direction="maximize", dtype=np.float32):
		"""

		Parameters
		----------
		shape : tuple
			The shape \( (n, m) \) of the problem, with n being the length of the first
			sequence s being aligned, and m being the length of the second sequence t
		s : array_like, optional
			The actual elements of the first sequence s
		t : array_like, optional
			The actual elements of the second sequence t
		direction : {'minimize', 'maximize'}, optional
		dtype : type, optional
		"""

		self._shape = tuple(shape)
		self._direction = direction
		self._dtype = dtype

		if s is not None and shape[0] != len(s):
			raise ValueError(f"sequence s [{len(s)}] does not match matrix shape {shape}")
		if t is not None and shape[1] != len(t):
			raise ValueError(f"sequence t [{len(t)}] does not match matrix shape {shape}")

		self._s = s
		self._t = t

	@property
	def shape(self):
		"""The problem's shape"""

		return self._shape

	@property
	def direction(self):
		"""The problem's direction, i.e. either 'maximize' or 'minimize'"""

		return self._direction

	@property
	def dtype(self):
		"""The dtype used for the problem affinity or distance values"""

		return self._dtype

	@property
	def s(self):
		"""The elements in the first sequence \( s \) or None if not available"""

		return self._s

	@property
	def t(self):
		"""The elements in the second sequence \( t \) or None if not available"""

		return self._t

	def build_matrix(self, out):
		"""
		Build a matrix M that describes an alignment problem for the two sequences
		\( s \) and \( t \). Depending on the Problem's, `direction` \( M_{i, j} \)
		contains either the affinity or distance between \( s_i \) and \( t_j \).

		Parameters
		----------
		out : array_like
			A suitably shaped matrix that receives \( M \).

		Returns
		-------

		"""
		raise NotImplementedError()

	@property
	def matrix(self):
		"""
		Returns
		-------
		The matrix \( M \) built by `self.built_matrix`.
		"""

		m = np.empty(self.shape, dtype=self._dtype)
		self.build_matrix(m)
		return m


class Form(enum.Enum):
	MATRIX_FORM = 0
	INDEXED_MATRIX_FORM = 1


class MatrixProblem(Problem):
	@property
	def form(self):
		return Form.MATRIX_FORM


class IndexedMatrixProblem(Problem):
	def build_matrix(self, out):
		sim = self.similarity_lookup_table()
		a = np.empty((self.shape[0],), dtype=np.uint32)
		b = np.empty((self.shape[1],), dtype=np.uint32)
		self.build_index_sequences(a, b)
		out[:, :] = sim[np.ix_(a, b)]

	def similarity_lookup_table(self):
		raise NotImplementedError()

	def build_index_sequences(self, a, b):
		raise NotImplementedError()

	@property
	def form(self):
		return Form.INDEXED_MATRIX_FORM


class ProblemBatch:
	def __init__(self, problems):
		self._problems = problems

		self._shape = problems[0].shape
		if not all(p.shape == self._shape for p in problems):
			raise ValueError("problems in a batch need to have same shape")

		self._direction = problems[0].direction
		if not all(p.direction == self._direction for p in problems):
			raise ValueError("problems in a batch need to have same direction")

		self._dtype = problems[0].dtype
		if not all(p.dtype == self._dtype for p in problems):
			raise ValueError("problems in a batch need to have same dtype")

		self._form = Form(min(p.form.value for p in problems))

	def __len__(self):
		return len(self._problems)

	@property
	def problems(self):
		return self._problems

	@property
	def shape(self):
		return self._shape

	@property
	def direction(self):
		return self._direction

	@property
	def dtype(self):
		return self._dtype

	@property
	def form(self):
		return self._form
