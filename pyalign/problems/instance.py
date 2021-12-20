import numpy as np
import enum

from pymorton import interleave2


class Form(enum.Enum):
	MATRIX_FORM = 0
	INDEXED_MATRIX_FORM = 1
	BINARY_MATRIX_FORM = 2


class Task:
	def __init__(self, code, index):
		self.code = code
		self.index = index


class Problem:
	"""
	A problem of finding an optimal alignment between two sequences \(s\) and \(t\).
	"""

	def __init__(self, shape, s=None, t=None, direction="maximize", dtype=np.float32):
		"""

		Parameters
		----------
		shape : tuple
			The shape \( (|s|, |t|) \) of the problem
		s : array_like, optional
			The actual elements of the first sequence \(s\)
		t : array_like, optional
			The actual elements of the second sequence \(t\)
		direction : {'minimize', 'maximize'}, optional
			Direction in which to optimize
		dtype : type, optional
			dtype used for computing problem affinity (or distance) values
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
		"""problem's shape as \(|s|, |t|\)"""

		return self._shape

	@property
	def direction(self):
		"""problem's direction, i.e. either 'maximize' or 'minimize'"""

		return self._direction

	@property
	def dtype(self):
		"""dtype used for computing problem affinity (or distance) values"""

		return self._dtype

	@property
	def s(self):
		"""elements in the first sequence \( s \) or None if not available"""

		return self._s

	@property
	def t(self):
		"""elements in the second sequence \( t \) or None if not available"""

		return self._t

	def build_matrix(self, out):
		"""
		Build a matrix M that describes an alignment problem for the two sequences
		\( s \) and \( t \). Depending on the Problem's, `direction` \( M_{i, j} \)
		contains either the affinity (or distance) between \( s_i \) and \( t_j \).

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
		The matrix \( M \) built by `self.build_matrix`.
		"""

		m = np.empty(self.shape, dtype=self._dtype)
		self.build_matrix(m)
		return m

	@property
	def form(self) -> Form:
		"""the sub form the problem is posed as"""

		raise NotImplementedError()

	def __str__(self):
		return str(self.matrix)


class MatrixProblem(Problem):
	"""
	A Problem that is posed as a matrix \(M\), such that \(M_{i,j}\)
	is the affinity (or distance) between the sequence elements \(s_i\)
	and \(t_j\).
	"""

	@property
	def form(self):
		return Form.MATRIX_FORM


class IndexedMatrixProblem(Problem):
	"""
	A Problem that is posed as a matrix \(M\) and two index vectors
	\(A, B\) such that \(M_{A_i,B_j}\) is the affinity (or distance)
	between the sequence elements \(s_i\) and \(t_j\).
	"""

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


class BinaryMatrixProblem(Problem):
	def binary_similarity_values(self):
		raise NotImplementedError()

	@property
	def form(self):
		return Form.BINARY_MATRIX_FORM


class ProblemBatch:
	def __init__(self, problems, indices, master_shape):
		shapes = np.array([p.shape for p in problems])
		self._max_shape = (np.max(shapes[:, 0]), np.max(shapes[:, 1]))

		self._shape = master_shape
		self._problems = problems
		self._indices = indices

	@property
	def shape(self):
		return self._shape

	@property
	def problems(self):
		return self._problems

	@property
	def indices(self):
		return self._indices


class ProblemBag:
	def __init__(self, problems):
		self._problems = problems

		self._direction = problems[0].direction
		if not all(p.direction == self._direction for p in problems):
			raise ValueError("problems in a bag need to have same direction")

		self._dtype = problems[0].dtype
		if not all(p.dtype == self._dtype for p in problems):
			raise ValueError("problems in a bag need to have same dtype")

		self._form = Form(min(p.form.value for p in problems))

		shapes = np.array([p.shape for p in problems])
		self._max_shape = (np.max(shapes[:, 0]), np.max(shapes[:, 1]))

	def __len__(self):
		return len(self._problems)

	@property
	def problems(self):
		return self._problems

	@property
	def max_shape(self):
		return self._max_shape

	@property
	def direction(self):
		return self._direction

	@property
	def dtype(self):
		return self._dtype

	@property
	def form(self):
		return self._form

	def batches(self, batch_size):
		bag_problems = self._problems

		if batch_size == 1:
			for i, p in enumerate(bag_problems):
				yield ProblemBatch([p], [i], p.shape)
		else:
			tasks = [
				Task(interleave2(*p.shape), i) for i, p in enumerate(bag_problems)]

			sorted_i = sorted(
				np.arange(len(tasks)),
				key=lambda i: tasks[i].code)

			while True:
				while sorted_i and tasks[sorted_i[-1]].index is None:
					sorted_i.pop()

				if not sorted_i:
					break

				task = tasks[sorted_i.pop()]
				master_shape = bag_problems[task.index].shape
				indices = [task.index]

				k = 1
				while k <= len(sorted_i):
					i = sorted_i[-k]
					k += 1

					task = tasks[i]
					if task.index is None:
						continue

					s = bag_problems[task.index].shape
					if s[0] <= master_shape[0] and s[1] <= master_shape[1]:
						indices.append(task.index)
						task.index = None

						if len(indices) == batch_size:
							break

				yield ProblemBatch(
					[bag_problems[i] for i in indices],
					indices,
					master_shape)
