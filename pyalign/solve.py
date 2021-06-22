import pyalign.algorithm


class Problem:
	def __init__(self, matrix, s=None, t=None):
		self._matrix = matrix
		if s is not None and matrix.shape[0] != len(s):
			raise ValueError(f"sequence s [{len(s)}] does not match matrix shape {matrix.shape}")
		if t is not None and matrix.shape[1] != len(t):
			raise ValueError(f"sequence t [{len(t)}] does not match matrix shape {matrix.shape}")
		self._s = s
		self._t = t

	@property
	def matrix(self):
		"""
		Returns a matrix M that describes a problem for two sequences \( s \) and \( t \).
		\( M_{i, j} \) contains the similarity between \( s_i \) and \( t_j \).
		"""

		return self._matrix

	@property
	def shape(self):
		return self._matrix.shape

	@property
	def s(self):
		return self._s

	@property
	def t(self):
		return self._t


class Alignment:
	def __init__(self, problem, alignment):
		self._problem = problem
		self._alignment = alignment

	@property
	def problem(self):
		return self._problem

	@property
	def score(self):
		return self._alignment.score

	@property
	def s_to_t(self):
		return self._alignment.s_to_t

	@property
	def t_to_s(self):
		return self._alignment.t_to_s

	def print(self):
		print(self.s_to_t)

		s = self._problem.s
		t = self._problem.t

		if s is None or t is None:
			return

		upper = []
		edges = []
		lower = []
		last_x = -1

		for i, x in enumerate(self.s_to_t):
			if x < 0:
				upper.append(s[i])
				edges.append(" ")
				lower.append(" ")
			else:
				for j in range(last_x + 1, x):
					upper.append(" ")
					edges.append(" ")
					lower.append(t[j])
				upper.append(s[i])
				edges.append("|")
				lower.append(t[x])
				last_x = x

		print("".join(upper))
		print("".join(edges))
		print("".join(lower))


class Solver:
	def __init__(self, **kwargs):
		self._options = kwargs

		max_len_s = self._options.get("max_len_s")
		max_len_t = self._options.get("max_len_t")

		if max_len_s and max_len_t:
			self._default_solver = pyalign.algorithm.create_solver(
				max_len_s, max_len_t, self._options)
		else:
			self._default_solver = None
		self._last_solver = None

	def solve(self, problem):
		matrix = problem.matrix
		solver = self._default_solver
		if solver is None:
			solver = pyalign.algorithm.create_solver(
				matrix.shape[0], matrix.shape[1], self._options)
		self._last_solver = solver
		solver.solve(matrix)
		return Alignment(problem, solver.alignment)

	@property
	def solution(self):
		"""
		Returns solution details of the last solved problem by this Solver instance.
		"""

		if self._last_solver is None:
			raise RuntimeError("need to solve a problem first to get solution details")
		return self._last_solver.solution

	def display_solution(self):
		pass


class LocalSolver:
	pass

