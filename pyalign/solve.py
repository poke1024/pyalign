import pyalign.algorithm
import numpy as np

from cached_property import cached_property
from .gaps import GapCost


class Problem:
	def __init__(self, matrix, s=None, t=None, mode="similarity"):
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
		Returns a matrix M that describes an alignment problem for two sequences \( s \) and \( t \).
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


class Solution:
	def __init__(self, problem, solver, solution):
		self._problem = problem
		self._solver = solver
		self._solution = solution

	@property
	def problem(self):
		return self._problem

	@property
	def score(self):
		return self._solution.score

	@property
	def alignment(self):
		return self._solution.alignment

	@cached_property
	def values(self):
		return self._solution.values

	@cached_property
	def traceback(self):
		return self._solution.traceback

	@cached_property
	def path(self):
		return self._solution.path

	@property
	def complexity(self):
		return self._solution.complexity

	def _ipython_display_(self):
		from pyalign.plot import TracebackPlotter
		plotter = TracebackPlotter(self)
		plotter.plot()


class Alignment:
	def __init__(self, problem, solver, alignment):
		self._problem = problem
		self._solver = solver
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

	@cached_property
	def edges(self):
		s_to_t = self.s_to_t
		i = np.nonzero(s_to_t >= 0)[0]
		return np.column_stack([i, s_to_t[i]])

	def print(self):
		s = self._problem.s
		t = self._problem.t

		if s is None or t is None:
			return

		upper = []
		edges = []
		lower = []
		last_x = -1

		is_elastic = self._solver.options["solver"] == "dtw"

		for i, x in enumerate(self.s_to_t):
			if x < 0:
				if not is_elastic:
					upper.append(s[i])
					edges.append(" ")
					lower.append(" ")
				else:
					upper.append(s[i])
					edges.append(" ")
					lower.append(t[max(last_x, 0)])
			else:
				for j in range(last_x + 1, x):
					if not is_elastic:
						upper.append(" ")
						edges.append(" ")
						lower.append(t[j])
					else:
						upper.append(s[i])
						edges.append(" ")
						lower.append(t[j])
				upper.append(s[i])
				edges.append("|")
				lower.append(t[x])
				last_x = x

		print("".join(upper))
		print("".join(edges))
		print("".join(lower))

	def _ipython_display_(self):
		self.print()


class Solver:
	def __init__(self, gap_cost: GapCost = None, **kwargs):
		self._options = dict(gap_cost=gap_cost, **kwargs)

		max_len_s = self._options.get("max_len_s")
		max_len_t = self._options.get("max_len_t")

		if max_len_s and max_len_t:
			self._default_solver = pyalign.algorithm.create_solver(
				max_len_s, max_len_t, self._options)
		else:
			self._default_solver = None
		self._last_solver = None
		self._last_problem = None

	def solve(self, problem, result="alignment"):
		matrix = problem.matrix

		solver = self._default_solver
		if solver is None:
			solver = pyalign.algorithm.create_solver(
				matrix.shape[0], matrix.shape[1], self._options)

		if result == "score":
			return solver.solve_for_score(matrix)
		elif result == "alignment":
			return Alignment(problem, solver, solver.solve_for_alignment(matrix))
		elif result == "solution":
			return Solution(problem, solver, solver.solve_for_solution(matrix))
		else:
			return ValueError(result)


class LocalSolver(Solver):
	def __init__(self, gap_cost: GapCost = None, zero: float = 0, **kwargs):
		super().__init__(solver="alignment", locality="local", gap_cost=gap_cost, zero=zero, **kwargs)


class GlobalSolver(Solver):
	def __init__(self, gap_cost: GapCost = None, **kwargs):
		super().__init__(solver="alignment", locality="global", gap_cost=gap_cost, **kwargs)


class SemiglobalSolver(Solver):
	def __init__(self, gap_cost: GapCost = None, **kwargs):
		super().__init__(solver="alignment", locality="semiglobal", gap_cost=gap_cost, **kwargs)


class ElasticSolver(Solver):
	def __init__(self, **kwargs):
		super().__init__(solver="dtw", **kwargs)
