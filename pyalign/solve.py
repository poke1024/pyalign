import pyalign.algorithm
import numpy as np
import time
import contextlib

from cached_property import cached_property
from functools import lru_cache
from pathlib import Path
from .gaps import GapCost, ConstantGapCost


class Problem:
	def __init__(self, shape, s=None, t=None, direction="maximize", dtype=np.float32):
		self._shape = tuple(shape)
		if s is not None and shape[0] != len(s):
			raise ValueError(f"sequence s [{len(s)}] does not match matrix shape {shape}")
		if t is not None and shape[1] != len(t):
			raise ValueError(f"sequence t [{len(t)}] does not match matrix shape {shape}")
		self._s = s
		self._t = t
		self._direction = direction
		self._dtype = dtype

	@property
	def shape(self):
		return self._shape

	@property
	def direction(self):
		return self._direction

	@property
	def dtype(self):
		return self._dtype

	def build_matrix(self, out):
		"""
		Returns a matrix M that describes an alignment problem for two sequences \( s \) and \( t \).
		\( M_{i, j} \) contains the similarity between \( s_i \) and \( t_j \).
		"""
		raise NotImplementedError()

	@property
	def s(self):
		return self._s

	@property
	def t(self):
		return self._t


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

	@cached_property
	def alignment(self):
		return Alignment(
			self._problem,
			self._solver,
			self._solution.alignment)

	@property
	def shape(self):
		return self._solution.values.shape

	@cached_property
	def values(self):
		return self._solution.values

	@lru_cache
	def traceback(self, form="matrix"):
		if form == "matrix":
			return self._solution.traceback_as_matrix
		elif form == "edges":
			return self._solution.traceback_as_edges
		else:
			return ValueError(form)

	@cached_property
	def path(self):
		return self._solution.path

	@property
	def complexity(self):
		return self._solution.complexity

	def display(self, layer=0):
		import bokeh.io
		from .utils.plot import TracebackPlotFactory
		f = TracebackPlotFactory(
			self._solution, self._problem, layer=layer)
		bokeh.io.show(f.create())

	def _ipython_display_(self):
		self.display()

	def export_image(self, path):
		import bokeh.io
		from .utils.plot import TracebackPlotFactory
		f = TracebackPlotFactory(self._solution, self._problem)
		path = Path(path)
		if path.suffix == ".svg":
			bokeh.io.export_svg(f.create(), filename=path)
		else:
			bokeh.io.export_png(f.create(), filename=path)


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


def next_power_of_2(x):
	return 1 if x == 0 else 2 ** (x - 1).bit_length()


class SolverCache:
	def __init__(self, options):
		self._options = options
		self._max_lim_s = 0
		self._max_lim_t = 0
		self._solvers = {}

	def ensure(self, len_s, len_t):
		lim_s = max(self._max_lim_s, next_power_of_2(len_s))
		lim_t = max(self._max_lim_t, next_power_of_2(len_t))

		if lim_s > self._max_lim_s or lim_t > self._max_lim_t:
			self._max_lim_s = lim_s
			self._max_lim_t = lim_t
			self._solvers = {}

	def get(self, len_s, len_t, batch):
		self.ensure(len_s, len_t)
		solver = self._solvers.get(batch)
		if solver is None:
			options = self._options.copy()
			options['batch'] = batch
			solver = pyalign.algorithm.create_solver(
				self._max_lim_s, self._max_lim_t, options)
			self._solvers[batch] = solver
		return solver


class Goal:
	details = set(["score", "alignment", "solution"])

	def __init__(self, detail, count):
		"""
		Args:
			detail (str): one of "score", "alignment", "solution"
			count (str): how many optimal solutions? either "one" or "all"
		"""

		if detail not in Goal.details:
			raise ValueError(detail)
		if count not in ("one", "all"):
			raise ValueError(count)

		self._detail = detail
		self._count = count

	@staticmethod
	def from_str(s):
		if s in Goal.details:
			return Goal(s, "one")
		if s in [x + "s" for x in Goal.details]:
			return Goal(s[:-1], "all")
		raise ValueError(s)

	@property
	def detail(self):
		return self._detail

	@property
	def count(self):
		return self._count


class NoTimings:
	@contextlib.contextmanager
	def measure(self, name):
		yield


class Timings:
	def __init__(self, solver):
		self._solver = solver
		self._timings = dict()

	def __enter__(self):
		self._solver._timings = self
		return self

	def __exit__(self, type, value, traceback):
		self._solver._timings = NoTimings()

	@contextlib.contextmanager
	def measure(self, name):
		t0 = time.perf_counter_ns()
		yield
		t1 = time.perf_counter_ns()
		self._timings[name] = self._timings.get(name, 0) + (t1 - t0)

	def get(self):
		return self._timings

	def _ipython_display_(self):
		for k, t in self._timings.items():
			print(f"{k}: {t / 1000:.1f} µs")


def chunks(items, n):
	for i in range(0, len(items), n):
		yield items[i:i + n]


class Solver:
	def __init__(
		self, gap_cost: GapCost = None, direction="maximize", generate="alignment", **kwargs):

		if generate is None:
			goal = Goal("alignment", False)
		elif isinstance(generate, str):
			goal = Goal.from_str(generate)
		else:
			raise ValueError(generate)

		if gap_cost is None:
			gap_cost = ConstantGapCost(0)

		self._direction = direction
		self._goal = goal

		self._options = dict(
			gap_cost=gap_cost,
			direction=direction,
			goal=goal,
			**kwargs)

		self._cache = SolverCache(self._options)
		self._timings = NoTimings()

		max_len_s = self._options.get("max_len_s")
		max_len_t = self._options.get("max_len_t")

		if max_len_s and max_len_t:
			self._cache.ensure(max_len_s, max_len_t)

	@property
	def batch_size(self):
		return self._cache.get(1, 1, batch=True).batch_size

	def timings(self):
		return Timings(self)

	def solve_batch(self, batch):
		if batch.direction != self._direction:
			raise ValueError(
				f"problem given is '{batch.direction}', "
				f"but solver is configured to '{self._direction}'")

		is_batch = len(batch) > 1
		shape = batch.shape
		solver = self._cache.get(shape[0], shape[1], batch=is_batch)
		batch_size = solver.batch_size

		detail = self._goal.detail
		result = []

		matrix = np.empty((shape[0], shape[1], batch_size), dtype=batch.dtype)

		for i in range(0, len(batch), batch_size):
			problems_chunk = batch.problems[i:i + batch_size]

			with self._timings.measure("build_matrix"):
				for i, p in enumerate(problems_chunk):
					p.build_matrix(matrix[:, :, i])

			with self._timings.measure("solve"):
				if detail == "score":
					result.extend(solver.solve_for_score(matrix)[:len(problems_chunk)])
				elif detail == "alignment":
					alignments = solver.solve_for_alignment(matrix)
					result.extend([
						Alignment(problem, solver, alignment)
						for problem, alignment in zip(problems_chunk, alignments)])
				elif detail == "solution":
					solutions = solver.solve_for_solution(matrix)
					result.extend([
						Solution(problem, solver, solution)
						for problem, solution in zip(problems_chunk, solutions)])
				else:
					return ValueError(detail)

		return result

	def solve_problem(self, problem):
		batch = ProblemBatch([problem])
		result = self.solve_batch(batch)
		return result[0]

	def solve(self, x):
		if isinstance(x, ProblemBatch):
			return self.solve_batch(x)
		else:
			return self.solve_problem(x)


class LocalSolver(Solver):
	def __init__(self, gap_cost: GapCost = None, **kwargs):
		super().__init__(solver="alignment", locality="local", gap_cost=gap_cost, **kwargs)


class GlobalSolver(Solver):
	def __init__(self, gap_cost: GapCost = None, **kwargs):
		super().__init__(solver="alignment", locality="global", gap_cost=gap_cost, **kwargs)


class SemiglobalSolver(Solver):
	def __init__(self, gap_cost: GapCost = None, **kwargs):
		super().__init__(solver="alignment", locality="semiglobal", gap_cost=gap_cost, **kwargs)


class ElasticSolver(Solver):
	def __init__(self, **kwargs):
		super().__init__(solver="dtw", **kwargs)
