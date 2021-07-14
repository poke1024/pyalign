import numpy as np
import time
import contextlib
import importlib
import re
import os
import logging

from cached_property import cached_property
from functools import lru_cache
from pathlib import Path

from .gaps import GapCost, ConstantGapCost
from .problems import ProblemBatch, Form


def has_avx2():
	import cpufeature
	return cpufeature.CPUFeature["AVX2"]


def import_algorithm():
	if os.environ.get('PYALIGN_PDOC') is not None:
		return None

	candidates = (
		('native', lambda: True),
		('avx2', has_avx2),
		('generic', lambda: True)
	)

	for name, check in candidates:
		module_name = f"pyalign.algorithm.{name}"
		if importlib.util.find_spec(module_name) is not None:
			if check():
				logging.info(f"running in {name} mode.")
				return importlib.import_module(module_name + ".algorithm")

	raise RuntimeError("no suitable c++ core found")


algorithm = import_algorithm()


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

	@lru_cache(maxsize=2)
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
		from pyalign.io.plot import TracebackPlotFactory
		f = TracebackPlotFactory(
			self._solution, self._problem, layer=layer)
		bokeh.io.show(f.create())

	def _ipython_display_(self):
		self.display()

	def export_image(self, path):
		import bokeh.io
		from pyalign.io.plot import TracebackPlotFactory
		f = TracebackPlotFactory(self._solution, self._problem)
		path = Path(path)
		if path.suffix == ".svg":
			bokeh.io.export_svg(f.create(), filename=path)
		else:
			bokeh.io.export_png(f.create(), filename=path)


class Alignment:
	"""
	A specific alignment that is part of a solution.
	"""

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

		for j in range(last_x + 1, len(t)):
			if not is_elastic:
				upper.append(" ")
				edges.append(" ")
				lower.append(t[j])
			else:
				upper.append(s[-1])
				edges.append(" ")
				lower.append(t[j])

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

	def get(self, len_s, len_t, direction, batch):
		self.ensure(len_s, len_t)
		key = (direction, batch)
		solver = self._solvers.get(key)
		if solver is None:
			options = self._options.copy()
			options['direction'] = direction
			options['batch'] = batch
			solver = algorithm.create_solver(
				self._max_lim_s, self._max_lim_t, options)
			self._solvers[batch] = solver
		return solver


class Goal:
	details = set(["score", "alignment", "solution"])

	def __init__(self, detail, qualifiers):
		"""
		Args:
			detail (str): one of "score", "alignment", "solution"
			qualifiers (List[str]): one of "one" or "all", optionally "optimal"
		"""

		if detail not in Goal.details:
			raise ValueError(f"illegal detail specification '{detail}'")

		count = "one"
		optimal = False

		for q in qualifiers:
			if q in ("one", "all"):
				count = q
			elif q == "optimal":
				optimal = True
			else:
				raise ValueError(f"illegal qualifier '{q}'")

		if count == "one":
			optimal = True

		self._detail = detail
		self._count = count
		self._optimal = optimal

		self._key = (self._detail, self._count,) + (("optimal",) if self._optimal else tuple())

	@staticmethod
	def from_str(s):
		# e.g. "solution[all, optimal]"
		if "[" in s:
			m = re.match(r"(?P<head>\w+)\[(?P<qual>[\w\s,]+)\]", s)
			if m is None:
				raise ValueError(f"illegal goal specification '{s}'")
			qualifiers = [re.sub(r"\s", "", q) for q in m.group("qual").split(",")]
			return Goal(m.group("head"), qualifiers)
		else:
			return Goal(s, ["one", "optimal"])

	def __str__(self):
		return f"{self._detail}[{', '.join(self._key[1:])}]"

	@property
	def detail(self):
		return self._detail

	@property
	def count(self):
		return self._count

	@property
	def key(self):
		return self._key


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


class Iterator:
	def __init__(self, problem, solver, iterator):
		self._problem = problem
		self._solver = solver
		self._iterator = iterator

	def __iter__(self):
		while True:
			x = self._iterator.next()
			if x is None:
				break
			yield self._element_class(self._problem, self._solver, x)


class AlignmentIterator(Iterator):
	_element_class = Alignment


class SolutionIterator(Iterator):
	_element_class = Solution


def solver_variants(prefix):
	data = {
		("score", "one", "optimal"): ("for_score", lambda _1, _2, x: x),
		#("score", "all"): ("for_score", lambda _1, _2, x: x),
		("alignment", "one", "optimal"): ("for_alignment", Alignment),
		("alignment", "all", "optimal"): ("for_alignment_iterator", AlignmentIterator),
		("solution", "one", "optimal"): ("for_solution", Solution),
		("solution", "all", "optimal"): ("for_solution_iterator", SolutionIterator)
	}
	return dict((k, (f"{prefix}_{v1}", v2)) for k, (v1, v2) in data.items())


class MatrixForm:
	_solvers = solver_variants("solve")

	def __init__(self, solver, goal, batch):
		self._solver = solver
		batch_size = solver.batch_size
		shape = batch.shape
		self._matrix = np.empty((shape[0], shape[1], batch_size), dtype=batch.dtype)

		self._len = np.empty((2, batch_size), dtype=np.uint16)
		self._len[0, :].fill(shape[0])
		self._len[1, :].fill(shape[1])

		variant = MatrixForm._solvers.get(goal.key)
		if variant is None:
			raise ValueError(f"{goal.detail}[{', '.join(goal.key[1:])}] is currently not supported")
		self._solve = getattr(solver, variant[0])
		self._construct = variant[1]

	def prepare(self, problems):
		matrix = self._matrix

		for k, p in enumerate(problems):
			p.build_matrix(matrix[:, :, k])

	def solve(self, problems):
		r = self._solve(self._matrix, self._len)
		return [
			self._construct(problem, self._solver, x)
			for problem, x in zip(problems, r)]


class IndexedMatrixForm:
	_solvers = solver_variants("solve_indexed")

	def __init__(self, solver, goal, batch):
		self._solver = solver
		batch_size = solver.batch_size
		shape = batch.shape

		self._a = np.empty((batch_size, shape[0]), dtype=np.uint32)
		self._b = np.empty((batch_size, shape[1]), dtype=np.uint32)

		self._len = np.empty((2, batch_size), dtype=np.uint16)
		self._len[0, :].fill(shape[0])
		self._len[1, :].fill(shape[1])

		self._sim = batch.problems[0].similarity_lookup_table()
		if not all(p.similarity_lookup_table() is self._sim for p in batch.problems):
			raise ValueError("similarity table must be identical for all problems in a batch")

		variant = IndexedMatrixForm._solvers.get(goal.key)
		if variant is None:
			raise ValueError(f"{goal.detail}[{', '.join(goal.key[1:])}] is currently not supported")
		self._solve = getattr(solver, variant[0])
		self._construct = variant[1]

	def prepare(self, problems):
		a = self._a
		b = self._b
		for k, p in enumerate(problems):
			p.build_index_sequences(a[k, :], b[k, :])

	def solve(self, problems):
		r = self._solve(self._a, self._b, self._sim, self._len)
		return [
			self._construct(problem, self._solver, x)
			for problem, x in zip(problems, r)]


class Solver:
	"""
	A solver that obtains solutions to alignment problems.
	"""

	def __init__(
		self, gap_cost: GapCost = None, generate="alignment", **kwargs):

		if generate is None:
			goal = Goal("alignment", False)
		elif isinstance(generate, str):
			goal = Goal.from_str(generate)
		else:
			raise ValueError(generate)

		if gap_cost is None:
			gap_cost = ConstantGapCost(0)

		self._goal = goal

		self._options = dict(
			gap_cost=gap_cost,
			goal=goal,
			**kwargs)

		self._cache = SolverCache(self._options)
		self._timings = NoTimings()

		max_len_s = self._options.get("max_len_s")
		max_len_t = self._options.get("max_len_t")

		if max_len_s and max_len_t:
			self._cache.ensure(max_len_s, max_len_t)

	@property
	def goal(self):
		"""the solver's goal"""
		return self._goal

	@property
	def batch_size(self):
		"""
		the solver's optimal batch size, i.e. the number of alignment pairs
		that can get processed in a single SIMD call on this machine.
		"""
		return self._cache.get(1, 1, direction='maximize', batch=True).batch_size

	def timings(self):
		return Timings(self)

	def solve_batch(self, batch):
		is_batch = len(batch) > 1
		shape = batch.shape
		solver = self._cache.get(
			shape[0], shape[1], direction=batch.direction, batch=is_batch)
		batch_size = solver.batch_size
		form = batch.form

		result = []

		if form == Form.MATRIX_FORM:
			form_solver = MatrixForm(solver, self._goal, batch)
		elif form == Form.INDEXED_MATRIX_FORM:
			form_solver = IndexedMatrixForm(solver, self._goal, batch)
		else:
			raise ValueError(form)

		for i in range(0, len(batch), batch_size):
			problems_chunk = batch.problems[i:i + batch_size]

			with self._timings.measure("prepare"):
				form_solver.prepare(problems_chunk)

			with self._timings.measure("solve"):
				result.extend(form_solver.solve(problems_chunk))

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
	"""
	A solver that obtains optimal local alignments. For linear gap costs,
	this conforms to the results obtained with the Smith-Waterman algorithm.
	"""

	def __init__(self, gap_cost: GapCost = None, **kwargs):
		super().__init__(solver="alignment", locality="local", gap_cost=gap_cost, **kwargs)


class GlobalSolver(Solver):
	"""
	A solver that obtains optimal global alignments. For linear gap costs,
	this conforms to the results obtained with the Needleman-Wunsch algorithm.
	"""

	def __init__(self, gap_cost: GapCost = None, **kwargs):
		super().__init__(solver="alignment", locality="global", gap_cost=gap_cost, **kwargs)


class SemiglobalSolver(Solver):
	"""
	A solver that obtains optimal semiglobal alignments.
	"""

	def __init__(self, gap_cost: GapCost = None, **kwargs):
		super().__init__(solver="alignment", locality="semiglobal", gap_cost=gap_cost, **kwargs)


class ElasticSolver(Solver):
	"""
	A solver that uses dynamic time warping (DTW) to find the optimal solution of
	an elastic alignment problem.
	"""

	def __init__(self, **kwargs):
		super().__init__(solver="dtw", **kwargs)
