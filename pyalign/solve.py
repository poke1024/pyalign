import numpy as np
import time
import contextlib
import importlib
import typing
import os
import logging
import pyalign.io.alignment

from cached_property import cached_property
from functools import lru_cache
from pathlib import Path
from collections.abc import Sequence

from .gaps import GapCost, ConstantGapCost
from .problems import ProblemBag, Form


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

	missing_module_names = []

	for name, check in candidates:
		module_name = f"pyalign.algorithm.{name}"
		if importlib.util.find_spec(module_name) is not None:
			if check():
				logging.info(f"running in {name} mode.")
				return importlib.import_module(module_name + ".algorithm")
		else:
			missing_module_names.append(module_name)

	raise RuntimeError(f"failed to find suitable c++ core in {missing_module_names}")


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
		if self._solution.alignment is None:
			return None
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
	def solver(self):
		return self._solver

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
		formatter = pyalign.io.alignment.Formatter(self)
		text = formatter.text
		if text is not None:
			print(text)

	def _repr_html_(self):
		formatter = pyalign.io.alignment.Formatter(self)
		return formatter.html


class Score:
	pass


def next_power_of_2(x):
	return 1 if x == 0 else 2 ** (int(x) - 1).bit_length()


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
			options['direction'] = algorithm.Direction.__members__[direction.upper()]
			options['batch'] = batch
			options['return_dup'] = options.get('return_dup', False)

			parsed_options = algorithm.create_options(options)

			solver = algorithm.create_solver(
				self._max_lim_s, self._max_lim_t, parsed_options)

			self._solvers[batch] = solver

		return solver


class Codomain:
	_1 = set([Score, Alignment, Solution])
	_n = dict(
		[(typing.Iterator[x], (x, "all", "iterator")) for x in [Score, Alignment, Solution]] +
		[(typing.List[x], (x, "all", "list")) for x in [Score, Alignment, Solution]]
	)

	def __init__(self, type):
		self._type = type

		if type in Codomain._1:
			self._base_type = type
			self._count = "one"
			container_type = None
			self._optimal = True
		else:
			base, self._count, container_type = Codomain._n.get(type)
			if base is None:
				raise ValueError(f"illegal codomain type '{type}'")
			self._base_type = base
			self._optimal = True

		self._key = (self.detail, self.count, container_type) + (("optimal",) if self._optimal else tuple())

	@property
	def type(self):
		return self._type

	def __str__(self):
		return str(self.type)

	@property
	def detail(self):
		return algorithm.Detail.__members__[self._base_type.__name__.upper()]

	@property
	def count(self):
		return algorithm.Count.__members__[self._count.upper()]

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


def make_list_factory(iterator):
	def to_list(*args, **kwargs):
		return list(iterator(*args, **kwargs))

	return to_list


def solver_variants(prefix):
	data = {
		(algorithm.Detail.SCORE, algorithm.Count.ONE, None, "optimal"): (
			"for_score", lambda _1, _2, x: x),

		(algorithm.Detail.ALIGNMENT, algorithm.Count.ONE, None, "optimal"): ("for_alignment", Alignment),
		(algorithm.Detail.ALIGNMENT, algorithm.Count.ALL, "iterator", "optimal"): (
			"for_alignment_iterator", AlignmentIterator),
		(algorithm.Detail.ALIGNMENT, algorithm.Count.ALL, "list", "optimal"): (
			"for_alignment_iterator", make_list_factory(AlignmentIterator)),

		(algorithm.Detail.SOLUTION, algorithm.Count.ONE, None, "optimal"): ("for_solution", Solution),
		(algorithm.Detail.SOLUTION, algorithm.Count.ALL, "iterator", "optimal"): (
			"for_solution_iterator", SolutionIterator),
		(algorithm.Detail.SOLUTION, algorithm.Count.ALL, "list", "optimal"): (
			"for_solution_iterator", make_list_factory(SolutionIterator))
	}

	return dict((k, (f"{prefix}_{v1}", v2)) for k, (v1, v2) in data.items())


class MatrixForm:
	_solvers = solver_variants("solve")

	def __init__(self, solver, codomain, bag):
		self._solver = solver
		batch_size = solver.batch_size
		max_shape = bag.max_shape
		self._matrix = np.zeros((max_shape[0], max_shape[1], batch_size), dtype=bag.dtype)
		self._len = np.zeros((2, batch_size), dtype=np.uint16)

		variant = MatrixForm._solvers.get(codomain.key)
		if variant is None:
			raise ValueError(f"codomain {codomain} is currently not supported")
		self._solve = getattr(solver, variant[0])
		self._construct = variant[1]

	def prepare(self, batch):
		shape = batch.shape
		matrix = self._matrix[:shape[0], :shape[1], :]
		p_len = self._len

		for k, p in enumerate(batch.problems):
			p.build_matrix(matrix[:, :, k])

			p_shape = p.shape
			p_len[0, k] = p_shape[0]
			p_len[1, k] = p_shape[1]

	def solve(self, batch):
		shape = batch.shape
		matrix = self._matrix[:shape[0], :shape[1], :]

		r = self._solve(matrix, self._len)
		c = self._construct

		return [
			c(problem, self._solver, x)
			for problem, x in zip(batch.problems, r)]


class AbstractIndexedMatrixForm:
	def prepare(self, batch):
		shape = batch.shape
		a = self._a
		b = self._b
		p_len = self._len

		for k, p in enumerate(batch.problems):
			p.build_index_sequences(
				a[k, :shape[0]],
				b[k, :shape[1]])

			p_shape = p.shape
			p_len[0, k] = p_shape[0]
			p_len[1, k] = p_shape[1]


class IndexedMatrixForm(AbstractIndexedMatrixForm):
	_solvers = solver_variants("solve_indexed")

	def __init__(self, solver, codomain, bag):
		self._solver = solver
		batch_size = solver.batch_size
		max_shape = bag.max_shape

		self._a = np.zeros((batch_size, max_shape[0]), dtype=np.uint32)
		self._b = np.zeros((batch_size, max_shape[1]), dtype=np.uint32)
		self._len = np.zeros((2, batch_size), dtype=np.uint16)

		self._sim = bag.problems[0].similarity_lookup_table()
		if not all(p.similarity_lookup_table() is self._sim for p in bag.problems):
			raise ValueError("similarity table must be identical for all problems in a batch")

		variant = IndexedMatrixForm._solvers.get(codomain.key)
		if variant is None:
			raise ValueError(f"codomain {codomain} is currently not supported")
		self._solve = getattr(solver, variant[0])
		self._construct = variant[1]

	def solve(self, batch):
		shape = batch.shape

		r = self._solve(
			self._a[:, :shape[0]],
			self._b[:, :shape[1]],
			self._sim, self._len)
		c = self._construct

		return [
			c(problem, self._solver, x)
			for problem, x in zip(batch.problems, r)]


class BinaryMatrixForm(AbstractIndexedMatrixForm):
	_solvers = solver_variants("solve_binary")

	def __init__(self, solver, codomain, bag):
		self._solver = solver
		batch_size = solver.batch_size
		max_shape = bag.max_shape

		self._a = np.zeros((batch_size, max_shape[0]), dtype=np.uint32)
		self._b = np.zeros((batch_size, max_shape[1]), dtype=np.uint32)
		self._len = np.zeros((2, batch_size), dtype=np.uint16)

		self._sim = bag.problems[0].binary_similarity_values()
		if not all(p.binary_similarity_values() is self._sim for p in bag.problems):
			raise ValueError("similarity must be identical for all problems in a batch")

		variant = BinaryMatrixForm._solvers.get(codomain.key)
		if variant is None:
			raise ValueError(f"codomain {codomain} is currently not supported")
		self._solve = getattr(solver, variant[0])
		self._construct = variant[1]

	def solve(self, batch):
		shape = batch.shape

		r = self._solve(
			self._a[:, :shape[0]],
			self._b[:, :shape[1]],
			*self._sim,  # eq, ne
			self._len)
		c = self._construct

		return [
			c(problem, self._solver, x)
			for problem, x in zip(batch.problems, r)]


class Solver:
	"""
	A solver that obtains solutions to alignment problems.
	"""

	_matrix_c = {
		Form.MATRIX_FORM: MatrixForm,
		Form.INDEXED_MATRIX_FORM: IndexedMatrixForm,
		Form.BINARY_MATRIX_FORM: BinaryMatrixForm
	}

	def __init__(
		self, gap_cost: GapCost = None, codomain=Solution, **kwargs):

		if codomain is None:
			codomain_obj = Codomain(Solution)
		else:
			codomain_obj = Codomain(codomain)
		if gap_cost is None:
			gap_cost = ConstantGapCost(0)

		self._codomain = codomain_obj

		self._options = dict(
			gap_cost=gap_cost,
			codomain=self._codomain,
			**kwargs)

		self._cache = SolverCache(self._options)
		self._timings = NoTimings()

		max_len_s = self._options.get("max_len_s")
		max_len_t = self._options.get("max_len_t")

		if max_len_s and max_len_t:
			self._cache.ensure(max_len_s, max_len_t)

	@property
	def gap_cost(self):
		return self._options["gap_cost"]

	def to_codomain(self, codomain, return_dup=False):
		kwargs = self._options.copy()
		kwargs['codomain'] = codomain
		kwargs['return_dup'] = return_dup
		return Solver(**kwargs)

	@property
	def codomain(self):
		"""the solver's codomain"""
		return self._codomain.type

	@property
	def batch_size(self):
		"""
		the solver's optimal batch size, i.e. the number of alignment pairs
		that can get processed in a single SIMD call on this machine.
		"""
		return self._cache.get(
			1, 1, direction=algorithm.Direction.MAXIMIZE, batch=True).batch_size

	def timings(self):
		return Timings(self)

	def _solve_bag(self, bag):
		max_shape = bag.max_shape
		solver = self._cache.get(
			max_shape[0], max_shape[1],
			direction=bag.direction,
			batch=len(bag) > 1)
		batch_size = solver.batch_size
		form = bag.form
		problems = bag.problems

		matrix_c = Solver._matrix_c.get(form)
		if matrix_c is None:
			raise ValueError(form)
		form_solver = matrix_c(
			solver, self._codomain, bag)

		result = [None] * len(problems)

		for batch in bag.batches(batch_size):
			with self._timings.measure("prepare"):
				form_solver.prepare(batch)

			with self._timings.measure("solve"):
				for i, r in zip(batch.indices, form_solver.solve(batch)):
					result[i] = r

		return result

	def _solve_problem(self, problem):
		batch = ProblemBag([problem])
		result = self._solve_bag(batch)
		return result[0]

	def solve(self, x):
		if isinstance(x, Sequence):
			return self._solve_bag(ProblemBag(list(x)))
		else:
			return self._solve_problem(x)


class LocalSolver(Solver):
	"""
	A solver that obtains optimal local alignments. For linear gap costs,
	this conforms to the results obtained with the Smith-Waterman algorithm.
	"""

	def __init__(self, gap_cost: GapCost = None, **kwargs):
		super().__init__(
			solver=algorithm.Type.ALIGNMENT,
			locality=algorithm.Locality.LOCAL,
			gap_cost=gap_cost, **kwargs)


class GlobalSolver(Solver):
	"""
	A solver that obtains optimal global alignments. For linear gap costs,
	this conforms to the results obtained with the Needleman-Wunsch algorithm.
	"""

	def __init__(self, gap_cost: GapCost = None, **kwargs):
		super().__init__(
			solver=algorithm.Type.ALIGNMENT,
			locality=algorithm.Locality.GLOBAL,
			gap_cost=gap_cost, **kwargs)


class SemiglobalSolver(Solver):
	"""
	A solver that obtains optimal semiglobal alignments.
	"""

	def __init__(self, gap_cost: GapCost = None, **kwargs):
		super().__init__(
			solver=algorithm.Type.ALIGNMENT,
			locality=algorithm.Locality.SEMIGLOBAL,
			gap_cost=gap_cost, **kwargs)


class ElasticSolver(Solver):
	"""
	A solver that uses dynamic time warping (DTW) to find the optimal solution of
	an elastic alignment problem.
	"""

	def __init__(self, **kwargs):
		super().__init__(solver=algorithm.Type.DTW, **kwargs)
