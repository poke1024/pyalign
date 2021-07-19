#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pyalign.problems
import pyalign.gaps
import pyalign.solve

import string
import random
import time
import collections
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json

from tqdm import tqdm
from pathlib import Path
from typing import Iterator


codomain_names = {
	str(pyalign.solve.Score): "score",
	str(pyalign.solve.Alignment): "alignment",
	str(pyalign.solve.Solution): "solution",
	str(Iterator[pyalign.solve.Alignment]): "all alignments",
	str(Iterator[pyalign.solve.Solution]): "all solutions"
}

class Aligner:
	def prepare(self, a, b):
		raise NotImplementedError()

	def solve(self):
		raise NotImplementedError()

	@property
	def name(self):
		raise NotImplementedError()

	@property
	def num_problems(self):
		raise NotImplementedError()


class PyAlignImplementation(Aligner):
	def __init__(self, codomain=pyalign.solve.Alignment, encoded=False, batch=False):
		self._codomain = codomain
		self._encoded = encoded
		self._batch = batch
		self._solver = None
		self._problem = None
		self._num_problems = None

	def prepare(self, a, b):
		if not self._encoded:
			pf = pyalign.problems.general(
				pyalign.problems.Equality(eq=1, ne=-1))
		else:
			pf = pyalign.problems.alphabetic(
				string.ascii_uppercase[:4],
				pyalign.problems.Equality(eq=1, ne=-1))

		self._solver = pyalign.solve.LocalSolver(
			gap_cost=pyalign.gaps.LinearGapCost(1),
			codomain=self._codomain)

		if not self._batch:
			self._problem = pf.new_problem(a, b)
			self._num_problems = 1
		else:
			self._num_problems = self._solver.batch_size
			self._problem = [
				pf.new_problem(a, b) for _ in range(self._num_problems)]

	def solve(self):
		return self._solver.solve(self._problem)

	@property
	def num_problems(self):
		return self._num_problems

	@property
	def name(self):
		terms = ["pyalign"]
		if self._encoded:
			terms.append("alphabet")
		if self._batch:
			batch_size = self._solver.batch_size
			if batch_size == 8:
				terms.append("AVX2")
			else:
				raise RuntimeError(f"unexpected batch_size {batch_size}")
		return " +".join(terms)


class PurePythonImplementation(Aligner):
	def __init__(self, backtrace=True):
		self._aEncoded = None
		self._bEncoded = None
		self._aligner = None
		self._v = None
		self._backtrace = backtrace

	def prepare(self, a, b):
		# see https://github.com/eseraygun/python-alignment

		from alignment.sequence import Sequence
		from alignment.vocabulary import Vocabulary
		from alignment.sequencealigner import SimpleScoring, LocalSequenceAligner

		# Create sequences to be aligned.
		a = Sequence(a)
		b = Sequence(b)

		# Create a vocabulary and encode the sequences.
		v = Vocabulary()
		self._v = v
		self._aEncoded = v.encodeSequence(a)
		self._bEncoded = v.encodeSequence(b)

		# Create a scoring and align the sequences using global aligner.
		scoring = SimpleScoring(1, -1)
		self._aligner = LocalSequenceAligner(scoring, -1)

	def solve(self):
		# returns: score, encodeds
		return self._aligner.align(
			self._aEncoded, self._bEncoded, backtrace=self._backtrace)

	def print(self, score, encodeds):
		# Iterate over optimal alignments and print them.
		for encoded in encodeds:
			alignment = self._v.decodeSequenceAlignment(encoded)

	@property
	def name(self):
		return "pure python"

	@property
	def num_problems(self):
		return 1


def random_seq(l=20, n=4):
	x = []
	for _ in range(l):
		x.append(random.choice(string.ascii_uppercase[:n]))
	return ''.join(x)


def benchmark(num_runs=1000, seq_len=20):
	random.seed(24242)

	a = random_seq(seq_len)  # "DDAAABDBADDBADBDBABB"
	b = random_seq(seq_len)  # "AADCCCCACBADCDACDBCA"

	codomains = [
		pyalign.solve.Score,
		pyalign.solve.Alignment,
		pyalign.solve.Solution,
		Iterator[pyalign.solve.Alignment],
		Iterator[pyalign.solve.Solution]
	]

	def aligners():
		yield str(pyalign.solve.Score), PurePythonImplementation(backtrace=False)
		yield str(pyalign.solve.Alignment), PurePythonImplementation(backtrace=True)
		for batch in (False, True):
			for encoded in (False, True):
				for codomain in codomains:
					yield str(codomain), PyAlignImplementation(
						codomain, encoded=encoded, batch=batch)

	path = Path(f"runtimes_{seq_len}.json")
	if path.exists():
		with open(path, "r") as f:
			runtimes_μs = json.loads(f.read())
	else:
		runtimes_μs = collections.defaultdict(dict)
		μs_to_ns = 1000

		for codomain, aligner in tqdm(list(aligners())):
			aligner.prepare(a, b)
			t0 = time.perf_counter_ns()
			for _ in range(num_runs):
				aligner.solve()
			t1 = time.perf_counter_ns()
			runtimes_μs[codomain][aligner.name] = (t1 - t0) // (μs_to_ns * num_runs * aligner.num_problems)

		with open(path, "w") as f:
			f.write(json.dumps(runtimes_μs))

	def variant_sort_order(s):
		if s == "pure python":
			return 1
		else:
			return 2 + len(s)

	variants = set()
	for codomain, times in runtimes_μs.items():
		for k in times.keys():
			variants.add(k)
	variants = sorted(list(variants), key=variant_sort_order)

	y = dict()
	for variant in variants:
		y[variant] = []
		for codomain in codomains:
			y[variant].append(runtimes_μs[str(codomain)].get(variant, np.nan))

	x = np.arange(0, len(codomains) * len(variants), len(variants))  # the label locations
	width = 0.9
	x_c = x - ((len(variants) - 1) / 2) * width

	cmap = matplotlib.cm.get_cmap('Set3')
	norm = matplotlib.colors.Normalize(vmin=0, vmax=len(variants) - 1)

	fig, ax = plt.subplots()
	for i, variant in enumerate(variants):
		ax.bar(x_c + width * i, y[variant], width, label=variant, color=cmap(norm(i)))

	ax.set_ylabel('time in μs')
	ax.set_yscale('log')

	from matplotlib.ticker import StrMethodFormatter, FixedLocator
	ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
	ax.yaxis.set_minor_formatter(StrMethodFormatter('{x:.0f}'))
	ax.yaxis.set_minor_locator(FixedLocator([15, 25, 50, 100, 150, 250, 500]))

	ax.set_xticks(x)
	ax.set_xticklabels([codomain_names[str(s)] for s in codomains])

	ax.legend(loc="upper right")
	plt.xticks(rotation=45)
	plt.grid(which="both", alpha=0.25)

	plt.title(f"local alignment with linear gap cost\nsequence length = {seq_len}")

	fig.tight_layout()

	plt.savefig(f'benchmark_{seq_len}.svg', bbox_inches='tight')


if __name__ == "__main__":
	benchmark(seq_len=10)
	#benchmark(seq_len=20)
	#benchmark(seq_len=50)
