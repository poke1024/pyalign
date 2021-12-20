from pyalign.tests import TestCase
from typing import Iterator, List

import pyalign.problems
import pyalign.solve
import pyalign.gaps


class TestGotoh(TestCase):
	# the following example is taken from the slides
	# about Gotoh by Rolf Backofen.

	def test_CC_ACCT(self):

		for pf in self._problems(
			"ACT",
			pyalign.problems.Equality(eq=0, ne=1),
			direction="minimize"):

			solver = pyalign.solve.GlobalSolver(
				gap_cost=pyalign.gaps.AffineGapCost(5, 1),
				codomain=Iterator[pyalign.solve.Alignment])

			problem = pf.new_problem("CC", "ACCT")
			alignments = solver.solve(problem)

			self._check_alignments(
				alignments,
				7.0,
				[[0, 0], [1, 1]],
				[[0, 2], [1, 3]]
			)

	# the following test cases are inspired from the examples given in
	# the preprint by Flouri et al., 2015:
	# "Are all global alignment algorithms and implementations correct?"

	def test_GGTGTGA_TCGCGT(self):
		'''
		from Bio import pairwise2
		distances = {
			("T", "C"): -1,
			("A", "T"): 0,
			("G", "T"): -2,
			("A", "A"): 6,
			("T", "T"): 6,
			("G", "G"): 6
		}
		def d(x, y):
			z = distances.get((x, y), distances.get((y, x), 0))
			return z
		alignments = pairwise2.align.globalcd(
			"GGTGTGA", "TCGCGT", d, -11, -1, -11, -1)
		alignments		
		'''

		for pf in self._problems(
			"ACGT",
			pyalign.problems.Dict({
				("T", "C"): -1,
				("A", "T"): 0,
				("G", "T"): -2,
				("A", "A"): 6,
				("T", "T"): 6,
				("G", "G"): 6
			}),
			direction="maximize"):

			solver = pyalign.solve.GlobalSolver(
				gap_cost=pyalign.gaps.AffineGapCost(11, 1),
				codomain=List[pyalign.solve.Alignment])

			problem = pf.new_problem("GGTGTGA", "TCGCGT")
			alignments = solver.solve(problem)

			self._check_alignments(
				alignments,
				-2,
				[[0, 0],
				 [1, 1],
				 [3, 2],
				 [4, 3],
				 [5, 4],
				 [6, 5]])

	def test_AAAGGG_TTAAAAGGGGTT(self):
		'''
		from Bio import pairwise2
		pairwise2.align.globalcd(
			"AAAGGG", "TTAAAAGGGGTT", lambda x, y: 0 if x == y else -1, -5, -1, -5, -1)
		'''

		for pf in self._problems(
			"AGT",
			pyalign.problems.Equality(eq=0, ne=-1),
			direction="maximize"):

			solver = pyalign.solve.GlobalSolver(
				gap_cost=pyalign.gaps.AffineGapCost(5, 1),
				codomain=Iterator[pyalign.solve.Alignment])

			problem = pf.new_problem("AAAGGG", "TTAAAAGGGGTT")
			alignments = solver.solve(problem)

			self._check_alignments(
				alignments,
				-14,
				[[0, 3],
				 [1, 4],
				 [2, 5],
				 [3, 6],
				 [4, 7],
				 [5, 8]],
				[[0, 0],
				 [1, 1],
				 [2, 2],
				 [3, 9],
				 [4, 10],
				 [5, 11]])

	def test_TAAATTTGC_TCGCCTTAC(self):
		'''
		from Bio import pairwise2
		pairwise2.align.globalcd(
		    "TAAATTTGC", "TCGCCTTAC", lambda x, y: 10 if x == y else -30, -40, -1, -40, -1)
		'''

		from typing import List

		for pf in self._problems(
			"ACGT",
			pyalign.problems.Equality(eq=10, ne=-30),
			direction="maximize"):

			solver = pyalign.solve.GlobalSolver(
				gap_cost=pyalign.gaps.AffineGapCost(40, 1),
				codomain=List[pyalign.solve.Alignment])

			problem = pf.new_problem("TAAATTTGC", "TCGCCTTAC")
			alignments = solver.solve(problem)

			self._check_alignments(
				alignments,
				-60,
				[[0, 0],
				 [1, 7],
				 [8, 8]],
				[[0, 6],
				 [1, 7],
				 [8, 8]])

	def test_AGAT_CTCT(self):
		'''
		from Bio import pairwise2
		pairwise2.align.globalcd(
			"AGAT", "CTCT", lambda x, y: 10 if x == y else -30, -25, -1, -25, -1)
		'''

		for open_cost, expected_score in ((25, -44), (30, -54)):

			for pf in self._problems(
				"ACGT",
				pyalign.problems.Equality(eq=10, ne=-30),
				direction="maximize"):

				solver = pyalign.solve.GlobalSolver(
					gap_cost=pyalign.gaps.AffineGapCost(open_cost, 1),
					codomain=List[pyalign.solve.Alignment])

				problem = pf.new_problem("AGAT", "CTCT")
				alignments = solver.solve(problem)

				self._check_alignments(
					alignments,
					expected_score,
					[[3, 3]])
