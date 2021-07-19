from pyalign.tests import TestCase
from typing import Iterator

import pyalign.problems
import pyalign.solve
import pyalign.gaps


class TestGotoh(TestCase):
	def test_cc_acct(self):
		# the following problem is taken from the slides
		# about Gotoh by Rolf Backofen.

		pf = pyalign.problems.general(
			pyalign.problems.Equality(eq=0, ne=1),
			direction="minimize")

		solver = pyalign.solve.GlobalSolver(
			gap_cost=pyalign.gaps.AffineGapCost(4, 1),
			codomain=Iterator[pyalign.solve.Alignment])

		problem = pf.new_problem("CC", "ACCT")
		alignments = solver.solve(problem)

		self._check_alignments(
			alignments,
			7.0,
			[[0, 0], [1, 1]],
			[[0, 2], [1, 3]]
		)
