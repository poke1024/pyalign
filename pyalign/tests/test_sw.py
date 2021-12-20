from pyalign.tests import TestCase
from typing import Iterator

import pyalign.problems
import pyalign.solve
import pyalign.gaps


class TestSmithWaterman(TestCase):
	def test_aatcg_aacg(self):
		# test case is taken from default settings at
		# http://rna.informatik.uni-freiburg.de/Teaching/index.jsp?toolName=Smith-Waterman

		for pf in self._problems(
			"ACGT",
			pyalign.problems.Equality(eq=1, ne=-1),
			direction="maximize"):

			problem = pf.new_problem("AATCG", "AACG")

			solver = pyalign.solve.LocalSolver(
				gap_cost=pyalign.gaps.LinearGapCost(2),
				codomain=Iterator[pyalign.solve.Alignment])

			alignments = list(solver.solve(problem))

			self._check_alignments(
				alignments,
				2,
				[[0, 0], [1, 1]],
				[[3, 2], [4, 3]])
