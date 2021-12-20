from pyalign.tests import TestCase

import pyalign.problems
import pyalign.solve
import pyalign.gaps


class TestNeedlemanWunsch(TestCase):
	def test_aatcg_aacg(self):
		# test case is taken from default settings at
		# http://rna.informatik.uni-freiburg.de/Teaching/index.jsp?toolName=Needleman-Wunsch

		for pf in self._problems(
			"ACGT",
			pyalign.problems.Equality(eq=1, ne=-1),
			direction="maximize"):

			problem = pf.new_problem("AATCG", "AACG")

			solver = pyalign.solve.GlobalSolver(
				gap_cost=pyalign.gaps.LinearGapCost(2),
				codomain=pyalign.solve.Alignment)

			alignment = solver.solve(problem)

			self._check_alignments(
				[alignment],
				2.0,
				[[0, 0], [1, 1], [3, 2], [4, 3]])
