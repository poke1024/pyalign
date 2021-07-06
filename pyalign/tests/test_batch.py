from pyalign.tests import TestCase

import pyalign.utils
import pyalign.solve
import pyalign.gaps


class TestBatch(TestCase):
	def test_simd(self):
		pf = pyalign.utils.SimilarityProblemFactory(
			pyalign.utils.Binary(eq=1, ne=-1))

		solver = pyalign.solve.GlobalSolver(
			gap_cost=pyalign.gaps.LinearGapCost(2),
			direction="maximize",
			generate="alignment[all, optimal]")

		alignments = solver.solve(pyalign.solve.ProblemBatch([
			pf.new_problem("AATCG", "AACG"),
			pf.new_problem("AATGC", "AACG"),
			pf.new_problem("AATCG", "AGTT")
		]))

		self._check_alignments(
			alignments[0],
			2,
			[[0, 0], [1, 1], [3, 2], [4, 3]])

		self._check_alignments(
			alignments[1],
			0,
			[[0, 0], [1, 1], [2, 2], [3, 3]])

		self._check_alignments(
			alignments[2],
			-2,
			[[0, 0], [1, 1], [2, 2], [3, 3]],
			[[0, 0], [1, 1], [2, 2], [4, 3]])
