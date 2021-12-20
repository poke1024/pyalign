from pyalign.tests import TestCase
from typing import Iterator

import pyalign.problems
import pyalign.solve
import pyalign.gaps


class TestBatch(TestCase):
	def test_simd(self):
		for pf in self._problems(
			"ACGT",
			pyalign.problems.Equality(eq=1, ne=-1),
			direction="maximize"):

			solver = pyalign.solve.GlobalSolver(
				gap_cost=pyalign.gaps.LinearGapCost(2),
				codomain=Iterator[pyalign.solve.Alignment],
				return_dup=False)

			alignments = solver.solve([
				pf.new_problem("AATCG", "AACG"),
				pf.new_problem("AATGC", "AACG"),
				pf.new_problem("AATCG", "AGTT")
			])

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
