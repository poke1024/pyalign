from unittest import TestCase

import pyalign.utils
import pyalign.solve
import pyalign.gaps
import numpy as np


class TestGotoh(TestCase):
	def _check_alignments(self, alignments, score, *edges):
		r_edges = []

		for x in alignments:
			self.assertTrue(x.score == score)
			r_edges.append(x.edges.tolist())

		edges = sorted(edges)
		r_edges = sorted(r_edges)
		self.assertTrue(len(edges) == len(r_edges))

		self.assertTrue((np.array(r_edges) == np.array(edges)).all())

	def test_cc_acct(self):
		# the following problem is taken from the slides
		# about Gotoh by Rolf Backofen.

		pf = pyalign.utils.DistanceProblemFactory(
			pyalign.utils.Binary(eq=0, ne=1))

		solver = pyalign.solve.GlobalSolver(
			gap_cost=pyalign.gaps.AffineGapCost(4, 1),
			direction="minimize",
			generate="alignment[all, optimal]")

		problem = pf.new_problem("CC", "ACCT")
		alignments = solver.solve(problem)

		self._check_alignments(
			alignments,
			7.0,
			[[0, 0], [1, 1]],
			[[0, 2], [1, 3]]
		)
