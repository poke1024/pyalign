import numpy as np
import unittest


class TestCase(unittest.TestCase):
	def _check_alignments(self, alignments, score, *edges, places=7):
		computed_edges = []

		for x in alignments:
			self.assertAlmostEqual(x.score, score, places=places)
			computed_edges.append([tuple(edge) for edge in x.edges.tolist()])

		true_edges = sorted([tuple(edge) for edge in edges])
		computed_edges = sorted(computed_edges)
		self.assertTrue(len(true_edges) == len(computed_edges))

		self.assertTrue((np.array(computed_edges) == np.array(true_edges)).all())
