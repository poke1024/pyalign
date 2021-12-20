from collections.abc import Sequence
import numpy as np
import unittest


def to_tuple(x):
	if isinstance(x, Sequence):
		return tuple(map(to_tuple, x))
	else:
		return x


class TestCase(unittest.TestCase):
	def _problems(self, alphabet, sim, **kwargs):
		from ..problems import general, alphabetic, binary

		yield general(sim, **kwargs)
		yield alphabetic(alphabet, sim, **kwargs)

		eq_ne = sim.binary_similarity_values
		if eq_ne is not None:
			yield binary(*eq_ne, **kwargs)

	def _check_alignments(self, alignments, score, *edges, places=7):
		computed_edges = []

		for x in alignments:
			self.assertAlmostEqual(x.score, score, places=places)
			computed_edges.append(x.edges.tolist())

		true_edges = sorted(to_tuple(edges))
		computed_edges = sorted(to_tuple(computed_edges))

		self.assertEqual(true_edges, computed_edges)
