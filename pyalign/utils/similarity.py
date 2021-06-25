import numpy as np

from typing import Dict


class Encoder:
	def __init__(self, alphabet):
		self._alphabet = set(alphabet)
		self._ids = dict((k, i) for i, k in enumerate(self._alphabet))

	def encode(self, s):
		ids = self._ids
		return np.array([ids[x] for x in s], dtype=np.uint32)

	@property
	def alphabet(self):
		return self._alphabet


class SimilarityOperator:
	def get(self, u, v):
		raise NotImplementedError()

	def build_matrix(self, encoder, matrix):
		raise NotImplementedError()


class CoalescedSimilarity(SimilarityOperator):
	def __init__(self, *ops):
		self._ops = ops

	def get(self, u, v):
		for op in self._ops:
			x = op.get(u, v)
			if x is not None:
				return x
		return 0

	def build_matrix(self, encoder, matrix):
		for op in self._ops:
			op.build_matrix(encoder, matrix)


class PairwiseSimilarity(SimilarityOperator):
	def __init__(self, pairs: Dict):
		self._dict = pairs

	def get(self, u, v):
		return self._dict.get(u, v)

	def build_matrix(self, encoder, matrix):
		for uv, w in self._dict.items():
			i, j = encoder.encode(uv)
			matrix[i, j] = w
			matrix[j, i] = w


class BinarySimilarity(SimilarityOperator):
	def __init__(self, eq=1, ne=-1):
		self._eq = eq
		self._ne = ne

	def get(self, u, v):
		if u == v:
			return self._eq
		else:
			return self._ne

	def build_matrix(self, encoder, matrix):
		matrix.fill(self._ne)
		np.fill_diagonal(matrix, self._eq)

