import numpy as np

from typing import Dict


class Function:
	def get(self, u, v):
		raise NotImplementedError()

	def __call__(self, u, v):
		return self.get(u, v)

	def build_matrix(self, encoder, matrix):
		raise NotImplementedError()


class Coalesced(Function):
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


class Dict(Function):
	def __init__(self, pairs: Dict):
		self._dict = pairs

	def get(self, u, v):
		a = self._dict.get((u, v))
		b = self._dict.get((v, u))

		if a is None and b is not None:
			return b
		elif b is None and a is not None:
			return a
		elif a is not None and b is not None:
			if a != b:
				raise ValueError(
					f"cost is not symmetric: d({u}, {v}) = {a} vs. d({v}, {u}) = {b}")
			return a
		else:
			return 0

	def build_matrix(self, encoder, matrix):
		for uv, w in self._dict.items():
			i, j = encoder.encode(uv)
			matrix[i, j] = w
			matrix[j, i] = w


class Equality(Function):
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

