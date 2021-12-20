import numpy as np

from typing import Dict


class Function:
	def get(self, u, v):
		raise NotImplementedError()

	def __call__(self, u, v):
		return self.get(u, v)

	def build_matrix(self, encoder, matrix):
		raise NotImplementedError()

	@property
	def binary_similarity_values(self):
		return None


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
	def __init__(self, pairs: Dict, default=0):
		self._dict = pairs
		self._default = default

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
		matrix.fill(self._default)
		set = np.zeros(matrix.shape, dtype=bool)
		for uv, w in self._dict.items():
			i, j = encoder.encode(uv)
			if set[i, j]:
				if w != matrix[j, i]:
					raise ValueError(
						f"asymmetric w: w({i}, {j}) = {w} != w({j}, {i}) = {matrix[j, i]}")
			else:
				matrix[i, j] = w
				matrix[j, i] = w
				set[i, j] = True
				set[j, i] = True


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

	@property
	def binary_similarity_values(self):
		return self._eq, self._ne
