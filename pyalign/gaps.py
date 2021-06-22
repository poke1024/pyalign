import numpy as np


class GapCost:
	def to_special_case(self):
		return {}

	def costs(self, n):
		raise NotImplementedError

	def _plot(self, ax, n):
		from matplotlib.ticker import MaxNLocator
		c = self.costs(n)
		ax.plot(c)
		ax.set_xlabel('gap length')
		ax.set_ylabel('cost')
		ax.set_ylim(-0.1, 1)
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax.grid()

	def plot(self, n):
		import matplotlib.pyplot as plt
		fig, ax = plt.subplots(1, 1, figsize=(12, 3))
		self._plot(ax, n)
		fig.tight_layout()
		fig.show()

	def plot_to_image(self, fig, ax, n, format='png'):
		self._plot(ax, n)
		buf = io.BytesIO()
		fig.tight_layout()
		fig.savefig(buf, format=format)
		buf.seek(0)
		data = buf.getvalue()
		buf.close()
		return data

	def _ipython_display_(self):
		# see https://ipython.readthedocs.io/en/stable/config/integrating.html
		self.plot(50)


class ConstantGapCost(GapCost):
	"""
	Models a constant gap cost \( w_k = u \). \( w_k \) is the gap
	cost at length \( k \). \( u \) is a configurable parameter.
	"""

	def __init__(self, u):
		self._cost = u

	def to_special_case(self):
		if self._cost == 0:
			return {
				'affine': 0
			}
		else:
			return {}

	def costs(self, n):
		c = np.empty((n,), dtype=np.float32)
		c.fill(self._cost)
		c[0] = 0
		return c


class AffineGapCost(GapCost):
	"""
	Models an affine gap cost \( w_k = u k \). \( w_k \) is the gap
	cost at length \( k \). \( u \) is a configurable parameter.
	"""

	def __init__(self, u):
		self._u = u

	def to_special_case(self):
		return {
			'affine': self._u
		}

	def costs(self, n):
		return np.linspace(0., (n - 1) * self._u, n, dtype=np.float32)


class GotohGapCost(GapCost):
	"""
	Models a gap cost that is \( w_k = u k + v \) if \( k \le K_1 \),
	and constant, i.e. \( w_k = w_{K_1} \), if \( k > K_1 \).

	Gotoh, O. (1982). An improved algorithm for matching biological
	sequences. Journal of Molecular Biology, 162(3), 705–708.
	https://doi.org/10.1016/0022-2836(82)90398-9
	"""

	def __init__(self, u, v, k1, wk1):
		self._u = u
		self._v = v
		self._k1 = k1
		self._wk1 = wk1

	def to_special_case(self):
		return {
			'gotoh': (self._u, self._v, self._k1, self._wk1)
		}

	def costs(self, n):
		w = np.linspace(0., (n - 1) * self._u, n, dtype=np.float32) + self._v
		w[np.arange(self._k1 + 1, n)] = self._wk1
		return w


class ExponentialGapCost(GapCost):
	"""
	Models a gap cost \( w_k = 1 - u^{-k v} \). \( w_k \) is the gap
	cost at length \( k \). \( u \) and \( v \) are configurable
	parameters.
	"""

	def __init__(self, u, v):
		self._u = u  # base
		self._v = v  # exp term

	def costs(self, n):
		c = np.empty((n,), dtype=np.float32)
		for i in range(n):
			c[i] = 1 - (self._u ** -(i * self._v))
		return c


class UserFuncGapCost(GapCost):
	def __init__(self, costs_fn):
		self._costs_fn = costs_fn

	def costs(self, n):
		c = np.empty((n,), dtype=np.float32)
		c[0] = 0
		for i in range(1, n):
			c[i] = self._costs_fn(i)
		return c


def smooth_gap_cost(k):
	"""
	Models gap cost as the complement of an exponentially decaying
	function that starts at \( w_0 = 1 \) and decreases
	slowly such that \( w_k = 0.5 \) for a given \( k \).
	"""

	if k > 0:
		return ExponentialGapCost(2, 1 / k)
	else:
		return ConstantGapCost(0)
