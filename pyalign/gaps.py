import numpy as np


class GapCost:
	"""
	A gap cost \( w_k \) that depends only on the
	gap length \( k \).
	"""

	def to_tuple(self):
		raise NotImplementedError()

	def to_special_case(self):
		return {}

	def costs(self, n):
		raise NotImplementedError()

	@property
	def title(self):
		raise NotImplementedError()

	def _plot_matplotlib(self, ax, n):
		from matplotlib.ticker import MaxNLocator
		c = self.costs(n)
		ax.plot(c)
		ax.set_xlabel('gap length')
		ax.set_ylabel('cost')
		ax.set_ylim(-0.1, 1)
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax.grid()

	def plot(self, n=None, backend="bokeh"):
		if n is None:
			n = 5

		if backend == "bokeh":
			import bokeh.plotting
			p = bokeh.plotting.figure(plot_width=600, plot_height=200)
			c = self.costs(n)
			p.line(np.arange(n), c, line_width=2)

			p.title = self.title
			p.xaxis.axis_label = 'gap length'
			p.yaxis.axis_label = 'cost'
			p.xaxis.ticker = np.arange(n)
			p.toolbar_location = None

			if np.amax(c) < 1:
				p.y_range.end = 1
			p.ray(
				x=[0], y=[1], length=0, angle=[0], color="black",
				line_width=1, line_dash='dotted')

			bokeh.plotting.show(p)
		elif backend == "matplotlib":
			import matplotlib.pyplot as plt
			fig, ax = plt.subplots(1, 1, figsize=(12, 3))
			self._plot_matplotlib(ax, n)
			fig.tight_layout()
			fig.show()
		else:
			raise ValueError(backend)

	def plot_to_image(self, fig, ax, n, format='png'):
		"""
		Produce a plot of this gap cost function.

		Parameters
		----------
		"""

		import io

		self._plot_matplotlib(ax, n)
		buf = io.BytesIO()
		fig.tight_layout()
		fig.savefig(buf, format=format)
		buf.seek(0)
		data = buf.getvalue()
		buf.close()
		return data

	def _ipython_display_(self):
		# see https://ipython.readthedocs.io/en/stable/config/integrating.html
		self.plot(5)


class ConstantGapCost(GapCost):
	"""
	A constant gap cost \( w_k = u \). \( w_k \) is the gap
	cost at length \( k \). \( u \) is a configurable parameter.
	"""

	def __init__(self, u):
		self._u = u
		assert u >= 0

	def to_tuple(self):
		return 'constant', self._u

	def to_special_case(self):
		if self._u == 0:
			return {
				'linear': 0
			}
		else:
			return {}

	def costs(self, n):
		w = np.full((n,), self._u, dtype=np.float32)
		w[0] = 0
		return w

	@property
	def title(self):
		return f"w(k) = {self._u:.1f}, w(0) = 0"


class LinearGapCost(GapCost):
	"""
	A linear gap cost \( w_k = u k \). \( w_k \) is the gap
	cost at length \( k \). \( u \) is a configurable parameter.

	Setting \( u = 0 \) effectively eliminates any gap costs.

	Notes
	-----
	   [1] Stojmirović, A., & Yu, Y.-K. (2009). Geometric Aspects of Biological Sequence
	       Comparison. Journal of Computational Biology, 16(4), 579–610. https://doi.org/10.1089/cmb.2008.0100
	"""

	def __init__(self, u):
		self._u = u
		assert u >= 0

	def to_tuple(self):
		return 'linear', self._u

	def to_special_case(self):
		return {
			'linear': self._u
		}

	def costs(self, n):
		return np.linspace(0., (n - 1) * self._u, n, dtype=np.float32)

	@property
	def title(self):
		return f"w(k) = {self._u:.1f} * k"


class AffineGapCost(GapCost):
	"""
	An affine gap cost \( w_k = u + v k \). \( w_k \) is the gap
	cost at length \( k \). \( u \) and \( v /) are configurable parameters.

	Notes
	-----
	   [1] Altschul, S. (1998). Generalized affine gap costs for protein sequence alignment.
           Proteins: Structure, 32.
	"""

	def __init__(self, open=None, extend=0, u=None, v=None):
		if open is not None:
			assert u is None and v is None
			self._u = open - extend
			self._v = extend
		else:
			assert extend == 0
			self._u = u
			self._v = v
		assert self._u >= 0
		assert self._v >= 0

	def to_tuple(self):
		return 'affine', self._u, self._v

	def to_special_case(self):
		return {
			'affine': (self._u, self._v)
		}

	def costs(self, n):
		w = self._u + np.linspace(0., (n - 1) * self._v, n, dtype=np.float32)
		w[0] = 0
		return w

	@property
	def title(self):
		return f"w(k) = {self._u:.1f} + {self._v:.1f} * k, w(0) = 0"


class LogarithmicGapCost(GapCost):
	""" A gap cost \( w_k =  u + v ln(k) \). \( w_k \) is the gap
	cost at length \( k \). \( u \) and \( v \) are configurable
	parameters.

	Notes
	-----
	   [1] Waterman, M. S. (1984). Efficient sequence alignment algorithms. Journal
	       of Theoretical Biology, 108(3), 333–337. https://doi.org/10.1016/S0022-5193(84)80037-5
	"""

	def __init__(self, u, v):
		self._u = u
		self._v = v
		assert u >= 0
		assert v >= 0

	def to_tuple(self):
		return 'logarithmic', self._u, self._v

	def costs(self, n):
		assert n > 0
		w = np.full((n,), self._u, dtype=np.float32)
		w[1:] += self._v * np.log(np.arange(1, n), dtype=np.float32)
		w[0] = 0
		return w

	@property
	def title(self):
		return f"w(k) = {self._u:.1f} + {self._v:.1f} * ln(k), w(0) = 0"


class ExponentialGapCost(GapCost):
	"""
	A gap cost \( w_k = 1 - u^{-k v} \). \( w_k \) is the gap
	cost at length \( k \). \( u \) and \( v \) are configurable
	parameters.
	"""

	def __init__(self, u, v):
		self._u = u  # base
		self._v = v  # exp term
		assert u >= 0
		assert v >= 0

	def to_tuple(self):
		return 'exponential', self._u, self._v

	def costs(self, n):
		c = np.empty((n,), dtype=np.float32)
		for i in range(n):
			c[i] = 1 - (self._u ** -(i * self._v))
		return c

	@property
	def title(self):
		return f"w(k) = 1 - {self._u:.1f}^(-k * {self._v:.1f})"


class UserFuncGapCost(GapCost):
	"""
	A custom gap cost defined by a Python user function.
	"""

	def __init__(self, costs_fn):
		self._costs_fn = costs_fn

	def to_tuple(self):
		return 'user', self._costs_fn

	def costs(self, n):
		c = np.empty((n,), dtype=np.float32)
		c[0] = 0
		for i in range(1, n):
			c[i] = self._costs_fn(i)
		return c

	@property
	def title(self):
		return f"w(k) = {self._costs_fn}, w(0) = 0"


def smooth_gap_cost(k):
	"""
	A gap cost modelled as the complement of an exponentially decaying
	function that starts at \( w_0 = 1 \) and decreases
	slowly such that \( w_k = 0.5 \) for a given \( k \).
	"""

	if k > 0:
		return ExponentialGapCost(2, 1 / k)
	else:
		return ConstantGapCost(0)
