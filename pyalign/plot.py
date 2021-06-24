import bokeh.plotting
import bokeh.models
import bokeh.io
import numpy as np


def inset_arrows(data, d0=0.2):
	dx = data['x_end'] - data['x_start']
	dy = data['y_end'] - data['y_start']
	mx = (data['x_start'] + data['x_end']) / 2
	my = (data['y_start'] + data['y_end']) / 2
	l = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
	d = d0 / l
	return dict(
		x_start=mx - dx * d,
		y_start=my - dy * d,
		x_end=mx + dx * d,
		y_end=my + dy * d
	)


def flat_ix(a):
	return np.flip(np.dstack(np.meshgrid(
		np.arange(a.shape[1]),
		np.arange(a.shape[0]))).reshape(-1, 2), axis=-1)


class TracebackPlotFactory:
	def __init__(self, solution):
		self._solution = solution
		self._p = None
		self._len = None
		self._path = None

	def _create_plot(self):
		solution = self._solution

		values = solution.values[1:, 1:]

		len_s = values.shape[0]
		len_t = values.shape[1]
		cell_size = 40

		self._p = bokeh.plotting.figure(
			plot_width=len_t * cell_size, plot_height=len_s * cell_size,
			title=None, toolbar_location=None)

		self._len = (len_s, len_t)

		path_mask = np.logical_and(
			solution.path[:, 0] >= 0,
			solution.path[:, 1] >= 0)
		self._path = solution.path[path_mask] + 1

	def _plot_grid(self):
		len_s, len_t = self._len

		len_s_1 = len_s + 1
		len_t_1 = len_t + 1
		grid_line_width = 2
		grid_color = 'lightgray'

		self._p.multi_line(
			xs=(1 + np.array([[-1, len_t]] * len_s_1) - 0.5).tolist(),
			ys=(np.repeat(1 + np.arange(0, len_s_1), 2).reshape(len_s_1, 2) - 0.5).tolist(),
			color=grid_color, line_width=grid_line_width)

		self._p.multi_line(
			xs=(np.repeat(1 + np.arange(0, len_t_1), 2).reshape(len_t_1, 2) - 0.5).tolist(),
			ys=(1 + np.array([[-1, len_s]] * len_t_1) - 0.5).tolist(),
			color=grid_color, line_width=grid_line_width)

	def _shade_optimal_path_cells(self):
		path = self._path

		source = bokeh.models.ColumnDataSource(dict(
			x=path[:, 1],
			y=path[:, 0],
			width=[1] * path.shape[0],
			height=[1] * path.shape[0]))

		glyph = bokeh.models.Rect(
			x="x", y="y", width="width", height="height",
			fill_color="orange",
			fill_alpha=0.25,
			line_color=None)
		self._p.add_glyph(source, glyph)

	def _plot_traceback_matrix_arrows(self):
		path = self._path

		arrow_color = 'blue'
		arrow_alpha = 0.5
		traceback = self._solution.traceback[1:, 1:]

		src = flat_ix(traceback) + 1
		dst = traceback.reshape(-1, 2) + 1

		path_set = set(tuple(x) for x in path)
		mask = np.array([tuple(x) not in path_set for x in src], dtype=bool)
		src = src[mask]
		dst = dst[mask]

		source = bokeh.models.ColumnDataSource(data=inset_arrows(dict(
			x_start=src[:, 1],
			y_start=src[:, 0],
			x_end=dst[:, 1],
			y_end=dst[:, 0]), 0.1))
		self._p.add_layout(bokeh.models.Arrow(
			end=bokeh.models.OpenHead(
				line_color=arrow_color, line_alpha=arrow_alpha, line_width=1, size=5),
			source=source, x_start='x_start', y_start='y_start', x_end='x_end', y_end='y_end',
			line_color=None))

	def _plot_optimal_path_arrows(self):
		path = self._path
		arrow_color = 'orange'
		source = bokeh.models.ColumnDataSource(data=inset_arrows(dict(
			x_start=path[:-1, 1],
			y_start=path[:-1, 0],
			x_end=path[1:, 1],
			y_end=path[1:, 0])))
		self._p.add_layout(bokeh.models.Arrow(
			end=bokeh.models.OpenHead(line_color=arrow_color, line_width=1, size=5),
			source=source, x_start='x_start', y_start='y_start', x_end='x_end', y_end='y_end',
			line_color=arrow_color, line_width=2))

	def _plot_value_magnitudes(self):
		len_s, len_t = self._len
		solution = self._solution

		ix = flat_ix(solution.values) - 1

		source = bokeh.models.ColumnDataSource(
			data=dict(
				x=np.tile(np.arange(0, len_t + 1), len_s + 1),
				y=np.repeat(np.arange(0, len_s + 1), len_t + 1),
				value=[f'{x:.1f}' for x in solution.values.flatten()],
				color=['gray' if x < 0 or y < 0 else 'black' for x, y in ix]))

		labels = bokeh.models.LabelSet(
			x='x', y='y', text='value', text_color='color',
			x_offset=0, y_offset=0, source=source, render_mode='canvas',
			text_font_size='9pt', text_align='center', text_baseline='middle')

		self._p.add_layout(labels)

	def _configure(self):
		p = self._p
		len_s, len_t = self._len

		# if solution.problem.s is not None:
		# p.xaxis.ticker = bokeh.models.FixedTicker(ticks=solution.problem.s)

		p.xaxis.ticker = bokeh.models.FixedTicker(ticks=np.arange(0, len_t) + 1)
		p.yaxis.ticker = bokeh.models.FixedTicker(ticks=np.arange(0, len_s) + 1)
		p.y_range.flipped = True

		p.grid.grid_line_color = None

	def create(self):
		self._create_plot()
		self._plot_grid()
		self._shade_optimal_path_cells()
		self._plot_traceback_matrix_arrows()
		self._plot_optimal_path_arrows()
		self._plot_value_magnitudes()
		self._configure()
		return self._p
