import collections.abc
import string


def _seq_type(x):
	if isinstance(x, str):
		return 'text'
	elif isinstance(x, collections.abc.Sequence):
		if all(isinstance(y, str) for y in x):
			return 'text'
		elif all(hasattr(y, '_repr_html_') for y in x):
			return 'html'
		else:
			return None
	else:
		return None


def det_seq_type(s, t):
	seq_type = _seq_type(s)
	if seq_type != _seq_type(t):
		return None
	else:
		return seq_type


def _make_rows(s, t, ts, elastic=False):
	upper = []
	edges = []
	lower = []
	last_x = -1

	# flip upper, lower
	# flip s, t

	for i, x in enumerate(ts):
		if x < 0:
			edges.append(" ")
			lower.append(t[i])
			upper.append("-" if not elastic else s[max(last_x, 0)])
		else:
			for j in range(last_x + 1, x):
				upper.append(s[j])
				edges.append(" ")
				lower.append("-" if not elastic else t[i])

			upper.append(s[x])
			edges.append("|")
			lower.append(t[i])
			last_x = x

	for j in range(last_x + 1, len(s)):
		upper.append(s[j])
		edges.append(" ")
		lower.append("-" if not elastic else t[-1])

	return upper, edges, lower


class Formatter:
	_html_template = string.Template("""
		<table style="border-collapse: collapse; border-spacing:0;">
		<tr>$upper</tr>
		<tr style="background-color: #F0F0F0; padding:0;">$edges</tr>
		<tr>$lower</tr>
		</table>
		""")

	def __init__(self, alignment, style="t_to_s"):
		self._alignment = alignment
		self._style = style

	def _rows(self, s, t):
		elastic = self._alignment.solver.options["solver"] == "dtw"

		if self._style == "s_to_t":
			upper, edges, lower = _make_rows(t, s, self._alignment.s_to_t, elastic)
			return lower, edges, upper
		elif self._style == "t_to_s":
			return _make_rows(s, t, self._alignment.t_to_s, elastic)
		else:
			raise ValueError(self._style)

	@property
	def html(self):
		s = self._alignment.problem.s
		t = self._alignment.problem.t

		if s is None or t is None:
			return None

		seq_type = det_seq_type(s, t)

		if seq_type == "text":
			upper, edges, lower = self._rows(s, t)
		elif seq_type == "html":
			upper, edges, lower = self._rows(
				[x._repr_html_() for x in s],
				[x._repr_html_() for x in t])
		else:
			return None

		return Formatter._html_template.substitute(
			upper="".join([f"<td>{x}</td>" for x in upper]),
			edges="".join([f'<td style="text-align: center;">{x}</td>' for x in edges]),
			lower="".join([f"<td>{x}</td>" for x in lower]))

	@property
	def text(self):
		s = self._alignment.problem.s
		t = self._alignment.problem.t

		if s is None or t is None:
			return None

		if det_seq_type(s, t) != "text":
			return None

		upper, edges, lower = self._rows(s, t)

		return "\n".join([
			"".join(upper),
			"".join(edges),
			"".join(lower)
		])
