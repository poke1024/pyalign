"""
Fast and Versatile Alignments for Python.
"""

__pdoc__ = {
	'algorithm': False,
	'tests': False
}

from pyalign._version import __version__

from .gaps import *
from .solve import *
from .problems import *
from .simple import global_alignment, semiglobal_alignment, local_alignment
