import numpy as np

from pathlib import Path
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

src_path = Path('pyalign/algorithm').resolve()
assert src_path.exists()

sources = [src_path / 'pyalign.cpp']

include_dirs = [Path(np.get_include()), src_path]

ext_modules = [
	Pybind11Extension(
		'pyalign.algorithm',
		[str(x) for x in sorted(sources)],
		cxx_std=17,
		extra_compile_args=[
			"-march=haswell",
			"-O3",
			"-ftemplate-backtrace-limit=0"],
		include_dirs=[str(x) for x in include_dirs],
	),
]

setup(
	name='pyalign',
	version='0.1',
	packages=find_packages(include=['pyalign', 'pyalign.utils']),
	license='GPLv2',
	author='Bernhard Liebl',
	author_email='liebl@informatik.uni-leipzig.de',
	long_description='',
	ext_modules=ext_modules,
	cmdclass={"build_ext": build_ext},
	install_requires=["pybind11"],
)
