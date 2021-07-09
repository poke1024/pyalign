import numpy as np
import yaml
import sys
import os

from pathlib import Path
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

with open('environment.yml') as f:
	required = yaml.safe_load(f.read())['dependencies'][-1]['pip']

src_path = Path('pyalign/algorithm').resolve()
assert src_path.exists()

sources = [src_path / 'pyalign.cpp']

include_dirs = [Path(np.get_include()), src_path]

if sys.platform == 'darwin':
	cc = os.environ.get("CC")
	if cc and cc.startswith("gcc"):
		import pybind11.setup_helpers
		pybind11.setup_helpers.MACOS = False


def mk_ext(name, march):
	extra_compile_args = []

	if os.name == 'nt':
		pass
	else:
		extra_compile_args.extend([
			"-O3",
			"-ftemplate-backtrace-limit=0"])
		if march is not None:
			extra_compile_args.append(f"-march={march}")

	return Pybind11Extension(
		f'pyalign.algorithm.{name}.algorithm',
		[str(x) for x in sorted(sources)],
		cxx_std=17,
		extra_compile_args=extra_compile_args,
		include_dirs=[str(x) for x in include_dirs],
	)


ext_modules = [
	mk_ext('generic', None)
]


if os.environ.get("PYALIGN_NO_SIMD") is not None:
	pass
else:
	ext_modules.append(mk_ext('avx2', 'haswell'))



setup(
	name='pyalign',
	version='0.1',
	packages=find_packages(include=[
		'pyalign', 'pyalign.problem', 'pyalign.util', 'pyalign.tests']),
	license='GPLv2',
	author='Bernhard Liebl',
	author_email='liebl@informatik.uni-leipzig.de',
	long_description='',
	ext_modules=ext_modules,
	cmdclass={"build_ext": build_ext},
	install_requires=required,
	test_suite='nose.collector',
	tests_require=['nose'],
)
