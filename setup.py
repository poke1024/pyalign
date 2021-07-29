import numpy as np
import yaml
import sys
import os

from pathlib import Path
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

script_dir = Path(os.path.abspath(os.path.dirname(__file__)))

with open(script_dir / 'environment.yml') as f:
	required = yaml.safe_load(f.read())['dependencies'][-1]['pip']

src_path = (script_dir / 'pyalign' / 'algorithm').resolve()
assert src_path.exists()

sources = [
	src_path / 'module.cpp'
]

include_dirs = [Path(np.get_include()), script_dir]

if sys.platform == 'darwin':
	cc = os.environ.get("CC")
	if cc and cc.startswith("gcc"):
		import pybind11.setup_helpers
		pybind11.setup_helpers.MACOS = False


def mk_ext(name, march):
	extra_compile_args = []
	extra_link_args = []

	is_sanitize = os.environ.get('PYALIGN_SANITIZE_ADDRESS', False)

	if is_sanitize:
		is_debug_build = True
	else:
		is_debug_build = os.environ.get("PYALIGN_DEBUG_BUILD", False)

	if os.name == 'nt':
		pass
	else:
		if is_debug_build:
			extra_compile_args.append("-O0")
			extra_compile_args.append("-g")
		else:
			extra_compile_args.append("-O3")

		extra_compile_args.extend([
			"-ftemplate-backtrace-limit=0"])
		if march is not None:
			extra_compile_args.append(f"-march={march}")

	if is_sanitize:
		extra_compile_args.append('-fsanitize=address')
		extra_compile_args.append('-fno-omit-frame-pointer')
		extra_compile_args.append('-fno-optimize-sibling-calls')
		extra_link_args.append('-fsanitize=address')

	return Pybind11Extension(
		f'pyalign.algorithm.{name}.algorithm',
		[str(x) for x in sorted(sources)],
		cxx_std=17,
		extra_compile_args=extra_compile_args,
		extra_link_args=extra_link_args,
		include_dirs=[str(x) for x in include_dirs],
	)

ext_modules = []

if os.environ.get("PYALIGN_PREBUILT_MARCH"):
	ext_modules.append(mk_ext('generic', None))
	ext_modules.append(mk_ext('avx2', 'haswell'))
else:
	ext_modules.append(mk_ext('native', 'native'))

with open(script_dir / 'README.md') as f:
	long_description = f.read()

setup(
	name='pyalign',
	version='0.3.2',
	packages=find_packages(include=[
		'pyalign',
		'pyalign.algorithm',
		'pyalign.problems',
		'pyalign.gaps',
		'pyalign.io',
		'pyalign.tests']),
	python_requires='>=3.8',
	license='GPLv2',
	author='Bernhard Liebl',
	author_email='liebl@informatik.uni-leipzig.de',
	ext_modules=ext_modules,
	cmdclass={"build_ext": build_ext},
	install_requires=required,
	test_suite='nose.collector',
	tests_require=['nose'],
	description='Fast and Versatile Alignments for Python',
	long_description=long_description,
	long_description_content_type='text/markdown',
	#include_package_data=True,
)
