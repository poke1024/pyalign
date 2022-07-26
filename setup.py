import numpy as np
import yaml
import sys
import os
import platform

from pathlib import Path
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

script_dir = Path(os.path.abspath(os.path.dirname(__file__)))

with open(script_dir / 'environment.yml') as f:
	required = yaml.safe_load(f.read())['dependencies'][-1]['pip']


is_arm = (platform.machine() == "arm64")  # Apple Silicon or ARM?
is_apple_arm = is_arm and platform.system() == 'Darwin'

if not is_arm:
	required.append("cpufeature==0.2.0")

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


def mk_ext(name, arch=None, cpu=None):
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
		if arch is not None:
			extra_compile_args.append(f"-march={arch}")
		if cpu is not None:
			extra_compile_args.append(f"-mcpu={cpu}")

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
	if is_apple_arm:
		ext_modules.append(mk_ext('apple_m1', cpu='apple-m1'))
	elif not is_arm:
		ext_modules.append(mk_ext('intel_avx2', arch='haswell'))
elif is_apple_arm:
	ext_modules.append(mk_ext('apple_m1', cpu='apple-m1'))
else:
	ext_modules.append(mk_ext('native', arch='native'))

with open(script_dir / 'README.md') as f:
	long_description = f.read()

exec(open("pyalign/_version.py").read())

setup(
	name='pyalign',
	version=__version__,
	packages=find_packages(include=[
		'pyalign',
		'pyalign.algorithm',
		'pyalign.problems',
		'pyalign.gaps',
		'pyalign.io',
		'pyalign.tests']),
	python_requires='>=3.7',
	license='MIT',
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
