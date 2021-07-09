import subprocess
import sys
import os
import pathlib
import pybind11

lib_name = sys.argv[1]
tag_name = sys.argv[2]

pybind11.get_cmake_dir()

repo_url = 'https://github.com/xtensor-stack/' + lib_name

home = pathlib.Path.home()

subprocess.check_call(['git', 'clone', repo_url], cwd=home)
subprocess.check_call(['git', 'checkout', f'tags/{tag_name}', '-b', tag_name], cwd=home / lib_name)
build = home / lib_name / "build"
build.mkdir()

cmake_args = []

if lib_name == 'xtensor':
	cmake_args.append("-DXTENSOR_USE_XSIMD=ON")

if os.name == 'nt':
	install_path = home / "xtensor_prefix"
	install_path.mkdir(exist_ok=True)

	cmake_args.append("-Dxtl_DIR={home}/xtl/build")
	cmake_args.append("-Dxtensor_DIR={home}/xtensor/build")

	cmake_args.append(f"-DCMAKE_PREFIX_PATH={install_path}")
	cmake_args.append(f"-DCMAKE_INSTALL_PREFIX={install_path}")

	cmake_args.extend(['-G', 'MinGW Makefiles'])

subprocess.check_call(['cmake', *cmake_args, '..'], cwd=build)
subprocess.check_call(['make', 'install'], cwd=build)
