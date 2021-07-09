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
if os.name == 'nt':
	subprocess.check_call(['cmake', '-G', 'MinGW Makefiles', '..'], cwd=build)
else:
	subprocess.check_call(['cmake', '..'], cwd=build)
subprocess.check_call(['make', 'install'], cwd=build)
