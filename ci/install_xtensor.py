import subprocess
import sys
import os
import pybind11
import re
from pathlib import Path

lib_name = sys.argv[1]
tag_name = sys.argv[2]

repo_url = f'https://github.com/xtensor-stack/{lib_name}'

home = Path.cwd() / "deps"
home.mkdir(exist_ok=True)

subprocess.check_call(['git', 'clone', repo_url], cwd=home)
subprocess.check_call(['git', 'checkout', f'tags/{tag_name}', '-b', tag_name], cwd=home / lib_name)
build = home / lib_name / "build"
build.mkdir()

cmake_args = [
    f"-DPYTHON_EXECUTABLE={sys.executable}",
]

if lib_name == 'xtensor':
    cmake_args.append("-DXTENSOR_USE_XSIMD=ON")

if os.name == 'nt':

    install_path = home / "xtensor_prefix"
    install_path.mkdir(exist_ok=True)

    cmake_args.append(f"-Dxtl_DIR={home}/xtl/build")
    cmake_args.append(f"-Dxtensor_DIR={home}/xtensor/build")
    cmake_args.append(f"-Dxsimd_DIR={home}/xsimd/build")

    cmake_args.append(f"-DCMAKE_PREFIX_PATH={install_path}")
    cmake_args.append(f"-DCMAKE_INSTALL_PREFIX={install_path}")

    cmake_args.extend(['-G', 'MinGW Makefiles'])

if sys.platform.startswith("darwin"):
    # Cross-compile support for macOS - respect ARCHFLAGS if set
    archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
    if archs:
        cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]


subprocess.check_call(['cmake', *cmake_args, '..'], cwd=build)
subprocess.check_call(['make', 'install'], cwd=build)
