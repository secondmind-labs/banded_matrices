#
# Copyright (c) 2021 The banded_matrices Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional

from setuptools import Extension
from setuptools.command.build_ext import build_ext as build_ext_orig

if sys.platform.startswith("linux"):
    _BANDED_MATRICES_COMPILER = "g++"
elif sys.platform.startswith("darwin"):
    _BANDED_MATRICES_COMPILER = "g++"
else:
    raise RuntimeError(
        f"Unsupported platform encountered ({sys.platform}) - only Linux and Darwin-based MacOS "
        f"are currently supported"
    )


_BANDED_MATRICES_BUILD_TYPE = "release"


def build_cmake_library(
    package_dir: Path,
    build_temp: Path,
    python_executable: str = sys.executable,
    announce: Optional[Callable[[str], None]] = None,
    spawn: Optional[Callable[[List[str]], None]] = None,
) -> List[Path]:
    cwd = Path().absolute()

    build_temp.mkdir(parents=True, exist_ok=True)

    package_lib_dir = package_dir / "lib"
    package_lib_dir.mkdir(parents=True, exist_ok=True)

    cmake_args = [
        str(package_dir),
        "-Wno-dev",
        f"-DPYTHON_BIN={python_executable}",
        f"-DCMAKE_BUILD_TYPE={_BANDED_MATRICES_BUILD_TYPE}",
        f"-DCMAKE_CXX_COMPILER={_BANDED_MATRICES_COMPILER}",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={str(package_lib_dir)}",
        f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={str(package_dir / 'bin')}",
        f"-DCMAKE_VERBOSE_MAKEFILE:BOOL=on",
        "-DCMAKE_CXX_STANDARD=17",
    ]

    if announce is not None:
        announce(f"Building banded_matrices library at {str(package_lib_dir)}")

    if spawn is None:
        spawn = subprocess.check_call

    os.chdir(str(build_temp))
    try:
        spawn(["cmake"] + cmake_args)
        spawn(["cmake", "--build", "."])
    finally:
        os.chdir(str(cwd))

    return list(package_lib_dir.glob("libbanded_matrices.*"))


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=["dummy.c"])


class build_ext(build_ext_orig):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        cwd = Path().absolute()

        # Poetry includes package data from the source tree. Put the generated TensorFlow op
        # under the package's lib directory so clean PEP 517 builds produce complete wheels.
        built_libraries = build_cmake_library(
            package_dir=cwd / ext.name,
            build_temp=Path(self.build_temp),
            announce=self.announce,
            spawn=self.spawn,
        )

        build_lib_dir = Path(self.build_lib) / ext.name / "lib"
        build_lib_dir.mkdir(parents=True, exist_ok=True)
        for library in built_libraries:
            shutil.copy2(library, build_lib_dir / library.name)


def build(setup_kwargs):
    """This custom build function will be called when running `poetry build`."""

    custom_kwargs = {
        "cmdclass": {"build_ext": build_ext},
        "ext_modules": [CMakeExtension("banded_matrices")],
        "include_package_data": True,
        "package_data": {"banded_matrices": ["lib/libbanded_matrices.*"]},
    }

    # Edit `setup_kwargs` in-place
    for key, value in custom_kwargs.items():
        assert not key in setup_kwargs, f"{key} already set: {setup_kwargs[key]}"
        setup_kwargs[key] = value
