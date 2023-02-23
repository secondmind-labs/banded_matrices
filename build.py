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
import sys
from pathlib import Path

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

        # A temporary directory for builds to be conducted in
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        # The location that the components is going to get installed into.
        # Note that we abuse the fact that a (pointless, empty) Cython extension is going to be
        # generated and installed, and from this we can calculate the install location.
        ext_dir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()

        # Define the CMake arguments that we want for the build
        cmake_args = [
            str(cwd / ext.name),
            "-Wno-dev",
            f"-DPYTHON_BIN={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={_BANDED_MATRICES_BUILD_TYPE}",
            f"-DCMAKE_CXX_COMPILER={_BANDED_MATRICES_COMPILER}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={str(ext_dir / ext.name / 'lib')}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={str(ext_dir / ext.name / 'bin')}",
            f"-DCMAKE_VERBOSE_MAKEFILE:BOOL=on",
        ]

        os.chdir(str(build_temp))
        self.announce(f"Building {ext.name} library at {str(ext_dir)}")
        self.spawn(["cmake"] + cmake_args)
        self.spawn(["cmake", "--build", "."])
        os.chdir(str(cwd))


def build(setup_kwargs):
    """This custom build function will be called when running `poetry build`."""

    custom_kwargs = {
        "cmdclass": {"build_ext": build_ext},
        "ext_modules": [CMakeExtension("banded_matrices")],
        "include_package_data": True,
    }

    # Edit `setup_kwargs` in-place
    for key, value in custom_kwargs.items():
        assert not key in setup_kwargs, f"{key} already set: {setup_kwargs[key]}"
        setup_kwargs[key] = value
