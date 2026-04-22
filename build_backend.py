#
# Copyright (c) 2026 The banded_matrices Contributors.
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

import subprocess
import sys
from pathlib import Path
from typing import List

from poetry.core.masonry import api as poetry_api

from build import build_cmake_library


def _spawn(command: List[str]) -> None:
    subprocess.check_call(command)


def _build_native_library() -> None:
    root = Path(__file__).resolve().parent
    build_cmake_library(
        package_dir=root / "banded_matrices",
        build_temp=root / "build" / "pep517",
        python_executable=sys.executable,
        spawn=_spawn,
    )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _build_native_library()
    return poetry_api.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _build_native_library()
    return poetry_api.build_editable(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    return poetry_api.build_sdist(sdist_directory, config_settings)


def get_requires_for_build_wheel(config_settings=None):
    return poetry_api.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_editable(config_settings=None):
    return poetry_api.get_requires_for_build_editable(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    return poetry_api.get_requires_for_build_sdist(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    return poetry_api.prepare_metadata_for_build_wheel(metadata_directory, config_settings)
