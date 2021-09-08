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
"""
Platform-specific code required for loading the `banded_matrices` library.
"""

import sys


def get_library_extension() -> str:
    """Get the expected library extension for the current platform."""
    if _is_running_on_linux():
        return "so"
    if _is_running_on_macos():
        return "dylib"
    raise UnsupportedPlatformError()


class UnsupportedPlatformError(RuntimeError):
    """An error noting that the code is being run on an unsupported platform."""

    def __init__(self):
        super().__init__(
            f"The current platform ({sys.platform}) is not currently supported by the TensorFlow "
            "ops library."
        )


def _is_running_on_linux() -> bool:
    """Returns `true` is the code is being run on a Linux system."""
    return sys.platform.startswith("linux")


def _is_running_on_macos() -> bool:
    """Returns `true` if the code is being run on a MacOS system."""
    return sys.platform.startswith("darwin")
