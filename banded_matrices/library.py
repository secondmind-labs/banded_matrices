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

from pathlib import Path

import tensorflow as tf

from banded_matrices.platform import get_library_extension

_EXPECTED_LIBRARY_LOCATION = Path(__file__).parent / "lib"
_EXPECTED_LIBRARY_NAME = f"libbanded_matrices.{get_library_extension()}"
_EXPECTED_LIBRARY_PATH = _EXPECTED_LIBRARY_LOCATION / _EXPECTED_LIBRARY_NAME


class CompiledLibraryError(BaseException):
    pass


def _load_library():
    """Attempt to load the Banded Matrices library."""
    if not _EXPECTED_LIBRARY_PATH.exists():
        raise CompiledLibraryError(
            f"A compiled version of the Banded Matrices library was not found in the expected "
            f"location ({_EXPECTED_LIBRARY_PATH})"
        )

    try:
        return tf.load_op_library(str(_EXPECTED_LIBRARY_PATH))
    except Exception as e:
        raise CompiledLibraryError(
            "An unknown error occurred when loading the Banded Matrices library. This can "
            "sometimes occur if the library was build against a different version of TensorFlow "
            "than you are currently running."
        ) from e


banded_ops = _load_library()
