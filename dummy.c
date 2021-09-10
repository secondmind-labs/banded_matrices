//
// Copyright (c) 2021 The banded_matrices Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

/**
In this project we need to compile a C++ library against the TensorFlow libraries in order to
implement some custom TensorFlow operations.

The easiest way to get this working with the Python distribution system is to perform the
compilation of this library as a `build_ext` step, pretending that we're interested in compiling a
standard C extension to Python, as that ensures that compilation of the custom TF operations is
performed when we would want it (i.e. during installing from source or SDIST). We don't actually
care about the C extension that gets built, only that it does _and_ that it triggers the compilation
of the library we care about.

As the Clang compiler (as used by MacOS) will complain if we try to build a C extension without any
files, this dummy C file is included to ensure that compilation of the C extension (and therefore
the custom TensorFlow operations) succeeds.
**/

int main(int argc, char **argv) {
    return 0;
}
