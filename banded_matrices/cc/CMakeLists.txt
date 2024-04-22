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

cmake_minimum_required(VERSION 3.10)


if(NOT PYTHON_BIN)
    set(PYTHON_BIN python3)
    execute_process(
        COMMAND which ${PYTHON_BIN}
        OUTPUT_VARIABLE PYTHON_BIN_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

message(STATUS "PYTHON_BIN=${PYTHON_BIN}")
message(STATUS "PYTHON_BIN_PATH=${PYTHON_BIN_PATH}")

if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
endif()

message(STATUS "CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")

# Postfix debug builds of the ops with "d" to prevent accidental misuse of debug builds
set(CMAKE_DEBUG_POSTFIX d)

# Compiler flags for Debug build
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

# Previously there has been numerical issues reported when using Fused Multiply-Add instructions,
# which caused the flag `-mno-fma` to be used. In later investigations we were unable to reproduce
# these issues, so that flag has been removed.

# Previously there has been numerical issues reported when allowing the compiler to optimise for
# native architectures, as builds on Jenkins were utilising the AVX-512 instruction set. To prevent
# these instructions being used (and to attempt to get some consistency between builds), we target
# the Haswell architecture for compilation.
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=haswell")

set(LIB_NAME "${PROJECT_NAME}")

if(NOT DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../lib)
endif()
message(STATUS "CMAKE_LIBRARY_OUTPUT_DIRECTORY=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

if(NOT DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../bin)
endif()
message(STATUS "CMAKE_RUNTIME_OUTPUT_DIRECTORY=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

# Get the flags for compiling and linking against TensorFlow
set(TF_CMD_COMPILE
    "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))")
set(TF_CMD_LINKS
    "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))")

execute_process(
    COMMAND ${PYTHON_BIN} -W ignore -c "${TF_CMD_COMPILE}"
    OUTPUT_VARIABLE TF_COMPILE_FLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE)

string(REPLACE "-I" "-isystem " TF_COMPILE_FLAGS "${TF_COMPILE_FLAGS}")
message(STATUS "TF_COMPILE_FLAGS=${TF_COMPILE_FLAGS}")

execute_process(
    COMMAND ${PYTHON_BIN} -W ignore -c "${TF_CMD_LINKS}"
    OUTPUT_VARIABLE TF_LINK_FLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE)

string(COMPARE EQUAL "${TF_LINK_FLAGS}" "" TF_LINK_FLAGS_NOT_FOUND)
if(TF_LINK_FLAGS_NOT_FOUND)
    message(FATAL_ERROR "TF_LINK_FLAGS is empty")
endif()

# Set the standard compilation and linking flags
set(CMAKE_CXX_FLAGS "-g -std=c++14 -Wall -Wextra -Wfloat-equal -Wshadow -Wconversion ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${TF_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")

set(LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${TF_LINK_FLAGS}")
string(STRIP ${LINKER_FLAGS} LINKER_FLAGS)
string(REPLACE " " ";" LINKER_FLAGS "${LINKER_FLAGS}")

if(APPLE)
    set(TF_CMD_PATH "import tensorflow as tf; import os; print(os.path.dirname(tf.__file__))")
    execute_process(
        COMMAND ${PYTHON_BIN} -W ignore -c "${TF_CMD_PATH}"
        OUTPUT_VARIABLE TF_PY_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(LINKER_FLAGS "${LINKER_FLAGS} -rpath ${TF_PY_PATH}")
    string(STRIP "${LINKER_FLAGS}" LINKER_FLAGS)
    message(STATUS "APPLE LINKER_FLAGS=${LINKER_FLAGS}")
endif()

message(STATUS "CXX_FLAGS=${CMAKE_CXX_FLAGS}")
message(STATUS "LINKER_FLAGS=${LINKER_FLAGS}")

link_directories("/usr/local/lib")
include_directories("/usr/local/include")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


#####################################################################
## TensorFlow ops library
#####################################################################

set(CC_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/banded_matrices")
set(TENSORFLOW_OPS_SOURCES
    ${CC_SRC}/reverse_inverse.cc
    ${CC_SRC}/block_band.cc
    ${CC_SRC}/cholesky.cc
    ${CC_SRC}/inverse.cc
    ${CC_SRC}/outer_vec_vec.cc
    ${CC_SRC}/pack_matrix.cc
    ${CC_SRC}/product_band_band.cc
    ${CC_SRC}/product_band_mat.cc
    ${CC_SRC}/solve_triang_band.cc
    ${CC_SRC}/solve_triang_mat.cc
    ${CC_SRC}/square_band.cc
    ${CC_SRC}/symmetrise.cc
    ${CC_SRC}/transpose_band.cc)

add_library(${LIB_NAME} SHARED ${TENSORFLOW_OPS_SOURCES})
target_link_libraries(${LIB_NAME} PRIVATE ${LINKER_FLAGS})
