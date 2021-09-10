# Banded Matrices

## Overview

A library providing C++ linear algebra operators (matmul, solve, ...) dedicated to banded matrices and a [TensorFlow](https://www.tensorflow.org/) interface.
This extends the set of existing TensorFlow operators which as of August 2021 only include `banded_triangular_solve`.

Details on the implemented operators may be found in Durrande et al.:
"[Banded Matrix Operators for Gaussian Markov Models in the Automatic Differentiation Era](http://proceedings.mlr.press/v89/durrande19a.html)", and in Adam et al.: "[Doubly Sparse Variational Gaussian Processes](http://proceedings.mlr.press/v108/adam20a.html)" 


## Installation

### For users

To install the latest (stable) release of the toolbox from [PyPI](https://pypi.org/), use `pip`:
```bash
$ pip install banded_matrices
```

### For contributors

This project uses [Poetry](https://python-poetry.org/docs) to
manage dependencies in a local virtual environment. To install Poetry, [follow the
instructions in the Poetry documentation](https://python-poetry.org/docs/#installation).

To install this project in editable mode, run the commands below from the root directory of the
`banded_matrices` repository.

```bash
poetry install
```

This command creates a virtual environment for this project
in a hidden `.venv` directory under the root directory.

You must also run the `poetry install` command to install updated dependencies when
the `pyproject.toml` file is updated, for example after a `git pull`.

**NOTE:** Unlike most other Python packages, by installing the `banded_matrices` package
from source you will trigger a compilation of the C++ TensorFlow ops library. This means that
running `poetry install` can take a while - in the order of 5 minutes, depending on the machine
you are installing onto.
  
#### Known issues

Poetry versions above `1.0.9` don't get along (for now) with Ubuntu 18.04, if you have this OS, 
you will likely need to install version `1.0.9`. This can be done with the following command

```bash
wget https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py
POETRY_VERSION=1.0.9 python get-poetry.py 
```

Recommended Poetry installation might pick up Python 2 if it is used by the operating system, 
this will cause problems with looking up libraries and sorting out dependencies if your 
library uses Python 3. If this happens, poetry has a command you can use to instruct it to use 
a correct Python version (here assuming you want to use python3.7 and have it installed on your 
system - note that `python3.7-venv` package will need to be installed as well). 

```bash
poetry env use python3.7 && poetry install
```

The `poetry install` command might fail to install certain Python packages 
(those that use the 'manylinux2010' platform tag), if the version of
`pip` installed when creating the Poetry virtual environment is too old.
Unfortunately the version of `pip` used when creating the virtual environment is vendored with each
Python version, and it is not possible to update this.

The solution is to update the version of `pip` in the Poetry virtual environment after the initial
install fails, and then reattempt the installation. To do this, use the command:

```bash
poetry install || { poetry run pip install -U pip==20.0.2 && poetry install; }
```

## Running the tests

Run these commands from the root directory of this repository. 
To run the full Python test suite, including pylint and Mypy, run: 

```bash
poetry run task test
```

Alternatively, you can run just the unit tests, starting with the failing tests and exiting after
the first test failure:

```bash
poetry run task quicktest
```

To run linting of the C++ code (using cpplint), run:

```bash
poetry run task cpplint
```

**NOTE:** Running the tests requires
that the project virtual environment has been updated. See [Installation](#Installation).
