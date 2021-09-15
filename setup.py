# -*- coding: utf-8 -*-
from setuptools import setup

from build import *

packages = ["banded_matrices"]

package_data = {
    "": ["*"],
    "banded_matrices": [
        "cc/*",
        "cc/include/banded_matrices/*",
        "cc/src/banded_matrices/*",
        "cc/test/*",
    ],
}

install_requires = [
    "cmake>=3.18.0,<3.19.0",
    "importlib_metadata>=1.6,<2.0",
    "numpy>=1.18.0,<2.0.0",
    "tensorflow>=2.2.1,<2.3.0",
]

setup_kwargs = {
    "name": "banded-matrices",
    "version": "0.0.4",
    "description": "Native (C++) implementation of Banded Matrices for TensorFlow",
    "long_description": None,
    "maintainer": None,
    "maintainer_email": None,
    "url": None,
    "packages": packages,
    "package_data": package_data,
    "install_requires": install_requires,
    "python_requires": ">=3.7,<4.0",
}

build(setup_kwargs)

setup(**setup_kwargs)
