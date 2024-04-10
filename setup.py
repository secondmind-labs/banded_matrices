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
    "importlib_metadata>=4.4,<5.0",
    "numpy>=1.18.0,<2.0.0",
    "tensorflow>=2.8.0,<2.9.0",
]

with open("VERSION") as file:
    version = file.read().strip()

with open("README.md") as file:
    long_description = file.read()

setup_kwargs = {
    "name": "banded_matrices",
    "version": version,
    "description": "Native (C++) implementation of Banded Matrices for TensorFlow",
    "long_description": long_description,
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
