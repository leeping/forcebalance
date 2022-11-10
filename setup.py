#!/usr/bin/env python
"""
setup.py: Install ForceBalance.
"""
from __future__ import print_function

__author__ = "Lee-Ping Wang"

import glob

from setuptools import Extension, setup

import versioneer

#===================================#
#|   ForceBalance version number   |#
#| Make sure to update the version |#
#| manually in :                   |#
#|                                 |#
#| doc/header.tex                  |#
#| doc/api_header.tex              |#
#| bin/ForceBalance.py             |#
#===================================#

__version__ = versioneer.get_version()

long_description = """

    ForceBalance (https://simtk.org/home/forcebalance) is a library
    that provides tools for automated optimization of force fields and
    empirical potentials.

    The philosophy of this program is to present force field
    optimization in a unified and easily extensible framework.  Since
    there are many different ways in theoretical chemistry to compute
    the potential energy of a collection of atoms, and similarly many
    types of reference data to fit these potentials to, we do our best
    to provide an infrastructure which allows a user or a contributor
    to fit any type of potential to any type of reference data.

    """

# DCD file reading module
DCD = Extension(
    "forcebalance/_dcdlib",
    sources=["ext/molfile_plugin/dcdplugin_s.c"],
    libraries=["m"],
    include_dirs=["ext/molfile_plugin/include/", "ext/molfile_plugin"],
)

setup(
    name="forcebalance",
    author="Lee-Ping Wang",
    author_email="leeping@ucdavis.edu",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="BSD 2.0",
    description="Automated force field optimization.",
    long_description=long_description,
    url="https://simtk.org/home/forcebalance",
    packages=["forcebalance", "forcebalance.test", "forcebalance.gui"],
    package_dir={
        "forcebalance": "src",
        "forcebalance.test": "src/tests",
    },
    package_data={
        "forcebalance": [
            "AUTHORS",
            "LICENSE.txt",
            "data/*.py",
            "data/*.sh",
            "data/*.bash",
            "data/uffparms.in",
            "data/oplsaa.ff/*",
        ],
        "forcebalance.test": glob.glob("src/tests/files/*.*")
        + glob.glob("src/tests/files/forcefield/*.*")
        + glob.glob("src/tests/files/targets/*/*.*")
        + glob.glob("src/tests/files/*.*")
        + ["src/tests/files/work_queue_worker"],
    },
    data_files=[],
    ext_modules=[DCD],
    scripts=[
        "bin/MakeInputFile.py",
        "bin/filecnv.py",
        "bin/ForceBalance",
        "bin/TidyOutput",
        "bin/ForceBalance.py",
    ],
)
