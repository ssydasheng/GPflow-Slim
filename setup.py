#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: skip-file

from setuptools import setup
from setuptools import find_packages

import re
import os
import sys
from pkg_resources import parse_version

# load version form _version.py
exec(open("gpflowSlim/_version.py").read())

# Dependencies
requirements = [
    'numpy>=1.10.0',
    'scipy>=0.18.0',
    'pandas>=0.18.1',
    'sympy>=1.1.1'
]

packages = find_packages('.')
package_data={'gpflowSlim': ['gpflowSlim/gpflowrc']}

setup(name='gpflowSlim',
      version=__version__,
      author="Shengyang Sun, Guodong Zhang",
      author_email="ssy@cs.toronto.edu",
      description=("customed GPflow with simple Tensorflow API"),
      license="Apache License 2.0",
      keywords="machine-learning gaussian-processes kernels tensorflow",
      url="http://github.com/ssydasheng/GPflow-Slim",
      packages=packages,
      install_requires=requirements,
      tests_require=['pytest'],
      package_data=package_data,
      include_package_data=True,
      test_suite='tests',
      classifiers=[
          'License :: OSI Approved :: Apache Software License',
          'Natural Language :: English',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ])
