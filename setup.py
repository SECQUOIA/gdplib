#!/usr/bin/env python
"""
GDPlib open source model library for Generalized Disjunctive Programming
"""
import sys
from logging import warning

from setuptools import setup, find_packages


kwargs = dict(
    name='gdplib',
    packages=find_packages(),
    install_requires=[],
    extras_require={},
    package_data={
        # If any package contains *.template or *.json files, include them:
        '': ['*.template', '*.json']
    },
    scripts=[],
    author='Qi Chen',
    author_email='qichen@andrew.cmu.edu',
    maintainer='Qi Chen',
    url="https://github.com/grossmann-group/gdplib",
    license='BSD 3-clause',
    description="GDPlib open source model library for Generalized Disjunctive Programming",
    long_description=__doc__,
    data_files=[],
    keywords=["pyomo", "generalized disjunctive programming"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
)

try:
    setup(setup_requires=['setuptools_scm'], use_scm_version=True, **kwargs)
except (ImportError, LookupError):
    default_version = '1.0.0'
    warning('Cannot use .git version: package setuptools_scm not installed '
            'or .git directory not present.')
    print('Defaulting to version: {}'.format(default_version))
    setup(**kwargs)
