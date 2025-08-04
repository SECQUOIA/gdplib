#!/usr/bin/env python
"""
GDPlib open source model library for Generalized Disjunctive Programming

This file is maintained for backward compatibility.
Modern installations should use `pip install -e .` which will use pyproject.toml.
"""
import sys
from logging import warning

from setuptools import setup, find_packages

# For backward compatibility with legacy installations
# The actual project configuration is now in pyproject.toml
if __name__ == "__main__":
    setup(
        name="gdplib",
        packages=find_packages(),
        install_requires=[
            "Pyomo>=5.6.1",
            "setuptools>=39.0.1",
            "pandas>=1.0.1",
            "matplotlib>=2.2.2",
            "scipy>=1.0.0",
            "pint>=0.15.0",
            "openpyxl>=3.0.0",
        ],
        python_requires=">=3.9, <3.13",
        # All other metadata is now in pyproject.toml
        # This setup.py is maintained only for backward compatibility
    )
