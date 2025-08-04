#!/usr/bin/env python
"""
GDPlib open source model library for Generalized Disjunctive Programming

This file is maintained for backward compatibility.
Modern installations should use `pip install -e .` which will use pyproject.toml.
"""
import sys
from logging import warning

from setuptools import setup

# For backward compatibility with legacy installations
# The actual project configuration is now in pyproject.toml
if __name__ == "__main__":
    setup()
