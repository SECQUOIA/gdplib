# GDPlib

We envision GDPlib as an open library of GDP models to provide examples for prospective modelers, and to provide a benchmarking set for algorithm developers.
We invite contributions to this library from the community, provided under the same BSD-3-clause or compatible license.

## Installation

GDPlib is an installable model library in Python.
To install GDPlib, you can use:

```
pip install gdplib
```

To update GDPlib:

```
pip install --upgrade gdplib
```

For a developer install, please clone this repository, activate the correct python environment, and run `python setup.py develop` on the `setup.py` file in this directory.

## Model descriptions

Details for each model are given in a separate README.md file in each directory.
Navigate to these directories to read the files.

## Using this library

Once GDPlib is installed, functions for constructing the desired models can be imported from each of the main subpackages.
For example, [``biofuel/__init__.py``](./gdplib/biofuel/__init__.py) exposes a ``build_model`` function, allowing the user to write the following:

```python
from gdplib.biofuel import build_model as build_biofuel_model
pyomo_model = build_biofuel_model()
```

## Adding models to the library

To add new models to the library, the following steps should be taken:

1. Ensure that you have the requisite permissions to contribute the model to an open source library.
2. Add your files into one of the existing directories or a new project directory: ``gdplib/mynewmodel``.
3. If a new directory is created, add the corresponding import to [``gdplib/__init__.py``](./gdplib/__init__.py).
4. Within your project directory, add the requisite imports and edits to the ``__all__`` statement in ``gdplib/mynewmodel/__init__.py`` to expose the appropriate build functions. See the other project directories for examples.
5. Within your project directory, create a ``README.md`` file describing the new model.

Directories are free to implement their own subpackages.

## Relative vs. absolute imports

Note that ``__main__`` scripts within projects (i.e. those that you plan to execute directly) will need to use absolute imports rather than relative imports.
For example, in [``gdplib/gdp_col/main.py``](./gdplib/gdp_col/main.py), we need to write ``from gdplib.gdp_col.fenske import calculate_Fenske`` rather than ``from .fenske import calculate_Fenske``.
