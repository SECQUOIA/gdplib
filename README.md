# GDPlib

[![codecov](https://codecov.io/gh/SECQUOIA/gdplib/branch/main/graph/badge.svg)](https://codecov.io/gh/SECQUOIA/gdplib)
[![Tests](https://github.com/SECQUOIA/gdplib/workflows/Test/badge.svg)](https://github.com/SECQUOIA/gdplib/actions?query=workflow%3ATest)
[![Lint](https://github.com/SECQUOIA/gdplib/workflows/Lint/badge.svg)](https://github.com/SECQUOIA/gdplib/actions?query=workflow%3ALint)

We envision GDPlib as an open library of GDP models to provide examples for prospective modelers, and to provide a benchmarking set for algorithm developers.
We invite contributions to this library from the community, provided under the same BSD-3-clause or compatible license.

## Model Size Comparison

The following table shows the size metrics for all models in GDPlib:

| Component             |   [biofuel](./gdplib/biofuel/) |   [disease_model](./gdplib/disease_model/) |   [gdp_col](./gdplib/gdp_col/) |   [hda](./gdplib/hda/) |   [jobshop](./gdplib/jobshop/) |   [med_term_purchasing](./gdplib/med_term_purchasing/) |   [methanol](./gdplib/methanol/) |   [modprodnet](./gdplib/modprodnet/) |   [positioning](./gdplib/positioning/) |   [spectralog](./gdplib/spectralog/) |   [stranded_gas](./gdplib/stranded_gas/) |   [syngas](./gdplib/syngas/) |
|:----------------------|-------------------------------:|-------------------------------------------:|-------------------------------:|-----------------------:|-------------------------------:|-------------------------------------------------------:|---------------------------------:|-------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|-----------------------------:|
| variables             |                          36840 |                                       1250 |                            442 |                   1158 |                             10 |                                                   1165 |                              287 |                                  488 |                                     56 |                                  128 |                                    57810 |                          367 |
| binary_variables      |                            516 |                                         52 |                             30 |                     12 |                              6 |                                                    216 |                                8 |                                    2 |                                     50 |                                   60 |                                      192 |                           46 |
| integer_variables     |                           4356 |                                          0 |                              0 |                      0 |                              0 |                                                      0 |                                0 |                                  363 |                                      0 |                                    0 |                                    45360 |                            0 |
| continuous_variables  |                          31968 |                                       1198 |                            412 |                   1146 |                              4 |                                                    949 |                              279 |                                  123 |                                      6 |                                   68 |                                    12258 |                          321 |
| disjunctions          |                            252 |                                         26 |                             15 |                      6 |                              3 |                                                     72 |                                4 |                                    1 |                                     25 |                                   30 |                                       96 |                           23 |
| disjuncts             |                            516 |                                         52 |                             30 |                     12 |                              6 |                                                    216 |                                8 |                                    2 |                                     50 |                                   60 |                                      192 |                           46 |
| constraints           |                          12884 |                                        831 |                            610 |                    728 |                              9 |                                                    762 |                              429 |                                  486 |                                     30 |                                  158 |                                    14959 |                          543 |
| nonlinear_constraints |                             12 |                                          0 |                            262 |                    151 |                              0 |                                                      0 |                               55 |                                    1 |                                     25 |                                    8 |                                       18 |                           48 |

This table was automatically generated using the `generate_model_size_report.py` script.

## Model Size Comparison

The following table shows the size metrics for all models in GDPlib:

| Component             |   [batch_processing](./gdplib/batch_processing/) |   [biofuel](./gdplib/biofuel/) |   [cstr](./gdplib/cstr/) |   [disease_model](./gdplib/disease_model/) |   [ex1_linan_2023](./gdplib/ex1_linan_2023/) |   [gdp_col](./gdplib/gdp_col/) |   [hda](./gdplib/hda/) |   [jobshop](./gdplib/jobshop/) |   [med_term_purchasing](./gdplib/med_term_purchasing/) |   [methanol](./gdplib/methanol/) |   [modprodnet](./gdplib/modprodnet/) |   [positioning](./gdplib/positioning/) |   [small_batch](./gdplib/small_batch/) |   [spectralog](./gdplib/spectralog/) |   [stranded_gas](./gdplib/stranded_gas/) |   [syngas](./gdplib/syngas/) |   [water_network](./gdplib/water_network/) |
|:----------------------|-------------------------------------------------:|-------------------------------:|-------------------------:|-------------------------------------------:|---------------------------------------------:|-------------------------------:|-----------------------:|-------------------------------:|-------------------------------------------------------:|---------------------------------:|-------------------------------------:|---------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|-----------------------------:|-------------------------------------------:|
| variables             |                                              288 |                          36840 |                       76 |                                       1250 |                                           12 |                            442 |                   1158 |                             10 |                                                   1165 |                              287 |                                  488 |                                     56 |                                     37 |                                  128 |                                    57810 |                          367 |                                        395 |
| binary_variables      |                                              138 |                            516 |                       20 |                                         52 |                                           10 |                             30 |                     12 |                              6 |                                                    216 |                                8 |                                    2 |                                     50 |                                     18 |                                   60 |                                      192 |                           46 |                                         10 |
| integer_variables     |                                                0 |                           4356 |                        0 |                                          0 |                                            0 |                              0 |                      0 |                              0 |                                                      0 |                                0 |                                  363 |                                      0 |                                      0 |                                    0 |                                    45360 |                            0 |                                          0 |
| continuous_variables  |                                              150 |                          31968 |                       56 |                                       1198 |                                            2 |                            412 |                   1146 |                              4 |                                                    949 |                              279 |                                  123 |                                      6 |                                     19 |                                   68 |                                    12258 |                          321 |                                        385 |
| disjunctions          |                                                9 |                            252 |                       10 |                                         26 |                                            2 |                             15 |                      6 |                              3 |                                                     72 |                                4 |                                    1 |                                     25 |                                      9 |                                   30 |                                       96 |                           23 |                                          5 |
| disjuncts             |                                               18 |                            516 |                       20 |                                         52 |                                           10 |                             30 |                     12 |                              6 |                                                    216 |                                8 |                                    2 |                                     50 |                                     18 |                                   60 |                                      192 |                           46 |                                         10 |
| constraints           |                                              601 |                          12884 |                      100 |                                        831 |                                           10 |                            610 |                    728 |                              9 |                                                    762 |                              429 |                                  486 |                                     30 |                                     34 |                                  158 |                                    14959 |                          543 |                                        329 |
| nonlinear_constraints |                                                1 |                             12 |                       17 |                                          0 |                                            0 |                            262 |                    151 |                              0 |                                                      0 |                               55 |                                    1 |                                     25 |                                      1 |                                    8 |                                       18 |                           48 |                                         33 |

This table was automatically generated using the `generate_model_size_report.py` script.


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

For a developer install, please clone this repository, activate the correct python environment, and run:

```bash
pip install -r requirements.txt
pip install -e .
```

### Development Setup

For development work with enhanced Copilot integration:

1. **Install development dependencies:**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Set up GitHub Copilot with custom instructions:**
   - The repository includes custom Copilot instructions in `.github/copilot-instructions.md`
   - Project-specific configurations are available in `.copilot/`

3. **Run tests:**
   ```bash
   pytest tests/
   ```

4. **Code formatting and linting:**
   ```bash
   black gdplib/
   flake8 gdplib/
   ```

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
