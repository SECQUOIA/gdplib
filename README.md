# GDPlib

[![codecov](https://codecov.io/gh/SECQUOIA/gdplib/branch/main/graph/badge.svg)](https://codecov.io/gh/SECQUOIA/gdplib)
[![Tests](https://github.com/SECQUOIA/gdplib/workflows/Test/badge.svg)](https://github.com/SECQUOIA/gdplib/actions?query=workflow%3ATest)
[![Lint](https://github.com/SECQUOIA/gdplib/workflows/Lint/badge.svg)](https://github.com/SECQUOIA/gdplib/actions?query=workflow%3ALint)

We envision GDPlib as an open library of GDP models to provide examples for prospective modelers, and to provide a benchmarking set for algorithm developers.
We invite contributions to this library from the community, provided under the same BSD-3-clause or compatible license.

## Model Size Comparison

The following table shows the size metrics for GDPlib models:

| Component             |   [batch_processing](./gdplib/batch_processing/) |   [biofuel](./gdplib/biofuel/) |   [cstr](./gdplib/cstr/) |   [disease_model](./gdplib/disease_model/) |   [ex1_linan_2023](./gdplib/ex1_linan_2023/) |   [gdp_col](./gdplib/gdp_col/) |   [grid](./gdplib/grid/) |   [hda](./gdplib/hda/) |   [jobshop](./gdplib/jobshop/) |   [kaibel](./gdplib/kaibel/) |   [med_term_purchasing](./gdplib/med_term_purchasing/) |   [methanol](./gdplib/methanol/) |   [mod_hens: conventional](./gdplib/mod_hens/) |   [mod_hens: mixed_discrete](./gdplib/mod_hens/) |   [mod_hens: mixed_integer](./gdplib/mod_hens/) |   [mod_hens: multiple_module_discrete](./gdplib/mod_hens/) |   [mod_hens: multiple_module_integer](./gdplib/mod_hens/) |   [mod_hens: single_module_discrete](./gdplib/mod_hens/) |   [mod_hens: single_module_integer](./gdplib/mod_hens/) |   [modprodnet: Decay](./gdplib/modprodnet/) |   [modprodnet: Dip](./gdplib/modprodnet/) |   [modprodnet: Distributed](./gdplib/modprodnet/) |   [modprodnet: Growth](./gdplib/modprodnet/) |   [modprodnet: QuarterDistributed](./gdplib/modprodnet/) |   [multiperiod_blending: mpbp_6](./gdplib/multiperiod_blending/) |   [pandemic](./gdplib/pandemic/) |   [positioning](./gdplib/positioning/) |   [reverse_electrodialysis](./gdplib/reverse_electrodialysis/) |   [small_batch](./gdplib/small_batch/) |   [spectralog](./gdplib/spectralog/) |   [stranded_gas: Gas_100](./gdplib/stranded_gas/) |   [stranded_gas: Gas_250](./gdplib/stranded_gas/) |   [stranded_gas: Gas_500](./gdplib/stranded_gas/) |   [stranded_gas: Gas_large](./gdplib/stranded_gas/) |   [stranded_gas: Gas_small](./gdplib/stranded_gas/) |   [syngas](./gdplib/syngas/) |   [water_network: none](./gdplib/water_network/) |   [water_network: piecewise](./gdplib/water_network/) |   [water_network: quadratic_nonzero_origin](./gdplib/water_network/) |   [water_network: quadratic_zero_origin](./gdplib/water_network/) |
|:----------------------|-------------------------------------------------:|-------------------------------:|-------------------------:|-------------------------------------------:|---------------------------------------------:|-------------------------------:|-------------------------:|-----------------------:|-------------------------------:|-----------------------------:|-------------------------------------------------------:|---------------------------------:|-----------------------------------------------:|-------------------------------------------------:|------------------------------------------------:|-----------------------------------------------------------:|----------------------------------------------------------:|---------------------------------------------------------:|--------------------------------------------------------:|--------------------------------------------:|------------------------------------------:|--------------------------------------------------:|---------------------------------------------:|---------------------------------------------------------:|-----------------------------------------------------------------:|---------------------------------:|---------------------------------------:|---------------------------------------------------------------:|---------------------------------------:|-------------------------------------:|--------------------------------------------------:|--------------------------------------------------:|--------------------------------------------------:|----------------------------------------------------:|----------------------------------------------------:|-----------------------------:|-------------------------------------------------:|------------------------------------------------------:|---------------------------------------------------------------------:|------------------------------------------------------------------:|
| Variables             |                                              288 |                          36840 |                       76 |                                       1250 |                                           12 |                            442 |                    47525 |                   1158 |                             10 |                         4033 |                                                   1165 |                              287 |                                            338 |                                             3802 |                                             498 |                                                       3802 |                                                       498 |                                                     4761 |                                                     501 |                                         488 |                                       488 |                                              3720 |                                          488 |                                                     1320 |                                                              726 |                             1221 |                                     56 |                                                            776 |                                     37 |                                  128 |                                              8805 |                                              8805 |                                              8805 |                                               11694 |                                               11694 |                          367 |                                              395 |                                                  1405 |                                                                  395 |                                                               420 |
| Binary variables      |                                              138 |                            516 |                       20 |                                         52 |                                           10 |                             30 |                    35000 |                     12 |                              6 |                          200 |                                                    216 |                                8 |                                             64 |                                             1728 |                                             128 |                                                       1728 |                                                       128 |                                                     2147 |                                                     131 |                                           2 |                                         2 |                                                42 |                                            2 |                                                       42 |                                                              252 |                              222 |                                     50 |                                                              8 |                                     18 |                                   60 |                                               164 |                                               164 |                                               164 |                                                 172 |                                                 172 |                           46 |                                               10 |                                                   510 |                                                                   10 |                                                                10 |
| Integer variables     |                                                0 |                           4356 |                        0 |                                          0 |                                            0 |                              0 |                        0 |                      0 |                              0 |                            0 |                                                      0 |                                0 |                                              0 |                                                0 |                                              96 |                                                          0 |                                                        96 |                                                        0 |                                                      96 |                                         363 |                                       363 |                                              1452 |                                          363 |                                                      492 |                                                                0 |                                0 |                                      0 |                                                              0 |                                      0 |                                    0 |                                              2520 |                                              2520 |                                              2520 |                                                5040 |                                                5040 |                            0 |                                                0 |                                                     0 |                                                                    0 |                                                                 0 |
| Continuous variables  |                                              150 |                          31968 |                       56 |                                       1198 |                                            2 |                            412 |                    12525 |                   1146 |                              4 |                         3833 |                                                    949 |                              279 |                                            274 |                                             2074 |                                             274 |                                                       2074 |                                                       274 |                                                     2614 |                                                     274 |                                         123 |                                       123 |                                              2226 |                                          123 |                                                      786 |                                                              474 |                              999 |                                      6 |                                                            768 |                                     19 |                                   68 |                                              6121 |                                              6121 |                                              6121 |                                                6482 |                                                6482 |                          321 |                                              385 |                                                   895 |                                                                  385 |                                                               410 |
| Disjunctions          |                                                9 |                            252 |                       10 |                                         26 |                                            2 |                             15 |                    12500 |                      6 |                              3 |                          100 |                                                     72 |                                4 |                                             32 |                                               64 |                                              64 |                                                         64 |                                                        64 |                                                       33 |                                                      65 |                                           1 |                                         1 |                                                21 |                                            1 |                                                       21 |                                                              126 |                              111 |                                     25 |                                                              4 |                                      9 |                                   30 |                                                79 |                                                79 |                                                79 |                                                  80 |                                                  80 |                           23 |                                                5 |                                                     5 |                                                                    5 |                                                                 5 |
| Disjuncts             |                                               18 |                            516 |                       20 |                                         52 |                                           10 |                             30 |                    35000 |                     12 |                              6 |                          200 |                                                    216 |                                8 |                                             64 |                                              128 |                                             128 |                                                        128 |                                                       128 |                                                       67 |                                                     131 |                                           2 |                                         2 |                                                42 |                                            2 |                                                       42 |                                                              252 |                              222 |                                     50 |                                                              8 |                                     18 |                                   60 |                                               158 |                                               158 |                                               158 |                                                 160 |                                                 160 |                           46 |                                               10 |                                                    10 |                                                                   10 |                                                                10 |
| Constraints           |                                              601 |                          12884 |                      112 |                                        831 |                                           10 |                            610 |                    52000 |                    728 |                              9 |                         5790 |                                                    762 |                              429 |                                            370 |                                             5362 |                                             562 |                                                       5362 |                                                       562 |                                                     8786 |                                                     565 |                                         486 |                                       486 |                                              1792 |                                          486 |                                                      672 |                                                             1020 |                             1107 |                                     30 |                                                            794 |                                     34 |                                  158 |                                              2671 |                                              2671 |                                              2671 |                                                3397 |                                                3397 |                          543 |                                              329 |                                                  1339 |                                                                  329 |                                                               334 |
| Nonlinear constraints |                                                1 |                             12 |                       17 |                                          0 |                                            0 |                            262 |                        0 |                    151 |                              0 |                         2128 |                                                      0 |                               55 |                                             96 |                                               96 |                                              96 |                                                         96 |                                                        96 |                                                       32 |                                                      96 |                                           1 |                                         1 |                                                36 |                                            1 |                                                       36 |                                                               60 |                              220 |                                     25 |                                                            162 |                                      1 |                                    8 |                                                 0 |                                                 0 |                                                 0 |                                                   0 |                                                   0 |                           48 |                                               33 |                                                    28 |                                                                   33 |                                                                33 |

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

For development work on Linux, use the Pixi environment when available. The
committed Pixi lock currently targets `linux-64`; on other platforms, use the
pip workflow below or add the appropriate Pixi platform and regenerate the lock
file.

```bash
pixi install
pixi run test
pixi run lint
```

The committed Pixi support surface is `linux-64` only because that is the
platform the maintainers can verify with `pixi install`, `pixi run test`, and
`pixi run lint`. The `pixi.lock` file should cover exactly the platforms listed
in `pixi.toml`; do not commit lock-file changes for `osx-64`, `osx-arm64`,
`win-64`, or another platform unless that platform has been added deliberately
and verified with the same commands. macOS and Windows users should use the pip
workflow unless a PR explicitly adds and verifies Pixi support for those
platforms.

The default Pixi environment intentionally excludes optional external
optimization solver stacks and licensed solver bindings. Keep GAMS, BARON,
IPOPT, Gurobi, HiGHS, and similar tools in optional local environments,
benchmark profiles, or documentation unless they become required for default
imports and tests.

For direct Gurobi access through Pyomo's Python interfaces, use the optional
Pixi environment with `gurobipy` and point Gurobi at a valid license:

```bash
export GRB_LICENSE_FILE=/path/to/gurobi.lic
pixi install -e gurobi
pixi run -e gurobi python -c "import gurobipy as gp; from pyomo.environ import SolverFactory; print(gp.gurobi.version()); print(SolverFactory('gurobi_direct').available(False))"
```

Pyomo's direct Gurobi interfaces support many LP, MIP, and quadratic workflows.
For nonlinear transformed GDP models that Pyomo cannot write directly to
Gurobi, use the documented GAMS/Gurobi benchmark profile.

### PyPI Release Workflow

PyPI releases are published by
[`.github/workflows/publish.yml`](./.github/workflows/publish.yml) when a
GitHub Release is published. Package versions come from git tags through
`setuptools_scm`, so create the release from the version tag intended for PyPI.
The build backend remains `setuptools.build_meta` as configured in
`pyproject.toml`, and package maintainer metadata remains defined there.

The publish workflow checks out full git history, builds both the sdist and
wheel with `python -m build`, validates them with
`python -m twine check dist/*`, installs the wheel on Python 3.10, 3.11, and
3.12, and then publishes with PyPI Trusted Publishing through the protected
`pypi` GitHub environment. The PyPI project must have a trusted publisher entry
for this repository, the `publish.yml` workflow, and the `pypi` environment
before publishing the first release.

Before publishing a GitHub Release, run the normal local checks:

```bash
pixi run test
pixi run lint
```

If local artifact validation is needed, install the PyPA build tools in the
active environment and run:

```bash
python -m build
python -m twine check dist/*
```

The committed Pixi environment remains `linux-64` only. Do not add Pixi
platforms for release work unless the lock file is regenerated and the selected
platforms are verified according to the development policy above.

### Benchmark Campaigns

The benchmark runner can preflight the PR #58 benchmark campaign before starting
long solver-backed runs. Preflight checks the selected strategy plugins, solver
interface, GAMS executable when applicable, and model construction.

The default profile is a local nonlinear GAMS profile that uses DICOPT for
transformed and local MINLP roles, IPOPTH for NLP roles, and Gurobi for MIP
roles. This is the recommended first pass before launching a global GAMS/BARON
run. Use `--solver-profile gams-gurobi` for a GAMS/Gurobi pass, or
`--solver-profile gams-baron` for a global BARON pass.

GAMS-backed benchmark solves use `option optcr=1e-6` by default so incumbent
solutions and solver bounds are compared at a consistent relative gap.

```bash
pixi run gdplib-benchmark preflight
```

To run the PR #58 campaign with the local nonlinear GAMS profile and 1-hour
per-case time limits:

```bash
pixi run gdplib-benchmark run --cases-file benchmark_cases/pr58_local.csv --run-id pr58_local
```

For row-by-row control over instances, methods, and subsolvers, pass a case
file:

```bash
pixi run gdplib-benchmark run --cases-file benchmark_cases.csv --run-id pr58_cases
```

CSV columns may include `instance`, `strategy`, `timelimit`, `solver_profile`,
`subsolver`, `gams_solver`, `gams_nlp_solver`, `gams_mip_solver`,
`gams_minlp_solver`, `gams_local_minlp_solver`, and `label`.

After the local nonlinear run has identified construction, transformation, and
local solver issues, run the global GAMS/BARON profile explicitly:

```bash
pixi run gdplib-benchmark run --solver-profile gams-baron --run-id pr58_global
```

To capture Pyomo/Python warnings from the same model set without starting
solver-backed benchmark jobs:

```bash
pixi run gdplib-benchmark warnings --run-id pr58_warnings
```

Benchmark outputs are generated under `gdplib/<model>/benchmark_result/<run-id>/`.
Aggregate run metadata, strict JSON summaries, and failure manifests are written
under `benchmark_runs/<run-id>/`. These generated outputs are ignored by git.

For a pip-based setup:

1. **Install development dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   pip install -e .
   ```

2. **Set up GitHub Copilot with custom instructions:**
   - The repository includes custom Copilot instructions in `.github/copilot-instructions.md`
   - Project-specific configurations are available in `.copilot/`

3. **Run tests:**
   ```bash
   pytest tests/ -v --tb=short
   ```

4. **Code formatting and linting:**
   ```bash
   black -S -C --target-version py310 --check --diff .
   flake8 gdplib/ --count --select=E9,F63,F7,F82 --show-source --statistics
   flake8 gdplib/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
   ```

   The `typos` CLI used by CI is included in the Pixi environment. Install it separately before running the spell check from a pip-only environment:

   ```bash
   typos --config ./.github/workflows/typos.toml
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
