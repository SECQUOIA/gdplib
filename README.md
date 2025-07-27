# GDPlib

We envision GDPlib as an open library of GDP models to provide examples for prospective modelers, and to provide a benchmarking set for algorithm developers.
We invite contributions to this library from the community, provided under the same BSD-3-clause or compatible license.

## Available Models

The library includes the following models:

- [Batch Processing](./gdplib/batch_processing/): Batch processing optimization model
- [Biofuel](./gdplib/biofuel/): Biofuel production optimization
- [CSTR](./gdplib/cstr/): Continuous Stirred Tank Reactor model
- [Disease Model](./gdplib/disease_model/): Disease spread modeling
- [Ex1 Linan 2023](./gdplib/ex1_linan_2023/): Example from Linan's 2023 paper
- [GDP Column](./gdplib/gdp_col/): GDP Column design optimization
- [HDA](./gdplib/hda/): Hydrodealkylation process model
- [Jobshop](./gdplib/jobshop/): Job shop scheduling optimization
- [Kaibel](./gdplib/kaibel/): Kaibel column design
- [Med Term Purchasing](./gdplib/med_term_purchasing/): Medium-term purchasing optimization
- [Methanol](./gdplib/methanol/): Methanol production process
- [Mod HENS](./gdplib/mod_hens/): Modified Heat Exchanger Network Synthesis
- [ModProdNet](./gdplib/modprodnet/): Modular Production Network
- [Positioning](./gdplib/positioning/): Positioning optimization
- [Small Batch](./gdplib/small_batch/): Small batch processing model
- [SpectraLog](./gdplib/spectralog/): Spectral logging optimization
- [Stranded Gas](./gdplib/stranded_gas/): Stranded gas utilization
- [Syngas](./gdplib/syngas/): Syngas production optimization
- [Water Network](./gdplib/water_network/): Water network design

Each model directory contains its own README.md with detailed model descriptions and specific usage instructions.

## Model Size Comparison

The following table shows the size metrics for all models in GDPlib:

| Component             |   batch_processing |   biofuel |   cstr |   disease_model |   ex1_linan_2023 |   gdp_col |   hda |   jobshop |   med_term_purchasing |   methanol |   modprodnet |   positioning |   small_batch |   spectralog |   stranded_gas |   syngas |   water_network |
|:----------------------|-------------------:|----------:|-------:|----------------:|-----------------:|----------:|------:|----------:|----------------------:|-----------:|-------------:|--------------:|--------------:|-------------:|---------------:|---------:|----------------:|
| variables             |                288 |     36840 |     76 |            1250 |               12 |       442 |  1158 |        10 |                  1165 |        287 |          488 |            56 |            37 |          128 |          57810 |      367 |             395 |
| binary_variables      |                138 |       516 |     20 |              52 |               10 |        30 |    12 |         6 |                   216 |          8 |            2 |            50 |            18 |           60 |            192 |       46 |              10 |
| integer_variables     |                  0 |      4356 |      0 |               0 |                0 |         0 |     0 |         0 |                     0 |          0 |          363 |             0 |             0 |            0 |          45360 |        0 |               0 |
| continuous_variables  |                150 |     31968 |     56 |            1198 |                2 |       412 |  1146 |         4 |                   949 |        279 |          123 |             6 |            19 |           68 |          12258 |      321 |             385 |
| disjunctions          |                  9 |       252 |     10 |              26 |                2 |        15 |     6 |         3 |                    72 |          4 |            1 |            25 |             9 |           30 |             96 |       23 |               5 |
| disjuncts             |                 18 |       516 |     20 |              52 |               10 |        30 |    12 |         6 |                   216 |          8 |            2 |            50 |            18 |           60 |            192 |       46 |              10 |
| constraints           |                601 |     12884 |    100 |             831 |               10 |       610 |   728 |         9 |                   762 |        429 |          486 |            30 |            34 |          158 |          14959 |      543 |             329 |
| nonlinear_constraints |                  1 |        12 |     17 |               0 |                0 |       262 |   151 |         0 |                     0 |         55 |            1 |            25 |             1 |            8 |             18 |       48 |              33 |

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

### Handling Multiple Cases

Many models in GDPlib support multiple cases or configurations. Here are some examples:

1. **Jobshop Scheduling with Different Problem Sizes**:
```python
from gdplib.jobshop import build_model
# Default case
model = build_model()
# Custom case with specific number of jobs and machines
model = build_model(num_jobs=4, num_machines=3)
```

2. **Water Network with Different Configurations**:
```python
from gdplib.water_network import build_model
# Default network configuration
model = build_model()
# Custom configuration with specific parameters
model = build_model(
    num_sources=3,
    num_sinks=4,
    treatment_options=['RO', 'NF', 'UF']
)
```

3. **Batch Processing with Different Products**:
```python
from gdplib.batch_processing import build_model
# Default product mix
model = build_model()
# Custom product mix with specific processing times
model = build_model(
    products=['A', 'B', 'C'],
    processing_times={
        'A': {'mixing': 2, 'reaction': 3, 'separation': 1},
        'B': {'mixing': 1, 'reaction': 4, 'separation': 2},
        'C': {'mixing': 3, 'reaction': 2, 'separation': 2}
    }
)
```

Each model's README.md file contains detailed information about available parameters and their effects on the model behavior.

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
