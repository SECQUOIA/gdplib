# Modular Production Network

This is a set of models based on a multi-period production planning, allocation, and design problem involving modular facilities.

Source paper (see Section "Capacity Expansion Case Study"):

> Qi Chen & I.E. Grossmann (2019). Effective Generalized Disjunctive Programming Models for Modular Process Synthesis. *Industrial & Engineering Chemistry Research*, 58(15), 5873–5886. https://doi.org/10.1021/acs.iecr.8b04600


Five model variants are described:

- ``build_cap_expand_growth`` - Single site capacity expansion (Growth scenario)
- ``build_cap_expand_dip`` - Single site capacity expansion (Dip scenario)
- ``build_cap_expand_decay`` - Single site capacity expansion (Decay scenario)
- ``build_distributed_model`` - Multi-site distributed design (monthly time periods)
- ``build_quarter_distributed_model`` - Multi-site distributed design (quarterly time periods)


## Problem Details

### Solution

Best known objective values:
- ``build_cap_expand_growth``: 3593 (optimal)
- ``build_cap_expand_dip``: 2096 (optimal)
- ``build_cap_expand_decay``: 851 (optimal)
- ``build_distributed_model``: 36262
- ``build_quarter_distributed_model``: 19568

### Size

| Problem   | vars | Bool | bin | int | cont | cons | nl | disj | disjtn |
|-----------|------|------|-----|-----|------|------|----|------|--------|
| Mod_grow | 488 | 2 | 0 | 363 | 123 | 486 | 1 | 2 | 1 |
| Mod_dip | 488 | 2 | 0 | 363 | 123 | 486 | 1 | 2 | 1 |
| Mod_decay | 488 | 2 | 0 | 363 | 123 | 486 | 1 | 2 | 1 |
| Mod_dist | 2224 | 26 | 0 | 718 | 1480 | 1387 | 22 | 26 | 13 |
| Mod_qtr | 1314 | 42 | 0 | 486 | 786 | 672 | 36 | 42 | 21 |


- ``vars``: variables
- ``Bool``: Boolean variables
- ``bin``: binary variables
- ``int``: integer variables
- ``cont``: continuous variables
- ``cons``: constraints
- ``nl``: nonlinear constraints
- ``disj``: disjuncts
- ``disjtn``: disjunctions
