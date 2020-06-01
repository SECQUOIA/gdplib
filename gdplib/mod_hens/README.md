# Modular heat exchanger network synthesis

This is a set of two-stage heat exchanger network synthesis model variants based on the [(Yee & Grossmann, 1990)](#references) paper.
The base model was adapted to allow for design of heat exchanger network with modular units.

Source paper (see Section "Modular HENS Case Study"):

> Qi Chen & I.E. Grossmann (2019). Effective Generalized Disjunctive Programming Models for Modular Process Synthesis. *Industrial & Engineering Chemistry Research*, 58(15), 5873–5886. https://doi.org/10.1021/acs.iecr.8b04600

The problem involves two hot and two cold process streams, as well as steam and cooling water utilities.
The objective is minimization of total annualized cost (TAC).

Seven model variants are described:

- ``build_conventional`` - Conventional HENS
- ``build_integer_single_module`` - Modular HENS - single module type allowed, integer formulation
- ``build_integer_require_modular`` - Modular HENS - multiple module types allowed, integer formulation
- ``build_integer_modular_option`` - Modular HENS - mixed modular and conventional exchangers allowed, integer formulation
- ``build_discrete_single_module`` - Modular HENS - single module type allowed, discretized formulation
- ``build_discrete_require_modular`` - Modular HENS - multiple module types allowed, discretized formulation
- ``build_discrete_modular_option`` - Modular HENS - mixed modular and conventional exchangers allowed, discretized formulation

The discretized formulations use the ``induced linearity`` reformulation described in the (Chen & Grossmann, 2019) source paper.

## Problem Details

### Solution

Best known objective values:
- ``build_conventional``: 106767 (optimal)
- ``build_integer_single_module``, ``build_discrete_single_module``: 134522
- ``build_integer_require_modular``, ``build_discrete_require_modular``: 111520
- ``build_integer_modular_option``, ``build_discrete_modular_option``: 101505

### Size

| Problem   | vars | Bool | bin | int | cont | cons | nl | disj | disjtn |
|-----------|------|------|-----|-----|------|------|----|------|--------|
| conv      | 214  | 24   | 0   | 0   | 190  | 250  | 36 | 24   | 32     |
| int_sing  | 313  | 27   | 0   | 96  | 190 | 265  |  24  |  27  |       45 |
| int_req | 313 | 27 | 0 | 96 | 190 | 265 | 24 | 27 | 45 |
| int_opt | 274 | 48 | 0 | 36 | 190 | 322 | 36 | 48 | 44 |
| disc_sing | 3077 | 27 | 2080 | 0 | 970 | 6006 | 12 | 27 | 33 |
| disc_req | 1114 | 24 | 300 | 0 | 790 | 1486 | 12 | 24 | 44 |
| disc_opt | 1138 | 48 | 300 | 0 | 790 | 2122 | 36 | 48 | 44 |

- ``vars``: variables
- ``Bool``: Boolean variables
- ``bin``: binary variables
- ``int``: integer variables
- ``cont``: continuous variables
- ``cons``: constraints
- ``nl``: nonlinear constraints
- ``disj``: disjuncts
- ``disjtn``: disjunctions

## References

> Yee, T. F., & Grossmann, I. E. (1990). Simultaneous optimization models for heat integration—II. Heat exchanger network synthesis. Computers & Chemical Engineering, 14(10), 1165–1184. https://doi.org/10.1016/0098-1354(90)85010-8
