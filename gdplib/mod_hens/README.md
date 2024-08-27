# Modular heat exchanger network synthesis

This is a set of two-stage heat exchanger network synthesis model variants based on the [(Yee & Grossmann, 1990)](#references) paper.
The base model was adapted to allow for design of heat exchanger network with modular units.

Source paper (see Section "Modular HENS Case Study"):

> Qi Chen & I.E. Grossmann (2019). Effective Generalized Disjunctive Programming Models for Modular Process Synthesis. *Industrial & Engineering Chemistry Research*, 58(15), 5873–5886. https://doi.org/10.1021/acs.iecr.8b04600

The problem involves two hot and two cold process streams, as well as steam and cooling water utilities.
The objective is minimization of total annualized cost (TAC).

Seven model variants are described:

- ``build_model('conventional')`` - Conventional HENS
- ``build_model('single_module_integer')`` - Modular HENS - single module type allowed, integer formulation
- ``build_model('multiple_module_integer')`` - Modular HENS - multiple module types allowed, integer formulation
- ``build_model('mixed_integer')`` - Modular HENS - mixed modular and conventional exchangers allowed, integer formulation
- ``build_model('single_module_discrete')`` - Modular HENS - single module type allowed, discretized formulation
- ``build_model('multiple_module_discrete')`` - Modular HENS - multiple module types allowed, discretized formulation
- ``build_model('mixed_discrete')`` - Modular HENS - mixed modular and conventional exchangers allowed, discretized formulation

The discretized formulations use the ``induced linearity`` reformulation described in the (Chen & Grossmann, 2019) source paper.

## Problem Details

### Solution

Best known objective values:
- ``conventional``: 106767 (optimal)
- ``single_module_integer``, ``single_module_discrete``: 134522
- ``multiple_module_integer``, ``multiple_module_discrete``: 111520
- ``mixed_integer``, ``mixed_discrete``: 101505

### Size

| Component             |   conventional |   single_module_integer |   multiple_module_integer |   mixed_integer |   single_module_discrete |   multiple_module_discrete |   mixed_discrete |
|:----------------------|---------------:|------------------------:|--------------------------:|----------------:|-------------------------:|---------------------------:|-----------------:|
| Variables             |            338 |                     501 |                       498 |             498 |                     4761 |                       3802 |             3802 |
| Binary variables      |             64 |                     131 |                       128 |             128 |                     2147 |                       1728 |             1728 |
| Integer variables     |              0 |                      96 |                        96 |              96 |                        0 |                          0 |                0 |
| Continuous variables  |            274 |                     274 |                       274 |             274 |                     2614 |                       2074 |             2074 |
| Disjunctions          |             32 |                      65 |                        64 |              64 |                       33 |                         64 |               64 |
| Disjuncts             |             64 |                     131 |                       128 |             128 |                       67 |                        128 |              128 |
| Constraints           |            370 |                     565 |                       562 |             562 |                     8786 |                       5362 |             5362 |
| Nonlinear constraints |             96 |                      96 |                        96 |              96 |                       32 |                         96 |               96 |

## References

> Yee, T. F., & Grossmann, I. E. (1990). Simultaneous optimization models for heat integration—II. Heat exchanger network synthesis. Computers & Chemical Engineering, 14(10), 1165–1184. https://doi.org/10.1016/0098-1354(90)85010-8
