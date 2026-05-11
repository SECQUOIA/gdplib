# Modular Production Network

This is a set of models based on a multi-period production planning, allocation, and design problem involving modular facilities.

Source paper (see Section "Capacity Expansion Case Study"):

> Qi Chen & I.E. Grossmann (2019). Effective Generalized Disjunctive Programming Models for Modular Process Synthesis. *Industrial & Engineering Chemistry Research*, 58(15), 5873–5886. https://doi.org/10.1021/acs.iecr.8b04600


Five model variants are described:

- ``build_model('Growth')`` - Single site capacity expansion (Growth scenario)
- ``build_model('Dip')`` - Single site capacity expansion (Dip scenario)
- ``build_model('Decay')`` - Single site capacity expansion (Decay scenario)
- ``build_model('Distributed')`` - Multi-site distributed design (monthly time periods)
- ``build_model('QuarterDistributed')`` - Multi-site distributed design (quarterly time periods)


## Problem Details

### Solution

Best known objective values:
- ``Growth``: 3593 (optimal)
- ``Dip``: 2096 (optimal)
- ``Decay``: 851 (optimal)
- ``Distributed``: 36262
- ``QuarterDistributed``: 19568

### Size

| Component             |   Growth |   Dip |   Decay |   Distributed |   QuarterDistributed |
|:----------------------|---------:|------:|--------:|--------------:|---------------------:|
| Variables             |      488 |   488 |     488 |          3720 |                 1320 |
| Binary variables      |        2 |     2 |       2 |            42 |                   42 |
| Integer variables     |      363 |   363 |     363 |          1452 |                  492 |
| Continuous variables  |      123 |   123 |     123 |          2226 |                  786 |
| Disjunctions          |        1 |     1 |       1 |            21 |                   21 |
| Disjuncts             |        2 |     2 |       2 |            42 |                   42 |
| Constraints           |      486 |   486 |     486 |          1792 |                  672 |
| Nonlinear constraints |        1 |     1 |       1 |            36 |                   36 |