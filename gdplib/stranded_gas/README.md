# Stranded gas production network

This is a multi-period production planning, allocation, and design model involving modular facilities for production of gasoline from stranded gas sources.

Source paper:

> Chen, Q., & Grossmann, I. E. (2019). Economies of Numbers for A Modular Stranded Gas Processing Network: Modeling And Optimization. In *Proceedings of the 9th International Conference on Foundations of Computer-Aided Process Design* (pp. 257–262). https://doi.org/10.1016/B978-0-12-818597-1.50041-2

Five model variants are described below, generated by introduction of additional constraints to the original GDP model.
- ``Gas_100``: Use only the U100 module size
- ``Gas_250``: Use only the U250 module size
- ``Gas_500``: Use only the U500 module size
- ``Gas_small``: Use the U100 or U250 module sizes
- ``Gas_large``: Use the U250 or U500 module sizes

## Problem Details

### Solution

Best known objective values:
- ``Gas_100``: -12.34
- ``Gas_250``: -18.37
- ``Gas_500``: -4.69
- ``Gas_small``: -18.37
- ``Gas_large``: -18.37

### Size

| Problem   | vars | Bool | bin | int | cont | cons | nl | disj | disjtn |
|-----------|------|------|-----|-----|------|------|----|------|--------|
| Gas_100 | 8816 | 158 | 0 | 2520 | 6138 | 14925 | 1 | 158 | 96 |
| Gas_250 | 8816 | 158 | 0 | 2520 | 6138 | 14925 | 1 | 158 | 96 |
| Gas_500 | 8816 | 158 | 0 | 2520 | 6138 | 14925 | 1 | 158 | 96 |
| Gas_small | 11698 | 160 | 0 | 5040 | 6498 | 14927 | 2 | 160 | 96 |
| Gas_large | 11698 | 160 | 0 | 5040 | 6498 | 14927 | 2 | 160 | 96 |

- ``vars``: variables
- ``Bool``: Boolean variables
- ``bin``: binary variables
- ``int``: integer variables
- ``cont``: continuous variables
- ``cons``: constraints
- ``nl``: nonlinear constraints
- ``disj``: disjuncts
- ``disjtn``: disjunctions