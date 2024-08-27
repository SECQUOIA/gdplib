# Small Batch Scheduling Problem

The gdp_small_batch.py module contains the GDP model for the small batch problem based on the Kocis and Grossmann (1988) paper.

The problem is based on the Example 4 of the paper.

The objective is to minimize the investment cost of the batch units.

## Problem Details

### Optimal Solution

The optimal solution is `167427.65711`.

### Size

| Component             |   Number |
|:----------------------|---------:|
| Variables             |       37 |
| Binary variables      |       18 |
| Integer variables     |        0 |
| Continuous variables  |       19 |
| Disjunctions          |        9 |
| Disjuncts             |       18 |
| Constraints           |       34 |
| Nonlinear constraints |        1 |

## References

> [1] Kocis, G. R.; Grossmann, I. E. Global Optimization of Nonconvex Mixed-Integer Nonlinear Programming (MINLP) Problems in Process Synthesis. Ind. Eng. Chem. Res. 1988, 27 (8), 1407-1421. https://doi.org/10.1021/ie00080a013
>
> [2] Ovalle, D., Liñán, D. A., Lee, A., Gómez, J. M., Ricardez-Sandoval, L., Grossmann, I. E., & Bernal Neira, D. E. (2024). Logic-Based Discrete-Steepest Descent: A Solution Method for Process Synthesis Generalized Disjunctive Programs. arXiv preprint arXiv:2405.05358. https://doi.org/10.48550/arXiv.2405.05358
