# Batch Processing Optimization Problem

The model is designed to minimize the total cost associated with the design and operation of a plant consisting of multiple parallel processing units with intermediate storage tanks. It involves determining the optimal number and sizes of processing units, batch sizes for different products at various stages, and sizes and placements of storage tanks to ensure operational efficiency while meeting production requirements within a specified time horizon.

## Problem Details
### Optimal Solution

TODO

### Size

| Component             |   Number |
|:----------------------|---------:|
| Variables             |      288 |
| Binary variables      |      138 |
| Integer variables     |        0 |
| Continuous variables  |      150 |
| Disjunctions          |        9 |
| Disjuncts             |       18 |
| Constraints           |      601 |
| Nonlinear constraints |        1 |


## References
> [1] Ravemark, E. Optimization models for design and operation of chemical batch processes. Ph.D. Thesis, ETH Zurich, 1995. https://doi.org/10.3929/ethz-a-001591449
> 
> [2] Vecchietti, A., & Grossmann, I. E. (1999). LOGMIP: a disjunctive 0–1 non-linear optimizer for process system models. Computers & chemical engineering, 23(4-5), 555-565. https://doi.org/10.1016/S0098-1354(97)87539-4