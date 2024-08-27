# GDP Distillation Column Design

This model solves a jobshop scheduling, which has a set of jobs which must be processed in sequence of stages but not all jobs require all stages. A zero wait transfer policy is assumed between stages. To obtain a feasible solution it is necessary to eliminate all clashes between jobs. It requires that no two jobs be performed at any stage at any time. The objective is to minimize the makespan, the time to complete all jobs.

## Problem Details

### Optimal Solution

### Size

| Component             |   Number |
|:----------------------|---------:|
| Variables             |       10 |
| Binary variables      |        6 |
| Integer variables     |        0 |
| Continuous variables  |        4 |
| Disjunctions          |        3 |
| Disjuncts             |        6 |
| Constraints           |        9 |
| Nonlinear constraints |        0 |

## References

> [1] Raman & Grossmann, Modelling and computational techniques for logic based integer programming, Computers and Chemical Engineering 18, 7, p.563-578, 1994. DOI: 10.1016/0098-1354(93)E0010-7.
> [2] Aldo Vecchietti, LogMIP User's Manual, http://www.logmip.ceride.gov.ar/, 2007