# GDP Reactor Series Design
Function that builds CSTR superstructure model of size NT (default = 5). 
NT is the number of reactors in series.
The CSTRs have a single 1st order auto catalytic reaction A -> B and minimizes total reactors series volume. 
The optimal solution should yield NT reactors with a recycle before reactor NT.

Reference:
> Linan, D. A., Bernal, D. E., Gomez, J. M., & Ricardez-Sandoval, L. A. (2021). Optimal synthesis and design of catalytic distillation columns: A rate-based modeling approach. Chemical Engineering Science, 231, 116294. https://doi.org/10.1016/j.ces.2020.116294

## Problem Details

### Optimal Solution

Best known objective value: 3.06181298849707

### Size

Number of reactors in series is 5.

| Component             |   Number |
|:----------------------|---------:|
| Variables             |       76 |
| Binary variables      |       20 |
| Integer variables     |        0 |
| Continuous variables  |       56 |
| Disjunctions          |       10 |
| Disjuncts             |       20 |
| Constraints           |      100 |
| Nonlinear constraints |       17 |