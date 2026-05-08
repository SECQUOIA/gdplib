# GDP Reactor Series Design
Function that builds CSTR superstructure model of size NT (default = 5). 
NT is the number of reactors in series.
The CSTRs have a single 1st order auto catalytic reaction A -> B and minimizes total reactors series volume. 
The optimal solution should yield NT reactors with a recycle before reactor NT.

Reference:
> Linan, D. A., Bernal, D. E., Gomez, J. M., & Ricardez-Sandoval, L. A. (2020). Optimal design of superstructures for placing units and streams with multiple and ordered available locations. Part I: A new mathematical framework. Computers & Chemical Engineering, 137, 106794.
https://doi.org/10.1016/j.compchemeng.2020.106794
 
### Solution

Best known objective value: 3.06181298849707

### Size

Number of reactors in series is 5.

| Problem   | vars | Bool | bin | int | cont | cons | nl | disj | disjtn |
|-----------|------|------|-----|-----|------|------|----|------|--------|
| gdp_reactors | 71 | 15 | 0 | 0 | 56 | 25 | 2 | 20 | 10 |

- ``vars``: variables
- ``Bool``: Boolean variables
- ``bin``: binary variables
- ``int``: integer variables
- ``cont``: continuous variables
- ``cons``: constraints
- ``nl``: nonlinear constraints
- ``disj``: disjuncts
- ``disjtn``: disjunctions