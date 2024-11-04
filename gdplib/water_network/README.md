## Water Network Design

In the Water Treatment Network (WTN) design problem, given is a set of water streams with known concentrations of contaminants and flow rate.
The objective is to find the set of treatment units and interconnections that minimize the cost of the WTN while satisfying maximum concentrations of contaminants in the reclaimed outlet stream.
The WTN superstructure consists of a set of treatment units, contaminated feed streams carrying a set of contaminants, and a discharge unit.
The fouled feed waters can be allocated to one or more treatment units or disposed of in the sink unit. Upon treatment, the reclaimed streams can be recycled, forwarded to other treatment units, or discharged into the sink unit.

The mass balances are defined in terms of total flows and contaminants concentration.
Nonconvexities arise from bilinear terms “flows times concentration” in the mixers mass balances and concave investment cost functions of treatment units.

The instance incorporates three approximations of the concave cost term when  the treatment unit exist, including one piecewise linear and two quadratic approximations, to reformulate the original GDP model into a bilinear quadratic problem [1].


The quadratic reformulations are as follows:

* `quadratic_zero_origin` which reads as $f(x) = a \ x + b \ x^2$. Absence of flow leads to no costs for the pump, which is what we expect. Optimal solution: $346,654 with a relative error with respect to the best known objective value of 0.5%
* `quadratic_nonzero_origin` takes the form of $f(x) = a + b \ x + b \ x^2$. This constraint leads to a non-zero pump expense in the absence of flow. Optimal solution: $349,521 with a relative error with respect to the best known objective value of 0.3%.

The two quadratic approximations are both effective in capturing pump costs, but `quadratic_nonzero_origin` provides a better fit for the active treatment unit's flowrate range.

The user can create each instance like this:

```
build_model(approximation='none')
build_model(approximation='quadratic_zero_origin')
build_model(approximation='quadratic_nonzero_origin')
build_model(approximation='piecewise')
```

The general model description can be summarized as follows:
Min Cost of Treatment Units
s.t.
Physical Constraints:
(a) Mass balance around each splitter
(b) Mass balance around each mixer
(c) Mass balance around each treatment unit
Performance Constraints:
(d) Contaminant composition of the purified stream less or equal than a given limit
for each contaminant.
Logic Constraints:
(e) Treatment units not chosen have their inlet flow set to zero
(f) Every treatment unit chosen must have a minimum flow

Assumptions:
(i) The performance of the treatment units only depends on the total flow entering the unit and its composition.
(ii) The flow of contaminants leaving the unit is a linear function of the inlet flow of contaminants.

### Case Study

The WTN comprises five inlet streams with four contaminants and four treatment units.
The contaminant concentration and flow rate of the feed streams, contaminant recovery rates, minimum flow rate and cost coefficients of the treatment units, and the upper limit on the molar flow of contaminant in the purified stream, are reported in [2].

### Solution

Best known objective value: $348,340

### Size

| Component             | original | pwl | quadratic |
| :-------------------- | -------- | :--: | :-------: |
| variables             | 395      | 1405 |    420    |
| binary_variables      | 10       | 510 |    10    |
| integer_variables     | 0        |  0  |     0     |
| continuous_variables  | 385      | 895 |    410    |
| disjunctions          | 5        |  5  |     5     |
| disjuncts             | 10       |  10  |    10    |
| constraints           | 329      | 1339 |    334    |
| nonlinear_constraints | 33       |  28  |    33    |

### References

> [1] Tristán C., Fallanza M., Ibáñez R., Grossmann I. E., and Bernal Neira D. E. (2024). Global Optimization via Quadratic Disjunctive Programming for Water Networks Design with Energy Recovery. Computer Aided Chemical Engineering, 53, 2161–2166. https://doi.org/10.1016/B978-0-443-28824-1.50361-6
>
> [2] Ruiz J., and Grossmann I. E. Water Treatment Network Design. 2009 Available from CyberInfrastructure for [MINLP](www.minlp.org), a collaboration of Carnegie Mellon University and IBM at: www.minlp.org/library/problem/index.php?i=24
>
> [3] Ruiz, J., and Grossmann, I. E. (2011). Using redundancy to strengthen the relaxation for the global optimization of MINLP problems. Computers & Chemical Engineering, 35(12), 2729–2740. https://doi.org/10.1016/J.COMPCHEMENG.2011.01.035
