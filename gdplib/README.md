## Water Network Design

In the Water Treatment Network (WTN) design problem, given is a set of water streams with known concentrations of contaminants and flow rate.
The objective is to find the set of treatment units and interconnections that minimize the cost of the WTN while satisfying maximum concentrations of contaminants in the reclaimed outlet stream.
The WTN superstructure consists of a set of treatment units, contaminated feed streams carrying a set of contaminants, and a discharge unit.
The fouled feed waters can be allocated to one or more treatment units or disposed of in the sink unit. Upon treatment, the reclaimed streams can be recycled, forwarded to other treatment units, or discharged into the sink unit.

The mass balances are defined in terms of total flows and contaminants concentration.
Nonconvexities arise from bilinear terms “flows times concentration” in the mixers mass balances and concave investment cost functions of treatment units.

The instance incorporates two approximations of the concave cost term (piecewise linear and quadratic) to reformulate the GDP model into a bilinear quadratic one.

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
The contaminant concentration and flow rate of the feed streams, contaminant recovery rates, minimum flow rate and cost coefficients of the treatment units, and the upper limit on the molar flow of contaminant in the purified stream, are reported in [1].

### Solution

Best known objective value: 348,340

### Size

Number of reactors in series is 5.

| Problem           | vars | Bool | bin | int | cont | cons | nl | disj | disjtn |
| ----------------- | ---- | ---- | --- | --- | ---- | ---- | -- | ---- | ------ |
| `water_network` |      |      |     |     |      |      |    |      |        |

- ``vars``: variables
- ``Bool``: Boolean variables
- ``bin``: binary variables
- ``int``: integer variables
- ``cont``: continuous variables
- ``cons``: constraints
- ``nl``: nonlinear constraints
- ``disj``: disjuncts
- ``disjtn``: disjunctions

### References

> [1] Ruiz J., Grossmann I. E. Water Treatment Network Design. 2009 Available from CyberInfrastructure for [MINLP](www.minlp.org), a collaboration of Carnegie Mellon University and IBM at: www.minlp.org/library/problem/index.php?i=24
>
> [2] Ruiz, J., & Grossmann, I. E. (2011). Using redundancy to strengthen the relaxation for the global optimization of MINLP problems. Computers & Chemical Engineering, 35(12), 2729–2740. https://doi.org/10.1016/J.COMPCHEMENG.2011.01.035