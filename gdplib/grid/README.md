# grid — Event-Constrained Optimal Power Flow on the IEEE 14-Bus Network

This model describes the optimal capacity expansion design of the IEEE 14-bus power
distribution network under stochastic nodal demands. The objective is to minimize the
total slack added to generator and transmission line capacities so that power balance
constraints are satisfied across a set of sampled demand scenarios.

The key modeling contribution is an event constraint: rather than requiring all
generators and lines to stay within their capacity limits in every scenario (a joint
chance constraint), the formulation uses ATLEAST logic to require that at least
`active_gens` generators and `active_lines` transmission lines simultaneously satisfy
their capacity bounds. This event must hold in at least 90% of scenarios (alpha=0.9),
approximated via Sample Average Approximation. The GDP formulation encodes whether each
capacity constraint is satisfied or violated per scenario using disjunctions, and
connects them to a per-scenario Boolean event variable via a logical equivalence.

Source paper:

> Ovalle, D., Mazzadi, S., Laird, C. D., Grossmann, I. E., & Pulsipher, J. L. (2025).
> Event constrained programming. *Computers & Chemical Engineering*, 199, 109145.
> https://doi.org/10.1016/j.compchemeng.2025.109145

## Problem Details

### Solution

The objective is to minimize total generator and line capacity slack subject to the
event constraint at alpha=0.9. Optimal objective values for various ATLEAST
configurations are reported in the reference above.

### Size

Problem size for the default instance with `active_gens=4`, `active_lines=20`, and
`num_samples=500`:

| Problem | vars  | Bool  | bin | int |  cont | cons  | nl | disj  | disjtn |
|---------|-------|-------|-----|-----|-------|-------|----|-------|--------|
| grid    | 47525 | 35000 |   0 |   0 | 12525 | 52000 |  0 | 35000 |  12500 |

- ``vars``: variables
- ``Bool``: Boolean variables (disjunct indicator variables)
- ``bin``: binary variables
- ``int``: integer variables
- ``cont``: continuous variables
- ``cons``: constraints
- ``nl``: nonlinear constraints
- ``disj``: disjuncts
- ``disjtn``: disjunctions

The model scales linearly with `num_samples`: each additional scenario adds variables
and constraints proportionally to the number of generators and lines.
