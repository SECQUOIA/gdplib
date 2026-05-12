# pandemic — Optimal Disease Control via Event-Constrained SEIR Model

This model describes the optimal control of quarantine measures to mitigate the spread
of an infectious disease with minimal intervention. The objective is to minimize the
total quarantine control effort u(t) over a time horizon [0, 200] days while ensuring
the fraction of infected individuals i(t) remains below a healthcare capacity threshold
i_max = 0.02 for at least 90% of the time horizon.

The SEIR (Susceptible–Exposed–Infectious–Recovered) dynamics are discretized via
backward finite differences over a non-uniform time grid. The grid combines `num_times`
equidistant points on [0, 200] with 10 additional fine-grained points near t=0
(0.001, 0.002, 0.004, 0.008, 0.02, 0.04, 0.08, 0.2, 0.4, 0.8) to resolve the sharp
early-time SEIR dynamics. An event constraint
with alpha = 0.9 enforces that the capacity constraint i(t) ≤ i_max holds for at least
90% of the time horizon, approximated by trapezoidal integration of disjunct indicator
variables. The GDP formulation uses disjunctions to encode whether the constraint is
satisfied or violated at each time point.

Source paper:

> Ovalle, D., Mazzadi, S., Laird, C. D., Grossmann, I. E., & Pulsipher, J. L. (2025).
> Event constrained programming. *Computers & Chemical Engineering*, 199, 109145.
> https://doi.org/10.1016/j.compchemeng.2025.109145

## Problem Details

### Solution

The objective is to minimize total quarantine intervention subject to the event
constraint at alpha = 0.9. Optimal objective values for various alpha levels are
reported in the reference above.

### Size

Problem size for the default instance with `num_times=101`:

| Problem  | vars | Bool | bin | int | cont | cons | nl  | disj | disjtn |
|----------|------|------|-----|-----|------|------|-----|------|--------|
| pandemic | 1221 |  222 |   0 |   0 |  999 | 1107 | 220 |  222 |    111 |

- ``vars``: variables
- ``Bool``: Boolean variables (disjunct indicator variables)
- ``bin``: binary variables
- ``int``: integer variables
- ``cont``: continuous variables
- ``cons``: constraints
- ``nl``: nonlinear constraints
- ``disj``: disjuncts
- ``disjtn``: disjunctions

The model scales linearly with `num_times`: each additional time point adds variables
and constraints proportionally to the number of SEIR states.
