# membrane — Optimal Membrane Cascade Design for Critical Mineral Recovery

This model describes the optimal design of a multistage nanofiltration membrane cascade
for separating Lithium (Li) and Cobalt (Co) from battery recycling streams.

The superstructure consists of K cascade stages, each discretized into N membrane elements
linked in series on both the retentate and permeate sides. The GDP formulation decides
where to inject the feed and diafiltrate streams, and where to install inter-stage reflux
connections, to maximize Cobalt recovery while satisfying a minimum Lithium recovery
constraint.

Each membrane element is governed by overall and component mass balances, a solvent flux
equation, and a sieving coefficient relationship. Disjunctions are defined over all
(stage, element) locations for feed injection, diafiltrate injection, and reflux
installation. Logical constraints enforce that exactly one feed and one diafiltrate are
placed across the entire cascade, and exactly one reflux per intermediate stage.

```
                              ---------------
                              |             |
feed + diaf + refl -->        |             |
                  --> rin --> |   element   | --> rout
              fin -->         |             |
                              |             |
                              ---------------
                              |             |
                  --> pin --> |             | --> pout
                              |             |
                              ---------------
```

Source paper:

> Ovalle, D., Tran, N., Laird, C. D., & Grossmann, I. E. (2024). Optimal Membrane Cascade
> Design for Critical Mineral Recovery Through Logic-based Superstructure Optimization.
> *Systems and Control Transactions*, 3, 853–859. https://doi.org/10.69997/sct.127917

## Problem Details

### Solution

The objective is to maximize Cobalt mass flow in the retentate of the first stage outlet,
subject to 90% Lithium recovery. Optimal objective values for various cascade
configurations (K stages, N elements per stage) are reported in the reference above.

### Size

Problem size for the default instance with K=3 stages and N=3 elements per stage:

| Problem  | vars | Bool | bin | int | cont | cons | nl | disj | disjtn |
|----------|------|------|-----|-----|------|------|----|------|--------|
| membrane |  294 |   48 |   0 |   0 |  246 |  317 | 55 |   48 |     24 |

- ``vars``: variables
- ``Bool``: Boolean variables (disjunct indicator variables)
- ``bin``: binary variables
- ``int``: integer variables
- ``cont``: continuous variables
- ``cons``: constraints
- ``nl``: nonlinear constraints
- ``disj``: disjuncts
- ``disjtn``: disjunctions

The model scales with K and N: increasing stages or discretization elements adds
variables and constraints proportionally.
