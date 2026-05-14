# Optimal positioning

``positioning`` is a demonstration problems for the Pyomo.GDP logical expression system, adapted from its equivalent in LOGMIP.

Source paper (Example 4):

> Duran, M. A., & Grossmann, I. E. (1986). An outer-approximation algorithm for a class of mixed-integer nonlinear programs. *Mathematical Programming*, 36(3), 307. https://doi.org/10.1007/BF02592064

## Problem Details

### Solution

Best known objective value: -8.06 (optimal)

The benchmark `gdpopt.gloa` custom-disjunct initialization marks consumers
1, 6, 8, 15, 17, 20, and 25 as active for the default model instance. In
PR #145, this active set was cross-checked by solving the Big-M reformulation
with GAMS/BARON, which reported optimal termination with objective
-8.0641361647335, `U = 0`, and the same selected consumers. The same active
set initialized `pixi run gdplib-benchmark run --instances positioning
--strategies gdpopt.gloa --timelimit 60 --run-id
issue73_positioning_gloa_custom_init_60s`, which reported optimal termination
with lower bound, upper bound, and objective -8.0641361676497.

### Size

| Component             |   Number |
|:----------------------|---------:|
| Variables             |       56 |
| Binary variables      |       50 |
| Integer variables     |        0 |
| Continuous variables  |        6 |
| Disjunctions          |       25 |
| Disjuncts             |       50 |
| Constraints           |       30 |
| Nonlinear constraints |       25 |
