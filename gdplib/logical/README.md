# Logical expression system demo problems

``positioning`` and ``spectralog`` are demonstration problems for the Pyomo.GDP logical expression system, adapted from their equivalents in LOGMIP.

## Spectralog

Source paper (Example 2):

> Vecchietti, A., & Grossmann, I. E. (1999). LOGMIP: A disjunctive 0-1 non-linear optimizer for process system models. *Computers and Chemical Engineering*, 23(4–5), 555–565. https://doi.org/10.1016/S0098-1354(98)00293-2

### Problem Details

#### Solution

Optimal objective value: 12.0893

#### Size
- Variables: 128
    - Boolean: 60
    - Binary: 0
    - Integer: 0
    - Continuous: 68
- Constraints: 158
    - Nonlinear: 8
- Disjuncts: 60
- Disjunctions: 30

## Optimal positioning

Source paper (Example 4):

> Duran, M. A., & Grossmann, I. E. (1986). An outer-approximation algorithm for a class of mixed-integer nonlinear programs. *Mathematical Programming*, 36(3), 307. https://doi.org/10.1007/BF02592064

### Problem Details

#### Solution

Optimal objective value: -8.06

#### Size
- Variables: 56
    - Boolean: 50
    - Binary: 0
    - Integer: 0
    - Continuous: 6
- Constraints: 30
    - Nonlinear: 25
- Disjuncts: 50
- Disjunctions: 25
