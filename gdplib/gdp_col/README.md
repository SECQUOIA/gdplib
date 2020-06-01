# GDP Distillation Column Design

This is a distillation column design problem to determine the optimal number of trays and feed stage.

Source paper:

> Ghouse, J. H., Chen, Q., Zamarripa, M. A., Lee, A., Burgard, A. P., Grossmann, I. E., & Miller, D. C. (2018). A comparative study between GDP and NLP formulations for conceptual design of distillation columns. *Computer Aided Chemical Engineering*, 44(1), 865–870. https://doi.org/10.1016/B978-0-444-64241-7.50139-7

## Problem Details

### Solution

Best known objective value: 19,430

### Size

| Problem   | vars | Bool | bin | int | cont | cons | nl | disj | disjtn |
|-----------|------|------|-----|-----|------|------|----|------|--------|
| syngas | 433 | 28 | 0 | 0 | 405 | 603 | 255 | 28 | 15 |

- ``vars``: variables
- ``Bool``: Boolean variables
- ``bin``: binary variables
- ``int``: integer variables
- ``cont``: continuous variables
- ``cons``: constraints
- ``nl``: nonlinear constraints
- ``disj``: disjuncts
- ``disjtn``: disjunctions
