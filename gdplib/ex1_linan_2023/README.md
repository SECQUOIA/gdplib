## Example 1 Problem of Liñán (2023)

The Example 1 Problem of Liñán (2023) is a simple optimization problem that involves two Boolean variables, two continuous variables, and a nonlinear objective function. The problem is formulated as a Generalized Disjunctive Programming (GDP) model.

The Boolean variables are associated with disjuncts that define the feasible regions of the continuous variables. The problem also includes logical constraints that ensure that only one Boolean variable is true at a time. Additionally, there are two disjunctions, one for each Boolean variable, where only one disjunct in each disjunction must be true. A specific logical constraint also enforces that `Y1[3]` must be false, making this particular disjunct infeasible.

The objective function is -0.9995999999999999 when the continuous variables are alpha = 0 (`Y1[2]=True`) and beta=-0.7 (`Y2[3]=True`).

The objective function originates from Problem No. 6 of Gomez's paper, and Liñán introduced logical propositions, logical disjunctions, and the following equations as constraints.

### References

[1] Liñán, D. A., & Ricardez-Sandoval, L. A. (2023). A Benders decomposition framework for the optimization of disjunctive superstructures with ordered discrete decisions. AIChE Journal, 69(5), e18008. https://doi.org/10.1002/aic.18008

[2] Gomez, S., & Levy, A. V. (1982). The tunnelling method for solving the constrained global optimization problem with several non-connected feasible regions. In Numerical Analysis: Proceedings of the Third IIMAS Workshop Held at Cocoyoc, Mexico, January 1981 (pp. 34-47). Springer Berlin Heidelberg. https://doi.org/10.1007/BFb0092958
