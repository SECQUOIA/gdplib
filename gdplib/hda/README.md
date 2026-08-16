# hda - Synthesis: Hydrodealkylation of Toluene

This model describes the profit maximization of a Hydrodealkylation of Toluene process first presented in:
> James M Douglas (1988). Conceptual Design of Chemical Processes, McGraw-Hill. ISBN-13: 978-0070177628

Later implemented as a GDP in:

> G.R. Kocis, and I.E. Grossmann (1989). Computational Experience with DICOPT Solving Minlp Problems in Process Synthesis. Computers and Chemical Engineering 13, 3, 307-315. https://doi.org/10.1016/0098-1354(89)85008-2

The MINLP formulation of this problem is available in GAMS https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_hda.html

This model was reimplemented by Yunshan Liu @Yunshan-Liu .

## Problem Details
### Optimal Solution

Best known objective value: 5801.27 (not yet globally certified)

Status after the PR #130 feasibility fix: the model was infeasible as
transformed (an eps1-perturbed log-Antoine equation conflicted with the
Antoine-derived vp bounds); the Antoine relations are now a pole-free
bilinear equation defining a bounded, shifted log vapor pressure plus a
bounded exponential. Bounded global runs (300 s, Big-M) bracket the optimum
in [5671.6, 5822.8], consistent with the recorded 5801.27. A GAMS/DICOPT
solve of the original gamslib `hda` model reports 4322.55, but that is a
local (OA) claim on a nonconvex MINLP, not a global certificate; a verified
feasible point of this port at 4322.513 exists with residuals below 1e-9.


### Size

| Component             |   Number |
|:----------------------|---------:|
| Variables             |     1158 |
| Binary variables      |       12 |
| Integer variables     |        0 |
| Continuous variables  |     1146 |
| Disjunctions          |        6 |
| Disjuncts             |       12 |
| Constraints           |      728 |
| Nonlinear constraints |      151 |