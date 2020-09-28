# hda - Synthesis: Hydrodealkylation of Toluene

This model describes the profit maximization of a Hydrodealkylation of Toluene process first presented in:
> James M Douglas (1988). Conceptual Design of Chemical Processes, McGraw-Hill. ISBN-13: 978-0070177628

Later implemented as a GDP in:

> G.R. Kocis, and I.E. Grossmann (1989). Computational Experience with DICOPT Solving Minlp Problems in Process Synthesis. Computers and Chemical Engineering 13, 3, 307-315. https://doi.org/10.1016/0098-1354(89)85008-2

The MINLP formulation of this problem is available in GAMS https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_hda.html

This model was reimplemented by Yunshan Liu @Yunshan-Liu

## Problem Details
### Solution

Best known objective value: 5966.51


### Size

| Problem   | vars | Bool | bin | int | cont | cons | nl | disj | disjtn |
|-----------|------|------|-----|-----|------|------|----|------|--------|
| Kaibel Column | 733 | 12 | 0 | 0 | 721 | 728 | 151 | 12 | 6 |

- ``vars``: variables
- ``Bool``: Boolean variables
- ``bin``: binary variables
- ``int``: integer variables
- ``cont``: continuous variables
- ``cons``: constraints
- ``nl``: nonlinear constraints
- ``disj``: disjuncts
- ``disjtn``: disjunctions

