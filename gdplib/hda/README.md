# hda - Synthesis: Hydrodealkylation of Toluene

This model describes the profit maximization of a Hydrodealkylation of Toluene process first presented in:
> James M Douglas (1988). Conceptual Design of Chemical Processes, McGraw-Hill. ISBN-13: 978-0070177628

Later implemented as a GDP in:

> G.R. Kocis, and I.E. Grossmann (1989). Computational Experience with DICOPT Solving Minlp Problems in Process Synthesis. Computers and Chemical Engineering 13, 3, 307-315. https://doi.org/10.1016/0098-1354(89)85008-2

The MINLP formulation of this problem is available in GAMS https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_hda.html

This model was reimplemented by Yunshan Liu @Yunshan-Liu .

## Problem Details
### Optimal Solution

Best known objective value: 5965.85 (verified feasible; not globally certified)

Best known configuration: no H2 feed purification, adiabatic reactor,
hydrogen recycle, methane recycle via membrane, methane stabilizing via
distillation column, toluene recovery via distillation column.

Provenance: complete enumeration of all 64 discrete configurations with
each fixed-configuration NLP solved by POUNCE (pure-Rust Ipopt port,
AMPL NL interface — no GAMS translation layer). 30 configurations solve
to local optimality (max constraint residual ~1e-8); the per-configuration
solves are local, so 5965.85 is a lower bound on the global optimum, not a
certificate. The runner-up configuration (with H2 purification) reaches
5801.63, matching the previously recorded best known value of 5801.27.

History: before the PR #130 feasibility fix the model was infeasible as
transformed (an eps1-perturbed log-Antoine equation conflicted with the
Antoine-derived vp bounds); the Antoine relations are now a pole-free
bilinear equation defining a bounded, shifted log vapor pressure plus a
bounded exponential, and all remaining eps guards are documented in the
module docstring. A GAMS/DICOPT solve of the original gamslib `hda` model
reports 4322.55, but that is a local (OA) claim on a nonconvex MINLP; the
corresponding configuration of this port sits at 4322.4.


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