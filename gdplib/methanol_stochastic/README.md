# Two-Stage Stochastic Methanol Production with Equipment Availability

This is a two-stage stochastic extension of the
[`methanol_availability`](../methanol_availability/) model, which in turn is
an availability-aware variant of the [methanol production](../methanol/) GDP
model. The process is designed before the product demand and the electricity
price are known; the expected operating profit over nine scenarios is weighted
by the system availability.

The process superstructure, feedstocks, unit models, and physical property
data are those of the [`methanol`](../methanol/) model, which implements
Example 3 of the source paper:

> Turkay, M., Grossmann, I. E. (1996). Logic-based MINLP algorithms for the optimal synthesis of process networks. *Computers and Chemical Engineering*, 125, 959-978. https://doi.org/10.1016/0098-1354(95)00219-7

The two model variants were written by Qi Chen in September 2020, alongside the
original `gdplib.methanol` implementation. They were ported to the current
Pyomo.GDP API, unit annotations, and formatting of `gdplib.methanol`; the port
builds the same variables, bounds, and constraints as the 2020 code.

Two model variants are available:

- ``build_model()`` or ``build_model(redundancy=False)`` - two-stage stochastic
  model with equipment availability.
- ``build_model(redundancy=True)`` - integrated reliability model, which also
  chooses how many redundant parallel units to install for each piece of
  equipment.

## Problem Details

### Stages and Scenarios

First-stage (design) decisions, shared by all scenarios:

- the four disjunctions of the `methanol` model: cheap or expensive feedstock,
  single- or two-stage feed compressor, cheap or expensive reactor, and single-
  or two-stage recycle compressor,
- equipment capacities `m.equipment_<u>.capacity` and availabilities,
- reactor volumes `reactor_<u>.reactor_volume` in [30, 150] [m^3],
- with `redundancy=True`, the parallel-unit selection binaries
  `m.equipment_<u>.ys`.

Second-stage (operating) decisions: all stream flows, temperatures, and
pressures, and every unit operating variable (compressor loads, heat duties,
conversions, flash conditions, split fractions), indexed by `m.scenarios`.
Capacities must cover the load or duty of every scenario:
$Q_u \ge 1.2\,W_{u,s}$.

The nine scenarios combine three product demand levels with three electricity
price levels. Each level has probability 0.3 (low), 0.4 (mid), or 0.3 (high),
and the scenario probabilities `m.probs` are the products of the two level
probabilities.

| Level | Demand `m.demands` (upper limit on `flows[23]`) [kg-mol/sec] | Electricity cost `m.electricity_cost` [$1000/yr per unit load] | Probability |
|-------|------:|------:|------:|
| low | 0.8 | 200 | 0.3 |
| mid | 1.0 | 255 | 0.4 |
| high | 1.2 | 300 | 0.3 |

Scenario names are `"<demand level> <electricity level>"`, for example
`"low high"`.

### Objective

The objective maximizes `m.profit` (the Pyomo objective minimizes `-m.profit`):

$$
\text{profit} = A \sum_s \pi_s P_{\text{op},s} - C_\text{reactor} - C_\text{equipment}
$$

where $A$ is the system availability (`m.system_availability`), $\pi_s$ is the
scenario probability, $P_{\text{op},s}$ is the operating profit of scenario
$s$ (product and byproduct revenue minus feed, electricity, cooling, and
heating costs), $C_\text{reactor}$ is the fixed and volume-proportional cost of
the selected reactor, and $C_\text{equipment}$ is the sum of fixed, repair, and
capacity costs of the installed units. The reliability data, the equipment
cost coefficients, and the differences from the deterministic `methanol` model
are the same as in [`methanol_availability`](../methanol_availability/README.md).

### Redundant Units

With `redundancy=True`, each unit $u$ has a set of candidate parallel units
`m.equipment_<u>.I` (three for compressors, four for heat exchangers and the
expansion valve) with binary selection variables $y_{u,i}$:

- An installed unit needs at least one selected copy, $\sum_i y_{u,i} \ge 1$.
  Units of the compressor alternative that is not selected have all
  $y_{u,i} = 0$ and availability 1.
- The availability of parallel copies is
  $a_u = 1 - \prod_i (1 - p_{u,i}\, y_{u,i})$.
- Each selected copy is sized for the full capacity and pays
  $\beta_u + r_{u,i} + \alpha_u Q_u$. The products
  $Q_u y_{u,i}$ are represented by `capacity_ys` with exact McCormick
  constraints (the capacity bound is 60).

| Units | Type | Availability $p_{u,i}$ of copy 1, 2, 3, 4 | Repair cost $r_{u,i}$ of copy 1, 2, 3, 4 |
|-------|------|------|------|
| 3, 4, 6, 16, 17, 19 | Compressors | 0.977, 0.975, 0.973 | 2.0, 1.9, 1.8 |
| 5, 7, 12, 14, 15, 18 | Coolers and heaters | 0.998, 0.996, 0.994, 0.992 | 1.4, 1.35, 1.3, 1.25 |
| 8 | Heater | 0.998, 0.996, 0.994, 0.992 | 0, 0, 0, 0 |
| 11 | Expansion valve | 0.986, 0.986, 0.986, 0.986 | 0.2, 0.2, 0.2, 0.2 |

Without redundancy, only the first copy's availability and repair cost are
used, and the model is the scenario-based counterpart of
`methanol_availability`: one unit per installed piece of equipment, with
capacities that cover the loads and duties of every scenario.

### Solution

Best known objective values:
- ``no_redundancy``: -2156.0423 (local NLP solves; not proven globally optimal)
- ``redundancy``: -2187.3902 (feasible point; not proven optimal)

The Pyomo objective minimizes negative expected profit. Both values use the
expensive feed, single-stage feed compressor, cheap reactor, and single-stage
recycle compressor.

- ``no_redundancy``: best of the 16 configurations, each fixed with
  `gdp.fix_disjuncts` and solved as an NLP with IPOPT 3.14.20 from the midpoint
  of the variable bounds; all 16 NLPs terminated locally optimal. GDPopt LOA
  (CBC for the MIP, IPOPT for the NLP subproblems) reaches the same design and
  objective with Pyomo 6.10.1 and with the Pyomo `main` branch of November
  2026. The system availability is 0.9318.
- ``redundancy``: MindtPy OA (CBC 2.10.3 for the MIP, IPOPT 3.14.20 for the
  NLP) on the fixed configuration above, termination condition `feasible`;
  the OA bounds are not valid for this nonconvex model, and other
  configurations were not searched. The design adds three parallel copies of
  heater 8, a second expansion valve 11, and a second recycle compressor 16,
  for a system availability of 0.9685.

### Size

| Component             |   no_redundancy |   redundancy |
|:----------------------|----------------:|-------------:|
| Variables             |            2518 |         2618 |
| Binary variables      |               8 |           58 |
| Integer variables     |               0 |            0 |
| Continuous variables  |            2510 |         2560 |
| Disjunctions          |               4 |            4 |
| Disjuncts             |               8 |            8 |
| Constraints           |            3986 |         4306 |
| Nonlinear constraints |             496 |          510 |

## Usage

```python
from gdplib.methanol_stochastic import build_model

m = build_model()  # two-stage stochastic model
m_redundant = build_model(redundancy=True)  # integrated reliability model
```

`build_model()` populates Big-M suffixes on the disjunct constraints, as in the
`methanol` model. `gdplib.methanol_stochastic.methanol_stochastic.solve_with_gdp_opt()`
solves the model with GDPopt (LOA by default). With `redundancy=True`, the
GDPopt subproblems keep the redundancy binaries, so also pass an MINLP
subsolver, for example `minlp_solver="gams"` and
`minlp_solver_args={"solver": "dicopt"}`.

As in `methanol`, the Pyomo objective minimizes `-m.profit`. Keep this form
rather than `maximize(m.profit)`: GDPopt LOA ignores a maximization sense in
Pyomo releases up to 6.10.1
([gdplib#114](https://github.com/SECQUOIA/gdplib/issues/114), fixed on the
Pyomo `main` branch by [Pyomo#3986](https://github.com/Pyomo/pyomo/pull/3986)),
and when MC++ is available GDPopt fails while bounding a nonlinear objective
whose expression is the named `m.profit` Expression (`AssertionError` in
`pyomo.contrib.mcpp`, also on Pyomo `main` as of November 2026).
