# Methanol Production with Equipment Availability

This is an availability-aware variant of the [methanol production](../methanol/)
GDP model. It determines the process configuration, equipment capacities, and
reactor volume that maximize the profit of a methanol synthesis process when
the operating profit is only earned while all installed equipment is
operational.

The process superstructure, feedstocks, unit models, and physical property
data are those of the [`methanol`](../methanol/) model, which implements
Example 3 of the source paper:

> Turkay, M., Grossmann, I. E. (1996). Logic-based MINLP algorithms for the optimal synthesis of process networks. *Computers and Chemical Engineering*, 125, 959-978. https://doi.org/10.1016/0098-1354(95)00219-7

The model was written by Qi Chen in September 2020, alongside the original
`gdplib.methanol` implementation. It was ported to the current Pyomo.GDP API,
unit annotations, and formatting of `gdplib.methanol`; the port builds the same
variables, bounds, and constraints as the 2020 code. A two-stage
stochastic extension of this model is available as
[`methanol_stochastic`](../methanol_stochastic/).

## Problem Details

The model keeps the four disjunctions of the `methanol` model (16
configurations):

- cheap or expensive feedstock,
- single- or two-stage compression of the reactor feed,
- cheap (low conversion) or expensive (high conversion) reactor,
- single- or two-stage compression of the recycle stream.

### Reliability Model

The equipment considered for reliability are the compressors (units 3, 4, 6,
16, 17, 19), coolers (units 5, 7, 12, 18), heaters (units 8, 14, 15), and the
expansion valve (unit 11). Units 3 and 16 belong to the single-stage compressor
disjuncts; units 4, 5, 6 and 17, 18, 19 belong to the two-stage compressor
disjuncts. The other units are always installed.

For each unit $u$, the model block `m.equipment_<u>` holds a capacity $Q_u$
and an availability $a_u$:

- If unit $u$ is installed, $a_u = p_u$, its availability. Units of the
  compressor alternative that is not selected have $a_u = 1$ and $Q_u = 0$.
- The capacity of a compressor or heat exchanger is $Q_u = 1.2\,W_u$, where
  $W_u$ is the compressor load or heat duty (20 [%] design margin).
- The system availability is $A = \prod_u a_u$ (`m.system_availability`).

The objective maximizes `m.profit` (the Pyomo objective minimizes `-m.profit`):

$$
\text{profit} = A \cdot P_\text{op} - C_\text{reactor} - \sum_{u \text{ installed}} \left( \beta_u + r_u + \alpha_u Q_u \right)
$$

where $P_\text{op}$ is the operating profit (product and byproduct revenue
minus feed, electricity, cooling, and heating costs), $C_\text{reactor}$ is
the fixed and volume-proportional cost of the selected reactor, $\beta_u$ is a
fixed cost per installed unit (`cost_b`), $r_u$ is a repair cost
(`repair_cost`), and $\alpha_u$ is a capacity cost coefficient (`cost_a`).
The fixed and repair costs of disjunct units are multiplied by the binary
indicator of their disjunct. The valve has no capacity cost.

| Units | Type | $p_u$ | $r_u$ | $\alpha_u$ | $\beta_u$ |
|-------|------|------:|------:|-----------:|----------:|
| 3, 4, 6, 16, 17, 19 | Compressors | 0.977 | 2.0 | 132 | 13.5 |
| 5, 7, 12, 14, 15, 18 | Coolers and heaters | 0.998 | 1.4 | 0.89098 | 39.216 |
| 8 | Heater | 0.998 | 0.0 | 0.89098 | 0 |
| 11 | Expansion valve | 0.986 | 0.2 | 0 | 1 |

The heat-exchanger coefficients are computed in the code as
`10**9 / 3600 / 8500 * 2.84 * 4.8 / 500` ($\alpha_u$) and
`10**9 / 3600 / 8500 * 1.2` ($\beta_u$). All cost coefficients are in the
objective scale of the `methanol` model, [$1000/yr]; the capacity and load
variables are in the source-model scale. The 2020 code comments described
`cost_a` as the fixed cost and `cost_b` as the capacity cost; the table follows
how the objective uses them.

### Differences from the `methanol` Model

| Item | `methanol` | `methanol_availability` |
|------|-----------|-------------------------|
| Objective | Profit | Availability-weighted operating profit minus reactor and equipment costs |
| Product price `price_of_product` | 7650 | 8650 |
| Electricity cost per unit compressor load | `fix_electricity_cost + electricity_cost` = 175 + 0.255 | `electricity_cost` = 255 |
| Two-stage compressor fixed cost `two_stage_fix_cost` | 50 | Not used (per-unit equipment costs instead) |
| Reactor volume | Fixed, 100 [m^3] | Variable `reactor_volume` in [30, 150] [m^3] for each reactor |
| Reactor feed temperature `temps[13]` | [4.23, 8.73] [100 K] | [3, 6] [100 K] |
| Reactor product temperature `temps[18]` | [5.23, 8.73] [100 K] | [3, 6] [100 K] |
| Reactor temperature `reactor_<u>.temp` | [3, 8.73] [100 K] | [3, 6] [100 K] |
| Product and byproduct temperatures `temps[23]`, `temps[25]` | Fixed at 4 [100 K] | Fixed at 4.2 [100 K] |
| Production limit | `flows[22] <= 1.0` [kg-mol/sec] | `demand_con`: `flows[23] <= 1.0` [kg-mol/sec] |

All other parameters, bounds, and unit models are the same as in the
[`methanol`](../methanol/README.md) model, whose README documents their units
and scaling.

### Solution

Best known objective value: -2174.8024 (local NLP solves; not proven globally optimal)

The Pyomo objective minimizes negative profit. The corresponding profit is
2174.8024 [$1000/yr], with expensive feed, single-stage feed compressor, cheap
reactor, single-stage recycle compressor, and a system availability of 0.9318.
This value was found by fixing each of the 16 configurations with
`gdp.fix_disjuncts` and solving the resulting NLP with IPOPT 3.14.20 from the
midpoint of the variable bounds; all 16 NLPs terminated locally optimal.
GDPopt LOA (CBC for the MIP, IPOPT for the NLP subproblems) reaches the same
design and objective with Pyomo 6.10.1 and with the Pyomo `main` branch of
November 2026. The model is nonconvex, so this is not a global optimality
certificate.

### Size

| Component             |   Number |
|:----------------------|---------:|
| Variables             |      318 |
| Binary variables      |        8 |
| Integer variables     |        0 |
| Continuous variables  |      310 |
| Disjunctions          |        4 |
| Disjuncts             |        8 |
| Constraints           |      474 |
| Nonlinear constraints |       56 |

## Usage

```python
from gdplib.methanol_availability import build_model

m = build_model()
```

`build_model()` populates Big-M suffixes on the disjunct constraints, as in the
`methanol` model. `gdplib.methanol_availability.methanol_availability.solve_with_gdp_opt()`
solves the model with GDPopt (LOA by default); it requires the subsolvers
passed to it, GAMS by default.

As in `methanol`, the Pyomo objective minimizes `-m.profit`. Keep this form
rather than `maximize(m.profit)`: GDPopt LOA ignores a maximization sense in
Pyomo releases up to 6.10.1
([gdplib#114](https://github.com/SECQUOIA/gdplib/issues/114), fixed on the
Pyomo `main` branch by [Pyomo#3986](https://github.com/Pyomo/pyomo/pull/3986)),
and when MC++ is available GDPopt fails while bounding a nonlinear objective
whose expression is the named `m.profit` Expression (`AssertionError` in
`pyomo.contrib.mcpp`, also on Pyomo `main` as of November 2026).
