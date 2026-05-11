# Methanol Production 

This is a GDP model to determine the optimal maximum profit for a methanol
production process.

Source paper:

> Turkay, M., Grossmann, I. E. (1996). Logic-based MINLP algorithms for the optimal synthesis of process networks. *Computers and Chemical Engineering*, 125, 959-978. https://doi.org/10.1016/0098-1354(95)00219-7

Related implementations used for unit checks:

- IDAES v1.3.0 workshop VLE parameter package for the same Turkay and
  Grossmann methanol synthesis problem:
  https://idaes-pse.readthedocs.io/en/1.3.0/_modules/idaes/examples/workshops/Module_3_Custom_Unit_Model/methanol_param_VLE.html
- IDAES examples-pse methanol synthesis flowsheet:
  https://idaes.github.io/examples-pse/latest/Examples/Flowsheets/methanol_synthesis_doc.html

## Problem Details

### Units and Constants

The source paper describes Example 3 with generic components `A`, `B`, `C`,
and `D`; this implementation uses the chemically suggestive names `H2`, `CO`,
`CH3OH`, and `CH4` for those same roles. Feed 1 is 0.60 `H2`, 0.25 `CO`, and
0.15 `CH4`. Feed 2 is 0.65 `H2`, 0.30 `CO`, and 0.05 `CH4`.

The implementation follows the scaled units used by the source model. Flow
variables are in [kg-mol/sec], temperature variables are in [100 K], pressure
variables are in [MPa], and the objective is scaled to [$1000/yr]. Temperature
values therefore use the source-model scale: for example, a value of 4
represents 400 K. The stream-cost and stream-revenue coefficients are the
source-paper prices annualized with 8500 h/yr, or 30.6e6 sec/yr, and divided by
1000 to match the objective scale.

The constants initialized by `MethanolModel` are implementation details used by
the model equations, not constructor arguments. Values marked as retained are
kept from the source model for traceability but are not currently referenced by
constraints.

| Name | Value | Unit or role | Source and scaling note |
|------|-------|--------------|-------------------------|
| `alpha` | 0.72 | Compressor power coefficient | Source compressor correlation |
| `eta` | 0.75 | Unitless compressor efficiency | Source compressor correlation |
| `gamma` | 0.23077 | Unitless exponent in compressor pressure and temperature equations | Isentropic pressure/temperature exponent; not the heat-capacity ratio `Cp/Cv` |
| `cp` | 35.0 | Heat capacity [kJ/kg-mol-K] | Equivalent to 0.035 MJ/(kg-mol-K), matching the IDAES workshop property package |
| `heat_of_reaction` | -15 | Scaled reactor energy-balance coefficient | Used as `0.01 * heat_of_reaction * consumption_rate` |
| `volume_conversion[9]` | 0.1 | Reactor 9 conversion coefficient | Multiplied by `reactor_volume` inside a dimensionless exponential term |
| `volume_conversion[10]` | 0.05 | Reactor 10 conversion coefficient | Multiplied by `reactor_volume` inside a dimensionless exponential term |
| `reactor_volume` | 100 | Reactor volume [m^3] | Source reactor volume used in both reactor alternatives |
| `electricity_cost` | 0.255 | Variable electricity coefficient in the scaled annual objective | Source $0.03/kWh annualized with 8500 h/yr and divided by 1000 |
| `cooling_cost` | 700 | Cooling-water coefficient in the scaled annual objective | Paired with `heat_unit_match` to represent source utility cost |
| `heating_cost` | 8000 | Steam-heating coefficient in the scaled annual objective | Paired with `heat_unit_match` to represent source utility cost |
| `purity_demand` | 0.9 | Unitless methanol product purity requirement | Source product specification of at least 90% C |
| `demand` | 1.0 | Retained source production-flow scaling value | Retained for traceability; product capacity is enforced through stream bounds instead |
| `flow_feed_lb` | 0.5 | Feed flow lower bound [kg-mol/sec] | Source-model feed lower bound |
| `flow_feed_ub` | 5 | Feed flow upper bound [kg-mol/sec] | Source-model feed upper bound |
| `flow_feed_temp` | 3 | Feed temperature [100 K] | 300 K in physical temperature units |
| `flow_feed_pressure` | 1 | Feed pressure [MPa] | Source feed pressure |
| `cost_flow_1` | 795.6 | Feed 1 cost coefficient in the scaled annual objective | Source $0.026/kg-mol annualized with 30.6e6 sec/yr and divided by 1000 |
| `cost_flow_2` | 1009.8 | Feed 2 cost coefficient in the scaled annual objective | Source $0.033/kg-mol annualized with 30.6e6 sec/yr and divided by 1000 |
| `price_of_product` | 7650 | Product revenue coefficient in the scaled annual objective | Source $0.25/kg-mol annualized with 30.6e6 sec/yr and divided by 1000 |
| `price_of_byproduct` | 642.6 | Byproduct revenue coefficient in the scaled annual objective | Source $0.021/kg-mol annualized with 30.6e6 sec/yr and divided by 1000 |
| `cheap_reactor_fixed_cost` | 100 | Cheap reactor fixed cost [$1000/yr] | Low-cost, low-conversion reactor alternative |
| `cheap_reactor_variable_cost` | 5 | Cheap reactor variable cost [$1000/m^3] | Low-cost, low-conversion reactor alternative |
| `expensive_reactor_fixed_cost` | 250 | Expensive reactor fixed cost [$1000/yr] | High-cost, high-conversion reactor alternative |
| `expensive_reactor_variable_cost` | 10 | Expensive reactor variable cost [$1000/m^3] | High-cost, high-conversion reactor alternative |
| `heat_unit_match` | 0.00306 | Annual heat-duty scaling coefficient | Converts the scaled flow-temperature heat balance into the utility-cost scale |
| `capacity_redundancy` | 1.2 | Retained source-model capacity redundancy value | Retained for traceability; not used directly in current constraints |
| `antoine_unit_trans` | 7500.6168 | Antoine pressure conversion [Torr/MPa] | Converts MPa vapor pressures for the Antoine equation |
| `K` | 0.415 | Unitless equilibrium-conversion coefficient | Source equilibrium conversion correlation |
| `delta_H` | 26.25 | Unitless equilibrium-correlation coefficient | Scaled so the equilibrium-conversion expression is dimensionless |
| `reactor_relation` | 0.9 | Unitless reactor outlet-to-inlet pressure ratio | Reactor outlet pressure equals 0.9 times reactor pressure |
| `fix_electricity_cost` | 175 | Fixed compressor electricity coefficient [$1000/yr] | Source fixed annual compressor charge in the objective scale |
| `two_stage_fix_cost` | 50 | Two-stage compressor fixed cost [$1000/yr] | Fixed annual charge for the two-stage compressor alternatives |

The flash block also defines Antoine vapor-pressure coefficients. These values
match the IDAES v1.3.0 workshop property package for the Turkay and Grossmann
problem.

| Component | `A` | `B` | `C` |
|-----------|----:|----:|----:|
| `H2` | 13.6333 | 164.9 | 3.19 |
| `CO` | 14.3686 | 530.22 | -13.15 |
| `CH3OH` | 18.5875 | 3626.55 | -34.29 |
| `CH4` | 15.2243 | 897.84 | -7.16 |

### Relationship to IDAES Methanol Examples

The GDPlib model and the IDAES methanol examples should not be compared as
drop-in equivalent formulations:

- This GDPlib model is the Turkay and Grossmann Example 3 GDP superstructure.
  It chooses between feedstocks, reactor alternatives, feed-compressor
  alternatives, and recycle-compressor alternatives through Pyomo.GDP
  disjunctions.
- The IDAES v1.3.0 workshop module is useful as a property-package cross-check
  for constants such as `cp`, Antoine coefficients, component ordering, and
  scaled state-variable units. It does not replace this GDP superstructure.
- The current IDAES examples-pse flowsheets are rigorous fixed-topology
  methanol synthesis flowsheets. They use IDAES unit models, SI-unit state
  variables, separate vapor and VLE property packages, initialization and
  scaling workflows, and process costing. They optimize operating specifications
  and revenue for a selected flowsheet rather than solving the 16-alternative
  GDP synthesis problem from the source paper.
- IDAES examples use an explicit maximization objective for revenue. This
  GDPlib benchmark exposes positive `m.profit`, but keeps the Pyomo objective
  as minimized negative profit because that is the verified GDPopt LOA solve
  path for this model.

### Solution

Best known objective value: -1793.4292381783 (optimal)

The Pyomo objective minimizes negative profit for GDPopt compatibility.
The corresponding maximum profit is 1793.4292381783 [$1000/yr].


### Size

| Component             |   Number |
|:----------------------|---------:|
| Variables             |      287 |
| Binary variables      |        8 |
| Integer variables     |        0 |
| Continuous variables  |      279 |
| Disjunctions          |        4 |
| Disjuncts             |        8 |
| Constraints           |      429 |
| Nonlinear constraints |       55 |
