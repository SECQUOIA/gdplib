# Methanol Production 

This is a GDP model to determine the optimal maximum profit for a methanol production process.

Source paper:

> Turkay, M., Grossmann, I. E. (1996). Logic-based MINLP algorithms for the optimal synthesis of process networks. *Computers and Chemical Engineering*, 125, 959-978. https://doi.org/10.1016/0098-1354(95)00219-7

## Problem Details

### Units and Constants

The implementation uses the scaled units from the source model. Flow variables
are in [kg-mol/sec], temperature variables are in [100 K], pressure variables
are in [MPa], and the objective is scaled to [$1000/yr]. Temperature values
therefore use the source-model scale: for example, a value of 4 represents
400 K.

The constants initialized by `MethanolModel` are implementation details used by
the model equations, not constructor arguments.

| Name | Value | Unit or role |
|------|-------|--------------|
| `alpha` | 0.72 | Compressor power coefficient |
| `eta` | 0.75 | Unitless compressor efficiency |
| `gamma` | 0.23077 | Unitless exponent in compressor pressure and temperature equations |
| `cp` | 35.0 | Heat capacity [kJ/kg-mol-K] |
| `heat_of_reaction` | -15 | Reactor energy-balance coefficient [kJ/kg-mol] |
| `volume_conversion[9]` | 0.1 | Reactor 9 conversion coefficient; multiplied by reactor volume in a dimensionless exponential term |
| `volume_conversion[10]` | 0.05 | Reactor 10 conversion coefficient; multiplied by reactor volume in a dimensionless exponential term |
| `reactor_volume` | 100 | Reactor volume [m^3] |
| `electricity_cost` | 0.255 | Electricity cost coefficient [$/10 kWh] |
| `cooling_cost` | 700 | Cooling-water cost coefficient [$/1e9 kJ] |
| `heating_cost` | 8000 | Steam-heating cost coefficient [$/1e9 kJ] |
| `purity_demand` | 0.9 | Unitless methanol product purity requirement |
| `demand` | 1.0 | Source-model production-flow scaling value; retained for reference and not used directly in the current constraints |
| `flow_feed_lb` | 0.5 | Feed flow lower bound [kg-mol/sec] |
| `flow_feed_ub` | 5 | Feed flow upper bound [kg-mol/sec] |
| `flow_feed_temp` | 3 | Feed temperature [100 K] |
| `flow_feed_pressure` | 1 | Feed pressure [MPa] |
| `cost_flow_1` | 795.6 | Feed 1 cost coefficient in the scaled annual objective |
| `cost_flow_2` | 1009.8 | Feed 2 cost coefficient in the scaled annual objective |
| `price_of_product` | 7650 | Product revenue coefficient in the scaled annual objective |
| `price_of_byproduct` | 642.6 | Byproduct revenue coefficient in the scaled annual objective |
| `cheap_reactor_fixed_cost` | 100 | Cheap reactor fixed cost [$1000/yr] |
| `cheap_reactor_variable_cost` | 5 | Cheap reactor variable cost [$1000/m^3] |
| `expensive_reactor_fixed_cost` | 250 | Expensive reactor fixed cost [$1000/yr] |
| `expensive_reactor_variable_cost` | 10 | Expensive reactor variable cost [$1000/m^3] |
| `heat_unit_match` | 0.00306 | Heat-duty scaling coefficient |
| `capacity_redundancy` | 1.2 | Source-model capacity redundancy value; retained for reference and not used directly in the current constraints |
| `antoine_unit_trans` | 7500.6168 | Antoine pressure conversion [Torr/MPa] |
| `K` | 0.415 | Unitless equilibrium-conversion coefficient |
| `delta_H` | 26.25 | Equilibrium-correlation coefficient scaled so the conversion expression is dimensionless |
| `reactor_relation` | 0.9 | Unitless reactor outlet-to-inlet pressure ratio |
| `fix_electricity_cost` | 175 | Fixed electricity cost coefficient [$/10 kWh] |
| `two_stage_fix_cost` | 50 | Two-stage compressor fixed cost [$1000/yr] |

Feed composition parameters are molar fractions. Feed 1 is 0.60 H2, 0.25 CO,
and 0.15 CH4. Feed 2 is 0.65 H2, 0.30 CO, and 0.05 CH4.

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
