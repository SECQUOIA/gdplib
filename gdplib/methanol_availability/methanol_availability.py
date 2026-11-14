"""
methanol_availability.py
Availability-aware variant of the methanol synthesis GDP in ``gdplib.methanol``
(Example 3 from Reference [1]).

The process superstructure is the same as in ``gdplib.methanol``: two
alternative feedstocks, single- or two-stage compression of the reactor feed
and of the recycle stream, and a cheap (low conversion) or expensive (high
conversion) reactor, followed by coolers, heaters, an expansion valve and a
flash separator.

This variant adds equipment reliability to the design problem:

- Each compressor, heat exchanger, and the expansion valve has an availability
  (probability of being operational). The system availability is the product
  of the availabilities of the installed units. Units in a disjunct that is not
  selected have availability 1 and zero capacity.
- The operating profit (product and byproduct revenue minus feed, electricity,
  cooling, and heating costs) is weighted by the system availability.
- Equipment capacities are decision variables, linked to the compressor load or
  heat duty with a 20 [%] design margin, and charged a capacity cost plus a
  fixed cost and a repair cost per installed unit.
- Reactor volumes are decision variables instead of the fixed 100 [m^3] of the
  base model.
- Product flow is limited by a demand constraint instead of an upper bound on
  the liquid product stream.

The model was written by Qi Chen in September 2020, alongside the original
``gdplib.methanol`` implementation. It was ported to the current Pyomo.GDP API,
unit annotations, and formatting of ``gdplib.methanol``.

References:
    [1] Turkay, M., Grossmann, I. E. (1996). Logic-based MINLP algorithms for the optimal synthesis of process networks. Computers and Chemical Engineering, 125, 959-978. https://doi.org/10.1016/0098-1354(95)00219-7
"""

import pyomo.environ as pe
from pyomo.contrib.fbbt.fbbt import fbbt, compute_bounds_on_expr
import logging
import pyomo.gdp as gdp


class MethanolModelAvailability(object):
    def __init__(self):
        """Initialize the Pyomo model and fixed parameter values.

        The process description and references are documented in the module
        docstring and the package README. Constructor-created attributes are
        implementation details used by the model builder methods.
        """

        self.model = m = pe.ConcreteModel("methanol availability model")

        self.alpha = 0.72  # compressor coefficient [dimensionless]
        self.eta = 0.75  # compressor efficiency [dimensionless]
        self.gamma = 0.23077  # isentropic pressure/temperature exponent [dimensionless]
        self.cp = 35.0  # heat capacity [kJ/kg-mol-K]
        self.heat_of_reaction = -15  # reactor heat-balance coeff. [source scale]
        self.volume_conversion = dict()
        self.volume_conversion[9] = 0.1  # reactor 9 conversion coefficient [1/m^3]
        self.volume_conversion[10] = 0.05  # reactor 10 conversion coefficient [1/m^3]
        self.electricity_cost = 255  # electricity cost [$1000/yr/load, source scale]
        self.cooling_cost = 700  # cooling-water cost [$1000/source heat-duty]
        self.heating_cost = 8000  # steam-heating cost [$1000/source heat-duty]
        self.purity_demand = 0.9  # purity demand in product stream [mole fraction]
        self.demand = 1.0  # product flow upper limit [kg-mol/sec]
        self.flow_feed_lb = 0.5  # feed flow lower bound [kg-mol/sec]
        self.flow_feed_ub = 5  # feed flow upper bound [kg-mol/sec]
        self.flow_feed_temp = 3  # feed temperature [100 K]
        self.flow_feed_pressure = 1  # feed pressure [MPa]
        self.cost_flow_1 = 795.6  # feed 1 cost coefficient [$1000/yr/(kg-mol/sec)]
        self.cost_flow_2 = 1009.8  # feed 2 cost coefficient [$1000/yr/(kg-mol/sec)]
        self.price_of_product = 8650  # product revenue [$1000/yr/(kg-mol/sec)]
        self.price_of_byproduct = 642.6  # byproduct revenue [$1000/yr/(kg-mol/sec)]
        self.cheap_reactor_fixed_cost = 100  # cheap reactor fixed cost [$1000/yr]
        self.cheap_reactor_variable_cost = 5  # cheap reactor variable cost [$1000/m^3]
        self.expensive_reactor_fixed_cost = 250  # expensive fixed cost [$1000/yr]
        self.expensive_reactor_variable_cost = 10  # expensive variable cost [$1000/m^3]
        self.heat_unit_match = 0.00306  # annual heat-duty scale [source scale]
        self.capacity_redundancy = 1.2  # capacity margin over load/duty [dimensionless]
        self.antoine_unit_trans = 7500.6168  # Antoine pressure conversion [Torr/MPa]
        self.K = 0.415  # equilibrium-conversion coefficient [dimensionless]
        self.delta_H = 26.25  # equilibrium-correlation coefficient [dimensionless]
        self.reactor_relation = 0.9  # reactor outlet/inlet pressure [dimensionless]

        m.streams = pe.Set(
            initialize=list(range(1, 34)), ordered=True, doc="Set of streams"
        )
        m.components = pe.Set(
            initialize=["H2", "CO", "CH3OH", "CH4"],
            ordered=True,
            doc="Set of components",
        )
        m.flows = pe.Var(m.streams, bounds=(0, 20), doc="Flowrates [kg-mol/sec]")
        m.temps = pe.Var(m.streams, bounds=(3, 9), doc="Temperature [100 K]")
        m.pressures = pe.Var(m.streams, bounds=(0.1, 15), doc="Pressures [MPa]")
        m.component_flows = pe.Var(
            m.streams,
            m.components,
            bounds=(0, 20),
            doc="Component flowrates [kg-mol/sec]",
        )

        flow_1 = dict()
        # Feed 1 (cheap feedstock; component molar fractions)
        flow_1["H2"] = 0.6  # Feed 1 H2 composition [mole fraction]
        flow_1["CO"] = 0.25  # Feed 1 CO composition [mole fraction]
        flow_1["CH4"] = 0.15  # Feed 1 CH4 composition [mole fraction]
        m.flow_1_composition = pe.Param(
            m.components,
            initialize=flow_1,
            default=0,
            doc="Molar composition of feed 1",
        )
        flow_2 = dict()
        # Feed 2 (expensive feedstock; component molar fractions)
        flow_2["H2"] = 0.65  # Feed 2 H2 composition [mole fraction]
        flow_2["CO"] = 0.30  # Feed 2 CO composition [mole fraction]
        flow_2["CH4"] = 0.05  # Feed 2 CH4 composition [mole fraction]
        m.flow_2_composition = pe.Param(
            m.components,
            initialize=flow_2,
            default=0,
            doc="Molar composition of feed 2",
        )

        # Setting operating conditions (bounds) for specific streams in the process
        m.pressures[13].setlb(2.5)  # Reactor feed pressure lower bound [MPa]
        m.temps[13].setlb(3.0)  # Reactor feed temperature lower bound [100 K]
        m.temps[13].setub(6.0)  # Reactor feed temperature upper bound [100 K]
        m.temps[18].setlb(3.0)  # Reactor product temperature lower bound [100 K]
        m.temps[18].setub(6.0)  # Reactor product temperature upper bound [100 K]
        m.flows[22].setlb(0.1)  # Liquid product flow lower bound [kg-mol/sec]
        m.temps[23].fix(4.2)  # Product stream temperature [100 K]
        m.temps[25].fix(4.2)  # Byproduct stream temperature [100 K]

        self.inlet_streams = dict()
        self.outlet_streams = dict()
        self.vapor_outlets = dict()
        self.liquid_outlets = dict()

        self.inlet_streams[3] = 4
        self.inlet_streams[4] = 5
        self.inlet_streams[5] = 7
        self.inlet_streams[6] = 8
        self.inlet_streams[7] = 11
        self.inlet_streams[8] = 12
        self.inlet_streams[9] = 15
        self.inlet_streams[10] = 14
        self.inlet_streams[11] = 18
        self.inlet_streams[12] = 19
        self.inlet_streams[13] = 20
        self.inlet_streams[14] = 22
        self.inlet_streams[15] = 24
        self.inlet_streams[16] = 27
        self.inlet_streams[17] = 28
        self.inlet_streams[18] = 30
        self.inlet_streams[19] = 31
        self.inlet_streams["feed_mixer"] = (1, 2)
        self.inlet_streams["feed_splitter"] = 3
        self.inlet_streams["compressed_feed_mixer"] = (6, 9)
        self.inlet_streams["recycle_feed_mixer"] = (10, 33)
        self.inlet_streams["reactor_feed_splitter"] = 13
        self.inlet_streams["reactor_product_mixer"] = (16, 17)
        self.inlet_streams["purge_splitter"] = 21
        self.inlet_streams["recycle_compressor_splitter"] = 26
        self.inlet_streams["recycle_compressor_mixer"] = (29, 32)

        self.outlet_streams[3] = 6
        self.outlet_streams[4] = 7
        self.outlet_streams[5] = 8
        self.outlet_streams[6] = 9
        self.outlet_streams[7] = 12
        self.outlet_streams[8] = 13
        self.outlet_streams[9] = 17
        self.outlet_streams[10] = 16
        self.outlet_streams[11] = 19
        self.outlet_streams[12] = 20
        self.outlet_streams[14] = 23
        self.outlet_streams[15] = 25
        self.outlet_streams[16] = 29
        self.outlet_streams[17] = 30
        self.outlet_streams[18] = 31
        self.outlet_streams[19] = 32
        self.outlet_streams["feed_mixer"] = 3
        self.outlet_streams["feed_splitter"] = (4, 5)
        self.outlet_streams["compressed_feed_mixer"] = 10
        self.outlet_streams["recycle_feed_mixer"] = 11
        self.outlet_streams["reactor_feed_splitter"] = (14, 15)
        self.outlet_streams["reactor_product_mixer"] = 18
        self.outlet_streams["purge_splitter"] = (26, 24)
        self.outlet_streams["recycle_compressor_splitter"] = (27, 28)
        self.outlet_streams["recycle_compressor_mixer"] = 33

        self.vapor_outlets[13] = 21
        self.liquid_outlets[13] = 22

        # ************************************
        # Reliability and equipment cost data
        # ************************************
        # Unit numbers of the compressors (3, 4, 6, 16, 17, 19), heat exchangers
        # (5, 7, 8, 12, 14, 15, 18), and expansion valve (11).
        self.equipment_list = (3, 4, 6, 16, 17, 19, 5, 12, 18, 7, 8, 14, 15, 11)
        # Units that only exist when their compressor disjunct is selected
        self.equipment_in_disjunct = (3, 4, 5, 6, 16, 17, 18, 19)
        # Units that are always part of the flowsheet
        self.equipment_not_in_disjunct = (7, 8, 11, 12, 14, 15)

        # Availability (probability that the unit is operational) [dimensionless]
        self.probability = {
            3: 0.977,
            4: 0.977,
            6: 0.977,
            16: 0.977,
            17: 0.977,
            19: 0.977,
            5: 0.998,
            12: 0.998,
            18: 0.998,
            7: 0.998,
            8: 0.998,
            14: 0.998,
            15: 0.998,
            11: 0.986,
        }
        # Repair cost per installed unit [$1000/yr, source scale]
        self.repair_cost = {
            3: 2.0,
            4: 2.0,
            6: 2.0,
            16: 2.0,
            17: 2.0,
            19: 2.0,
            5: 1.4,
            12: 1.4,
            18: 1.4,
            7: 1.4,
            8: 0.0,
            14: 1.4,
            15: 1.4,
            11: 0.2,
        }
        # cost_a multiplies the unit capacity [$1000/yr per capacity unit, source
        # scale]; cost_b is a fixed cost per installed unit [$1000/yr, source
        # scale]. The 2020 source comments described these two roles the other
        # way round; the roles documented here follow the objective function.
        self.cost_a = {}
        self.cost_b = {}
        for u in (3, 4, 6, 16, 17, 19):  # compressors
            self.cost_a[u] = 132
            self.cost_b[u] = 13.5
        for u in (5, 12, 18, 7, 8, 14, 15):  # heat exchangers
            self.cost_a[u] = 10**9 / 3600 / 8500 * 2.84 * 4.8 / 500
            self.cost_b[u] = 10**9 / 3600 / 8500 * 1.2
        self.cost_b[8] = 0
        # expansion valve 11 (no capacity cost)
        self.cost_a[11] = 0
        self.cost_b[11] = 1

        def _total_flow(_m, _s):
            """
            This function calculates the total flowrate of a stream.

            Parameters
            ----------
            _m : Pyomo.ConcreteModel
                model
            _s : int
                stream index

            Returns
            -------
            Pyomo.Constraint.Expression
                The total flowrate of the stream.
            """
            return _m.flows[_s] == sum(
                _m.component_flows[_s, _c] for _c in _m.components
            )

        m.total_flow_con = pe.Constraint(
            m.streams, rule=_total_flow, doc="Total flowrate of a stream"
        )

        m.purity_con = pe.Constraint(
            expr=m.component_flows[23, "CH3OH"] >= self.purity_demand * m.flows[23],
            doc="Purity constraint",
        )

        m.demand_con = pe.Constraint(
            expr=m.flows[23] <= self.demand, doc="Product demand constraint"
        )

        for eq in self.equipment_not_in_disjunct:
            self.add_reliability_capacity(None, eq, True)

        # ************************************
        # Feed
        # ************************************
        m.cheap_feed_disjunct = gdp.Disjunct()
        # cheap feed constraints
        self.build_equal_streams(m.cheap_feed_disjunct, 1, 3)
        self.build_stream_doesnt_exist_con(m.cheap_feed_disjunct, 2)
        m.cheap_feed_disjunct.feed_cons = c = pe.ConstraintList()
        c.add(m.component_flows[1, "H2"] == m.flow_1_composition["H2"] * m.flows[1])
        c.add(m.component_flows[1, "CO"] == m.flow_1_composition["CO"] * m.flows[1])
        c.add(m.component_flows[1, "CH4"] == m.flow_1_composition["CH4"] * m.flows[1])
        c.add(m.flows[1] >= self.flow_feed_lb)
        c.add(m.flows[1] <= self.flow_feed_ub)
        c.add(m.temps[1] == self.flow_feed_temp)
        c.add(m.pressures[1] == self.flow_feed_pressure)

        m.expensive_feed_disjunct = gdp.Disjunct()
        # expensive feed constraints
        self.build_equal_streams(m.expensive_feed_disjunct, 2, 3)
        self.build_stream_doesnt_exist_con(m.expensive_feed_disjunct, 1)
        m.expensive_feed_disjunct.feed_cons = c = pe.ConstraintList()
        c.add(m.component_flows[2, "H2"] == m.flow_2_composition["H2"] * m.flows[2])
        c.add(m.component_flows[2, "CO"] == m.flow_2_composition["CO"] * m.flows[2])
        c.add(m.component_flows[2, "CH4"] == m.flow_2_composition["CH4"] * m.flows[2])
        c.add(m.flows[2] >= self.flow_feed_lb)
        c.add(m.flows[2] <= self.flow_feed_ub)
        c.add(m.temps[2] == self.flow_feed_temp)
        c.add(m.pressures[2] == self.flow_feed_pressure)

        m.feed_disjunctions = gdp.Disjunction(
            expr=[m.cheap_feed_disjunct, m.expensive_feed_disjunct],
            doc="Feed disjunctions",
        )

        # ************************************
        # Feed compressors
        # ************************************
        m.single_stage_feed_compressor_disjunct = gdp.Disjunct()
        # single-stage compressor constraints
        self.add_reliability_capacity_for_disjunct(
            m.single_stage_feed_compressor_disjunct, [3], [4, 5, 6]
        )
        self.build_equal_streams(m.single_stage_feed_compressor_disjunct, 3, 4)
        self.build_stream_doesnt_exist_con(m.single_stage_feed_compressor_disjunct, 5)
        self.build_stream_doesnt_exist_con(m.single_stage_feed_compressor_disjunct, 7)
        self.build_stream_doesnt_exist_con(m.single_stage_feed_compressor_disjunct, 8)
        self.build_stream_doesnt_exist_con(m.single_stage_feed_compressor_disjunct, 9)
        self.build_equal_streams(m.single_stage_feed_compressor_disjunct, 6, 10)
        self.build_compressor(m.single_stage_feed_compressor_disjunct, 3)

        m.two_stage_feed_compressor_disjunct = gdp.Disjunct()
        # two-stage compressor constraints
        self.add_reliability_capacity_for_disjunct(
            m.two_stage_feed_compressor_disjunct, [4, 5, 6], [3]
        )
        self.build_equal_streams(m.two_stage_feed_compressor_disjunct, 3, 5)
        self.build_equal_streams(m.two_stage_feed_compressor_disjunct, 9, 10)
        self.build_stream_doesnt_exist_con(m.two_stage_feed_compressor_disjunct, 4)
        self.build_stream_doesnt_exist_con(m.two_stage_feed_compressor_disjunct, 6)
        self.build_compressor(m.two_stage_feed_compressor_disjunct, 4)
        self.build_cooler(m.two_stage_feed_compressor_disjunct, 5)
        self.build_compressor(m.two_stage_feed_compressor_disjunct, 6)
        m.two_stage_feed_compressor_disjunct.equal_electric_requirements = pe.Constraint(
            expr=m.two_stage_feed_compressor_disjunct.compressor_4.electricity_requirement
            == m.two_stage_feed_compressor_disjunct.compressor_6.electricity_requirement,
            doc="Electricity requirement constraint",
        )
        m.two_stage_feed_compressor_disjunct.exists = pe.Var(
            bounds=(0, 1), doc="Feed compressor choice variable"
        )
        m.two_stage_feed_compressor_disjunct.exists_con = pe.Constraint(
            expr=m.two_stage_feed_compressor_disjunct.exists == 1,
            doc="Feed compressor choice constraint",
        )

        m.feed_compressor_disjunction = gdp.Disjunction(
            expr=[
                m.single_stage_feed_compressor_disjunct,
                m.two_stage_feed_compressor_disjunct,
            ],
            doc="Feed compressor disjunction",
        )

        self.build_mixer(m, "recycle_feed_mixer")
        self.build_cooler(m, 7)
        self.build_heater(m, 8)

        # ************************************
        # Reactors
        # ************************************
        m.expensive_reactor = gdp.Disjunct()
        # expensive (high cost, high conversion) reactor constraints
        self.build_equal_streams(m.expensive_reactor, 13, 15)
        self.build_equal_streams(m.expensive_reactor, 17, 18)
        self.build_stream_doesnt_exist_con(m.expensive_reactor, 14)
        self.build_stream_doesnt_exist_con(m.expensive_reactor, 16)
        self.build_reactor(m.expensive_reactor, 9)
        m.expensive_reactor.exists = pe.Var(
            bounds=(0, 1), doc="Reactor choice variable"
        )
        m.expensive_reactor.exists_con = pe.Constraint(
            expr=m.expensive_reactor.exists == 1, doc="Reactor choice constraint"
        )
        m.expensive_reactor.composition_cons = c = pe.ConstraintList()
        for _comp in m.components:
            c.add(m.component_flows[17, _comp] >= 0.01)

        m.cheap_reactor = gdp.Disjunct()
        # cheap (low cost, low conversion) reactor constraints
        self.build_equal_streams(m.cheap_reactor, 13, 14)
        self.build_equal_streams(m.cheap_reactor, 16, 18)
        self.build_stream_doesnt_exist_con(m.cheap_reactor, 15)
        self.build_stream_doesnt_exist_con(m.cheap_reactor, 17)
        self.build_reactor(m.cheap_reactor, 10)
        m.cheap_reactor.exists = pe.Var(bounds=(0, 1), doc="Reactor choice variable")
        m.cheap_reactor.exists_con = pe.Constraint(
            expr=m.cheap_reactor.exists == 1, doc="Reactor choice constraint"
        )
        m.cheap_reactor.composition_cons = c = pe.ConstraintList()
        for _comp in m.components:
            c.add(m.component_flows[16, _comp] >= 0.01)

        m.reactor_disjunction = gdp.Disjunction(
            expr=[m.expensive_reactor, m.cheap_reactor], doc="Reactor disjunction"
        )

        self.build_expansion_valve(m, 11)
        self.build_cooler(m, 12)
        self.build_flash(m, 13)
        self.build_heater(m, 14)
        self.build_splitter(m, "purge_splitter")
        self.build_heater(m, 15)

        # ************************************
        # Recycle compressors
        # ************************************
        m.single_stage_recycle_compressor_disjunct = gdp.Disjunct()
        # single-stage compressor constraints
        self.add_reliability_capacity_for_disjunct(
            m.single_stage_recycle_compressor_disjunct, [16], [17, 18, 19]
        )
        self.build_equal_streams(m.single_stage_recycle_compressor_disjunct, 26, 27)
        self.build_equal_streams(m.single_stage_recycle_compressor_disjunct, 29, 33)
        self.build_stream_doesnt_exist_con(
            m.single_stage_recycle_compressor_disjunct, 28
        )
        self.build_stream_doesnt_exist_con(
            m.single_stage_recycle_compressor_disjunct, 30
        )
        self.build_stream_doesnt_exist_con(
            m.single_stage_recycle_compressor_disjunct, 31
        )
        self.build_stream_doesnt_exist_con(
            m.single_stage_recycle_compressor_disjunct, 32
        )
        self.build_compressor(m.single_stage_recycle_compressor_disjunct, 16)

        m.two_stage_recycle_compressor_disjunct = gdp.Disjunct()
        # two-stage compressor constraints
        self.add_reliability_capacity_for_disjunct(
            m.two_stage_recycle_compressor_disjunct, [17, 18, 19], [16]
        )
        self.build_equal_streams(m.two_stage_recycle_compressor_disjunct, 26, 28)
        self.build_equal_streams(m.two_stage_recycle_compressor_disjunct, 32, 33)
        self.build_stream_doesnt_exist_con(m.two_stage_recycle_compressor_disjunct, 27)
        self.build_stream_doesnt_exist_con(m.two_stage_recycle_compressor_disjunct, 29)
        self.build_compressor(m.two_stage_recycle_compressor_disjunct, 17)
        self.build_cooler(m.two_stage_recycle_compressor_disjunct, 18)
        self.build_compressor(m.two_stage_recycle_compressor_disjunct, 19)
        m.two_stage_recycle_compressor_disjunct.equal_electric_requirements = pe.Constraint(
            expr=m.two_stage_recycle_compressor_disjunct.compressor_17.electricity_requirement
            == m.two_stage_recycle_compressor_disjunct.compressor_19.electricity_requirement,
            doc="Electricity requirement constraint",
        )
        m.two_stage_recycle_compressor_disjunct.exists = pe.Var(
            bounds=(0, 1), doc="Recycle compressor choice variable"
        )
        m.two_stage_recycle_compressor_disjunct.exists_con = pe.Constraint(
            expr=m.two_stage_recycle_compressor_disjunct.exists == 1,
            doc="Recycle compressor choice constraint",
        )

        m.recycle_compressor_disjunction = gdp.Disjunction(
            expr=[
                m.single_stage_recycle_compressor_disjunct,
                m.two_stage_recycle_compressor_disjunct,
            ],
            doc="Recycle compressor disjunction",
        )

        # ************************************
        # System availability
        # ************************************
        m.system_availability = pe.Var(
            bounds=(0, 1), doc="System availability [dimensionless]"
        )
        m.system_available_con = pe.Constraint(
            expr=m.system_availability
            == pe.prod(
                getattr(m, "equipment_" + str(eq)).availability
                for eq in self.equipment_list
            ),
            doc="System availability as the product of unit availabilities",
        )

        # ************************************
        # Objective
        # ************************************
        """
        Objective function to maximize the availability-weighted operating
        profit minus reactor and equipment costs [$1000/yr].
        """
        e = 0
        e -= self.cost_flow_1 * m.flows[1]
        e -= self.cost_flow_2 * m.flows[2]
        e += self.price_of_product * m.flows[23]
        e += self.price_of_byproduct * m.flows[25]
        e -= (
            self.electricity_cost
            * m.single_stage_feed_compressor_disjunct.compressor_3.electricity_requirement
        )
        e -= (
            self.electricity_cost
            * m.two_stage_feed_compressor_disjunct.compressor_4.electricity_requirement
        )
        e -= (
            self.electricity_cost
            * m.two_stage_feed_compressor_disjunct.compressor_6.electricity_requirement
        )
        e -= self.cooling_cost * m.two_stage_feed_compressor_disjunct.cooler_5.heat_duty
        e -= (
            self.electricity_cost
            * m.single_stage_recycle_compressor_disjunct.compressor_16.electricity_requirement
        )
        e -= (
            self.electricity_cost
            * m.two_stage_recycle_compressor_disjunct.compressor_17.electricity_requirement
        )
        e -= (
            self.electricity_cost
            * m.two_stage_recycle_compressor_disjunct.compressor_19.electricity_requirement
        )
        e -= (
            self.cooling_cost
            * m.two_stage_recycle_compressor_disjunct.cooler_18.heat_duty
        )
        e -= self.cooling_cost * m.cooler_7.heat_duty
        e -= self.heating_cost * m.heater_8.heat_duty
        e -= self.cooling_cost * m.cooler_12.heat_duty
        e -= self.heating_cost * m.heater_14.heat_duty
        e -= self.heating_cost * m.heater_15.heat_duty

        # The operating profit is only earned while the system is available.
        e *= m.system_availability

        e -= (
            self.cheap_reactor_variable_cost
            * m.cheap_reactor.reactor_10.reactor_volume
            * m.cheap_reactor.exists
        )
        e -= self.cheap_reactor_fixed_cost * m.cheap_reactor.exists
        e -= (
            self.expensive_reactor_variable_cost
            * m.expensive_reactor.reactor_9.reactor_volume
            * m.expensive_reactor.exists
        )
        e -= self.expensive_reactor_fixed_cost * m.expensive_reactor.exists

        # Equipment that is always installed: fixed, repair, and capacity costs
        for eq in self.equipment_not_in_disjunct:
            cost = self.cost_b[eq] + self.repair_cost[eq]
            if eq != 11:
                cost += self.cost_a[eq] * getattr(m, "equipment_" + str(eq)).capacity
            e -= cost
        # Equipment in the compressor disjuncts: fixed and repair costs only if
        # the disjunct is selected; capacity is zero otherwise.
        for disjunct, units in (
            (m.single_stage_feed_compressor_disjunct, (3,)),
            (m.two_stage_feed_compressor_disjunct, (4, 5, 6)),
            (m.single_stage_recycle_compressor_disjunct, (16,)),
            (m.two_stage_recycle_compressor_disjunct, (17, 18, 19)),
        ):
            for eq in units:
                e -= (
                    self.cost_b[eq] + self.repair_cost[eq]
                ) * disjunct.binary_indicator_var
                e -= self.cost_a[eq] * getattr(m, "equipment_" + str(eq)).capacity

        m.profit = pe.Expression(expr=e, doc="Profit [$1000/yr]")
        m.objective = pe.Objective(expr=-m.profit)

    def build_compressor(self, block, unit_number):
        """
        Build a compressor block linked to its equipment capacity.

        Parameters
        ----------
        b : Pyomo.Block
            The block to add the compressor to.
        unit_number : int
            The unit number.

        Returns
        -------
        None
        """
        u = unit_number
        m = self.model
        t = m.temps
        p = m.pressures
        in_stream = self.inlet_streams[u]
        out_stream = self.outlet_streams[u]

        b = pe.Block()
        setattr(block, "compressor_" + str(u), b)
        b.p_ratio = pe.Var(bounds=(0, 1.74), doc="Pressure ratio")
        b.electricity_requirement = pe.Var(
            bounds=(0, 50), doc="Compressor load variable [source scale]"
        )

        equipment = getattr(m, "equipment_" + str(u))
        b.capacity_con = pe.Constraint(
            expr=equipment.capacity
            == self.capacity_redundancy * b.electricity_requirement,
            doc="Equipment capacity constraint",
        )

        def _component_balances(_b, _c):
            return m.component_flows[out_stream, _c] == m.component_flows[in_stream, _c]

        b.component_balances = pe.Constraint(
            m.components, rule=_component_balances, doc="Component balances"
        )
        b.t_ratio_con = pe.Constraint(
            expr=t[out_stream] == b.p_ratio * t[in_stream],
            doc="Temperature ratio constraint",
        )
        b.electricity_requirement_con = pe.Constraint(
            expr=(
                b.electricity_requirement
                == self.alpha
                * (b.p_ratio - 1)
                * t[in_stream]
                * m.flows[in_stream]
                / (10.0 * self.eta * self.gamma)
            ),
            doc="Electricity requirement constraint",
        )
        b.p_ratio_con = pe.Constraint(
            expr=p[out_stream] ** self.gamma == b.p_ratio * p[in_stream] ** self.gamma,
            doc="Pressure ratio constraint",
        )

    def build_expansion_valve(self, block, unit_number):
        """
        Build an expansion valve block.

        Parameters
        ----------
        b : Pyomo.Block
            The block to add the expansion valve to.

        unit_number : int
            The unit number.

        Returns
        -------
        None
        """
        u = unit_number
        m = self.model
        t = m.temps
        p = m.pressures
        in_stream = self.inlet_streams[u]
        out_stream = self.outlet_streams[u]
        b = pe.Block()
        setattr(block, "expansion_valve_" + str(u), b)

        def _component_balances(_b, _c):
            return m.component_flows[out_stream, _c] == m.component_flows[in_stream, _c]

        b.component_balances = pe.Constraint(
            m.components, rule=_component_balances, doc="Component balances"
        )
        b.ratio_con = pe.Constraint(
            expr=t[out_stream] * p[in_stream] ** self.gamma
            == t[in_stream] * p[out_stream] ** self.gamma,
            doc="Pressure ratio constraint",
        )
        b.expansion_con = pe.Constraint(
            expr=p[out_stream] <= p[in_stream], doc="Expansion constraint"
        )

    def build_cooler(self, block, unit_number):
        """
        Build a cooler block linked to its equipment capacity.

        Parameters
        ----------
        b : Pyomo.Block
            The block to add the cooler to.
        unit_number : int
            The unit number.

        Returns
        -------
        None
        """
        u = unit_number
        m = self.model
        t = m.temps
        p = m.pressures
        f = m.flows
        in_stream = self.inlet_streams[u]
        out_stream = self.outlet_streams[u]
        b = pe.Block()
        setattr(block, "cooler_" + str(u), b)
        b.heat_duty = pe.Var(bounds=(0, 50), doc="Scaled annual heat duty")

        equipment = getattr(m, "equipment_" + str(u))
        b.capacity_con = pe.Constraint(
            expr=equipment.capacity == self.capacity_redundancy * b.heat_duty,
            doc="Equipment capacity constraint",
        )

        def _component_balances(_b, _c):
            return m.component_flows[out_stream, _c] == m.component_flows[in_stream, _c]

        b.component_balances = pe.Constraint(
            m.components, rule=_component_balances, doc="Component balances"
        )
        b.heat_duty_con = pe.Constraint(
            expr=b.heat_duty
            == self.heat_unit_match
            * self.cp
            * (f[in_stream] * t[in_stream] - f[out_stream] * t[out_stream]),
            doc="Heat duty constraint",
        )
        b.pressure_con = pe.Constraint(
            expr=p[out_stream] == p[in_stream], doc="Pressure constraint"
        )

    def build_heater(self, block, unit_number):
        """
        Build a heater block linked to its equipment capacity.

        Parameters
        ----------
        b : Pyomo.Block
            The block to add the heater to.
        unit_number : int
            The unit number.

        Returns
        -------
        None
        """
        u = unit_number
        m = self.model
        t = m.temps
        p = m.pressures
        f = m.flows
        in_stream = self.inlet_streams[u]
        out_stream = self.outlet_streams[u]
        b = pe.Block()
        setattr(block, "heater_" + str(u), b)
        b.heat_duty = pe.Var(bounds=(0, 50), doc="Scaled annual heat duty")

        equipment = getattr(m, "equipment_" + str(u))
        b.capacity_con = pe.Constraint(
            expr=equipment.capacity == self.capacity_redundancy * b.heat_duty,
            doc="Equipment capacity constraint",
        )

        def _component_balances(_b, _c):
            return m.component_flows[out_stream, _c] == m.component_flows[in_stream, _c]

        b.component_balances = pe.Constraint(
            m.components, rule=_component_balances, doc="Component balances"
        )
        b.heat_duty_con = pe.Constraint(
            expr=b.heat_duty
            == self.heat_unit_match
            * self.cp
            * (f[out_stream] * t[out_stream] - f[in_stream] * t[in_stream]),
            doc="Heat duty constraint",
        )
        b.pressure_con = pe.Constraint(
            expr=p[out_stream] == p[in_stream], doc="Pressure constraint"
        )

    def build_mixer(self, block, unit_number):
        """
        Build a mixer block.

        Parameters
        ----------
        b : Pyomo.Block
            The block to add the mixer to.

        unit_number : int
            The unit number.

        Returns
        -------
        None
        """
        u = unit_number
        m = self.model
        t = m.temps
        p = m.pressures
        f = m.flows
        in_stream1, in_stream2 = self.inlet_streams[u]
        out_stream = self.outlet_streams[u]
        b = pe.Block()
        setattr(block, "mixer_" + str(u), b)

        def _component_balances(_b, _c):
            return (
                m.component_flows[out_stream, _c]
                == m.component_flows[in_stream1, _c] + m.component_flows[in_stream2, _c]
            )

        b.component_balances = pe.Constraint(
            m.components, rule=_component_balances, doc="Component balances"
        )
        b.average_temp = pe.Constraint(
            expr=(
                t[out_stream] * f[out_stream]
                == (t[in_stream1] * f[in_stream1] + t[in_stream2] * f[in_stream2])
            ),
            doc="Average temperature constraint",
        )
        b.pressure_con1 = pe.Constraint(
            expr=p[in_stream1] == p[out_stream], doc="Pressure constraint"
        )
        b.pressure_con2 = pe.Constraint(
            expr=p[in_stream2] == p[out_stream], doc="Pressure constraint"
        )

    def build_splitter(self, block, unit_number):
        """
        Build a splitter block.

        Parameters
        ----------
        b : Pyomo.Block
            The block to add the splitter to.
        unit_number : int
            The unit number.

        Returns
        -------
        None
        """
        u = unit_number
        m = self.model
        t = m.temps
        p = m.pressures
        in_stream = self.inlet_streams[u]
        out_stream1, out_stream2 = self.outlet_streams[u]
        b = pe.Block()
        setattr(block, "splitter_" + str(u), b)
        b.split_fraction = pe.Var(bounds=(0, 1), doc="Split fraction")
        if unit_number == "purge_splitter":
            b.split_fraction.setlb(0.01)
            b.split_fraction.setub(0.99)

        def _split_frac_rule(_b, _c):
            return (
                m.component_flows[out_stream1, _c]
                == b.split_fraction * m.component_flows[in_stream, _c]
            )

        b.split_frac_con = pe.Constraint(
            m.components, rule=_split_frac_rule, doc="Split fraction constraint"
        )

        def _component_balances(_b, _c):
            return (
                m.component_flows[in_stream, _c]
                == m.component_flows[out_stream1, _c]
                + m.component_flows[out_stream2, _c]
            )

        b.component_balances = pe.Constraint(
            m.components, rule=_component_balances, doc="Component balances"
        )
        b.temp_con1 = pe.Constraint(
            expr=t[in_stream] == t[out_stream1], doc="Temperature constraint"
        )
        b.temp_con2 = pe.Constraint(
            expr=t[in_stream] == t[out_stream2], doc="Temperature constraint"
        )
        b.pressure_con1 = pe.Constraint(
            expr=p[in_stream] == p[out_stream1], doc="Pressure constraint"
        )
        b.pressure_con2 = pe.Constraint(
            expr=p[in_stream] == p[out_stream2], doc="Pressure constraint"
        )

    def build_equal_streams(self, block, stream1, stream2):
        """
        Build a block that enforces equality between two streams.

        Parameters
        ----------
        b : Pyomo.Block
            The block to add the constraint to.
        stream1 : Pyomo.Set index
            The first stream to be equal.
        stream2 : Pyomo.Set index
            The second stream to be equal.

        Returns
        -------
        None
        """
        m = self.model
        t = m.temps
        p = m.pressures
        b = pe.Block()
        setattr(block, "equal_streams_" + str(stream1) + "_" + str(stream2), b)

        def _component_balances(_b, _c):
            return m.component_flows[stream2, _c] == m.component_flows[stream1, _c]

        b.component_balances = pe.Constraint(
            m.components, rule=_component_balances, doc="Component balances"
        )
        b.temp_con = pe.Constraint(
            expr=t[stream1] == t[stream2], doc="Temperature constraint"
        )
        b.pressure_con = pe.Constraint(
            expr=p[stream1] == p[stream2], doc="Pressure constraint"
        )

    def build_reactor(self, block, unit_number):
        """
        Build a reactor block with a variable reactor volume.

        Parameters
        ----------
        b : Pyomo.Block
            The block to add the reactor to.
        unit_number : int
            The unit number.
        """
        u = unit_number
        m = self.model
        t = m.temps
        p = m.pressures
        f = m.flows
        component_f = m.component_flows
        in_stream = self.inlet_streams[u]
        out_stream = self.outlet_streams[u]

        b = pe.Block()
        setattr(block, "reactor_" + str(u), b)
        b.consumption_rate = pe.Var(bounds=(0, 5), doc="Consumption rate variable")
        b.conversion = pe.Var(bounds=(0, 0.42), doc="Conversion variable")
        b.equilibrium_conversion = pe.Var(
            bounds=(0, 0.42), doc="Equilibrium conversion variable"
        )
        b.temp = pe.Var(bounds=(3.0, 6.0), doc="Temperature variable [100 K]")
        b.pressure = pe.Var(bounds=(1, 15), doc="Pressure variable [MPa]")
        b.reactor_volume = pe.Var(bounds=(30, 150), doc="Reactor volume [m^3]")
        b.p_sq_inv = pe.Var(doc="Pressure squared inverse variable")
        b.t_inv = pe.Var(doc="Temperature inverse variable")

        key = "H2"

        b.p_sq_inv_con = pe.Constraint(
            expr=b.pressure**2 * b.p_sq_inv == 1,
            doc="Pressure squared inverse constraint",
        )
        b.t_inv_con = pe.Constraint(
            expr=b.temp * b.t_inv == 1, doc="Temperature inverse constraint"
        )
        fbbt(b.p_sq_inv_con)  # just getting bounds on p_sq_inv
        fbbt(b.t_inv_con)  # just getting bounds on t_inv
        b.conversion_consumption_con = pe.Constraint(
            expr=b.consumption_rate == b.conversion * component_f[in_stream, key],
            doc="Conversion consumption constraint",
        )
        b.energy_balance = pe.Constraint(
            expr=(f[in_stream] * t[in_stream] - f[out_stream] * t[out_stream]) * self.cp
            == 0.01 * self.heat_of_reaction * b.consumption_rate,
            doc="Energy balance constraint",
        )
        b.H2_balance = pe.Constraint(
            expr=component_f[out_stream, "H2"]
            == component_f[in_stream, "H2"] - b.consumption_rate,
            doc="H2 balance constraint",
        )
        b.CO_balance = pe.Constraint(
            expr=component_f[out_stream, "CO"]
            == component_f[in_stream, "CO"] - 0.5 * b.consumption_rate,
            doc="CO balance constraint",
        )
        b.CH3OH_balance = pe.Constraint(
            expr=component_f[out_stream, "CH3OH"]
            == component_f[in_stream, "CH3OH"] + 0.5 * b.consumption_rate,
            doc="CH3OH balance constraint",
        )
        b.CH4_balance = pe.Constraint(
            expr=component_f[out_stream, "CH4"] == component_f[in_stream, "CH4"],
            doc="CH4 balance constraint",
        )
        b.eq_conversion_con = pe.Constraint(
            expr=b.equilibrium_conversion
            == self.K * (1 - (self.delta_H * pe.exp(-18 * b.t_inv) * b.p_sq_inv)),
            doc="Equilibrium conversion constraint",
        )
        b.conversion_con = pe.Constraint(
            expr=b.conversion * f[in_stream]
            == b.equilibrium_conversion
            * (1 - pe.exp(-self.volume_conversion[u] * b.reactor_volume))
            * (
                component_f[in_stream, "H2"]
                + component_f[in_stream, "CO"]
                + component_f[in_stream, "CH3OH"]
            ),
            doc="Conversion constraint",
        )
        b.pressure_con1 = pe.Constraint(
            expr=b.pressure == p[in_stream], doc="Pressure constraint"
        )
        b.pressure_con2 = pe.Constraint(
            expr=p[out_stream] == self.reactor_relation * b.pressure,
            doc="Pressure constraint",
        )
        b.temp_con = pe.Constraint(
            expr=b.temp == t[out_stream], doc="Temperature constraint"
        )

    def build_flash(self, block, unit_number):
        """
        Build a flash block.

        Parameters
        ----------
        b : Pyomo.Block
            The block to add the flash to.
        unit_number : int
            The unit number.

        Returns
        -------
        None
        """
        u = unit_number
        m = self.model
        t = m.temps
        p = m.pressures
        f = m.flows
        in_stream = self.inlet_streams[u]
        vapor_stream = self.vapor_outlets[u]
        liquid_stream = self.liquid_outlets[u]
        b = pe.Block()
        setattr(block, "flash_" + str(u), b)

        b.vapor_pressure = pe.Var(
            m.components, bounds=(0.001, 80), doc="Vapor pressure variable [MPa]"
        )
        b.flash_t = pe.Var(bounds=(3, 5), doc="Flash temperature variable [100 K]")
        b.flash_p = pe.Var(bounds=(0.25, 15), doc="Flash pressure variable [MPa]")
        b.vapor_recovery = pe.Var(
            m.components, bounds=(0.01, 0.9999), doc="Vapor recovery variable"
        )

        b.antoine_A = pe.Param(
            m.components, mutable=True, doc="Antoine equation parameter A"
        )
        b.antoine_B = pe.Param(
            m.components, mutable=True, doc="Antoine equation parameter B"
        )
        b.antoine_C = pe.Param(
            m.components, mutable=True, doc="Antoine equation parameter C"
        )
        b.antoine_A["H2"] = 13.6333  # Antoine A coefficient [dimensionless]
        b.antoine_A["CO"] = 14.3686  # Antoine A coefficient [dimensionless]
        b.antoine_A["CH3OH"] = 18.5875  # Antoine A coefficient [dimensionless]
        b.antoine_A["CH4"] = 15.2243  # Antoine A coefficient [dimensionless]
        b.antoine_B["H2"] = 164.9  # Antoine B coefficient [K]
        b.antoine_B["CO"] = 530.22  # Antoine B coefficient [K]
        b.antoine_B["CH3OH"] = 3626.55  # Antoine B coefficient [K]
        b.antoine_B["CH4"] = 897.84  # Antoine B coefficient [K]
        b.antoine_C["H2"] = 3.19  # Antoine C coefficient [K]
        b.antoine_C["CO"] = -13.15  # Antoine C coefficient [K]
        b.antoine_C["CH3OH"] = -34.29  # Antoine C coefficient [K]
        b.antoine_C["CH4"] = -7.16  # Antoine C coefficient [K]

        def _component_balances(_b, _c):
            return (
                m.component_flows[in_stream, _c]
                == m.component_flows[vapor_stream, _c]
                + m.component_flows[liquid_stream, _c]
            )

        b.component_balances = pe.Constraint(
            m.components, rule=_component_balances, doc="Component balances"
        )

        def _antoine(_b, _c):
            return (
                _b.antoine_A[_c]
                - pe.log(self.antoine_unit_trans * _b.vapor_pressure[_c])
            ) * (100 * _b.flash_t - _b.antoine_C[_c]) == _b.antoine_B[_c]

        b.antoine_con = pe.Constraint(
            m.components, rule=_antoine, doc="Antoine equation"
        )

        def _vle(_b, _c):
            return (
                _b.vapor_recovery["H2"]
                * (
                    _b.vapor_recovery[_c] * _b.vapor_pressure["H2"]
                    + (1 - _b.vapor_recovery[_c]) * _b.vapor_pressure[_c]
                )
                == _b.vapor_pressure["H2"] * _b.vapor_recovery[_c]
            )

        b.vle_set = pe.Set(
            initialize=["CO", "CH3OH", "CH4"], ordered=True, doc="VLE set"
        )
        b.vle_con = pe.Constraint(
            b.vle_set, rule=_vle, doc="Vapor Liquid Equilibrium constraint"
        )

        def _vapor_recovery(_b, _c):
            return (
                m.component_flows[vapor_stream, _c]
                == _b.vapor_recovery[_c] * m.component_flows[in_stream, _c]
            )

        b.vapor_recovery_con = pe.Constraint(
            m.components, rule=_vapor_recovery, doc="Vapor recovery constraint"
        )

        b.total_p_con = pe.Constraint(
            expr=b.flash_p * f[liquid_stream]
            == sum(
                b.vapor_pressure[_c] * m.component_flows[liquid_stream, _c]
                for _c in m.components
            ),
            doc="Total pressure constraint",
        )

        b.flash_p_con = pe.ConstraintList()
        b.flash_p_con.add(b.flash_p == p[in_stream])
        b.flash_p_con.add(b.flash_p == p[vapor_stream])
        b.flash_p_con.add(b.flash_p == p[liquid_stream])

        b.flash_t_con = pe.ConstraintList()
        b.flash_t_con.add(b.flash_t == t[in_stream])
        b.flash_t_con.add(b.flash_t == t[vapor_stream])
        b.flash_t_con.add(b.flash_t == t[liquid_stream])

    def build_stream_doesnt_exist_con(self, block, stream):
        m = self.model
        b = pe.Block()
        setattr(block, "stream_doesnt_exist_con_" + str(stream), b)
        b.zero_flow_con = pe.Constraint(expr=m.flows[stream] == 0)

        def _zero_component_flows(_b, _c):
            return m.component_flows[stream, _c] == 0

        b.zero_component_flows_con = pe.Constraint(
            m.components,
            rule=_zero_component_flows,
            doc="Zero component flows constraint",
        )
        b.fixed_temp_con = pe.Constraint(
            expr=m.temps[stream] == 3, doc="Fixed temperature constraint"
        )
        b.fixed_pressure_con = pe.Constraint(
            expr=m.pressures[stream] == 1, doc="Fixed pressure constraint"
        )

    def add_reliability_capacity(self, block, equipment_number, selected):
        """
        Add availability and capacity constraints for one equipment unit.

        The unit's ``capacity`` and ``availability`` variables live on a
        model-level block ``m.equipment_<number>``, created on first use.

        Parameters
        ----------
        block : Pyomo.Block or None
            Disjunct that holds the constraints, or None for units that are
            always part of the flowsheet (constraints go on the model-level
            equipment block).
        equipment_number : int
            The unit number.
        selected : bool
            True if the unit is installed when ``block`` is active. False if
            ``block`` is the alternative disjunct in which the unit does not
            exist (availability 1 and zero capacity).

        Returns
        -------
        None
        """
        m = self.model
        name = "equipment_" + str(equipment_number)
        if hasattr(m, name):
            b = getattr(m, name)
        else:
            b = pe.Block()
            setattr(m, name, b)
            b.capacity = pe.Var(
                within=pe.NonNegativeReals,
                bounds=(0, 60),
                doc="Equipment capacity [source scale of load or heat duty]",
            )
            b.availability = pe.Var(
                within=pe.NonNegativeReals,
                bounds=(0, 1),
                doc="Equipment availability [dimensionless]",
            )
        if block is None:
            b1 = b
        else:
            b1 = pe.Block()
            setattr(block, name, b1)
        if selected:
            b1.available_con = pe.Constraint(
                expr=b.availability == self.probability[equipment_number],
                doc="Availability of an installed unit",
            )
        else:
            b1.available_con = pe.Constraint(
                expr=b.availability == 1, doc="Availability of a unit not installed"
            )
            b1.zero_capacity_con = pe.Constraint(
                expr=b.capacity == 0, doc="Capacity of a unit not installed"
            )

    def add_reliability_capacity_for_disjunct(
        self, block, equipment_list_of_this_disjunct, equipment_list_of_that_disjunct
    ):
        """
        Add availability and capacity constraints for a compressor disjunct.

        Parameters
        ----------
        block : Pyomo.gdp.Disjunct
            The compressor disjunct.
        equipment_list_of_this_disjunct : list of int
            Units installed when this disjunct is selected.
        equipment_list_of_that_disjunct : list of int
            Units of the alternative disjunct, absent when this one is selected.

        Returns
        -------
        None
        """
        for eq in equipment_list_of_this_disjunct:
            self.add_reliability_capacity(block, eq, True)
        for eq in equipment_list_of_that_disjunct:
            self.add_reliability_capacity(block, eq, False)


def build_model():
    """Build the availability-aware methanol synthesis process model.

    Returns
    -------
    Pyomo.ConcreteModel
        Methanol synthesis process model with equipment availability, with
        Big-M suffixes populated on active disjunct constraints.
    """
    m = MethanolModelAvailability().model
    for _d in m.component_data_objects(
        gdp.Disjunct, descend_into=True, active=True, sort=True
    ):
        _d.BigM = pe.Suffix()
        for _c in _d.component_data_objects(
            pe.Constraint, descend_into=True, active=True, sort=True
        ):
            lb, ub = compute_bounds_on_expr(_c.body)
            _d.BigM[_c] = max(abs(lb), abs(ub))

    return m


def solve_with_gdp_opt(
    algorithm="LOA",
    mip_solver="gams",
    nlp_solver="gams",
    tee=True,
    mip_solver_args=None,
    nlp_solver_args=None,
    return_results=False,
    print_results=True,
    **solve_options,
):
    """
    Solve the availability-aware methanol model with GDPopt.

    Parameters
    ----------
    algorithm : str
        GDPopt algorithm name. Defaults to ``'LOA'``.
    mip_solver : str
        Solver used for GDPopt MIP master problems.
    nlp_solver : str
        Solver used for GDPopt NLP subproblems.
    tee : bool
        Stream GDPopt output during the solve.
    mip_solver_args : dict, optional
        Additional keyword arguments for the MIP solver.
    nlp_solver_args : dict, optional
        Additional keyword arguments for the NLP solver.
    return_results : bool
        If True, return ``(model, results)``. Otherwise return only the model.
    print_results : bool
        Print active disjunct names and the solver result object after solving.
    **solve_options
        Additional keyword arguments passed through to ``GDPopt.solve()``.

    Returns
    -------
    m : Pyomo.ConcreteModel
        The Pyomo model.
    res : SolverResults, optional
        GDPopt solve results when ``return_results`` is True.
    """

    m = build_model()
    opt = pe.SolverFactory("gdpopt")
    solve_kwargs = {
        "algorithm": algorithm,
        "mip_solver": mip_solver,
        "nlp_solver": nlp_solver,
        "tee": tee,
    }
    if mip_solver_args is not None:
        solve_kwargs["mip_solver_args"] = mip_solver_args
    if nlp_solver_args is not None:
        solve_kwargs["nlp_solver_args"] = nlp_solver_args
    solve_kwargs.update(solve_options)

    res = opt.solve(m, **solve_kwargs)

    if print_results:
        for d in m.component_data_objects(
            ctype=gdp.Disjunct, active=True, sort=True, descend_into=True
        ):
            if d.indicator_var.value is True:
                print(d.name)
        print(res)

    if return_results:
        return m, res
    return m


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, filename="out.log")
    from pyomo.util.model_size import build_model_size_report

    m = solve_with_gdp_opt()
    print(build_model_size_report(m))
