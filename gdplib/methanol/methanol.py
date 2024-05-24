import pyomo.environ as pe
import sys
from pyomo.contrib.fbbt.fbbt import fbbt, compute_bounds_on_expr
import logging
import pyomo.gdp as gdp


assert sys.version_info.major == 3
assert sys.version_info.minor >= 6


class InfeasibleError(Exception):
    pass


def fix_vars_with_equal_bounds(m, tol=1e-8):
    for v in m.component_data_objects(pe.Var, descend_into=True):
        if v.is_fixed():
            continue
        lb = pe.value(v.lb)
        ub = pe.value(v.ub)
        if lb is None or ub is None:
            continue
        if lb > ub + tol:
            raise InfeasibleError(
                'Variable lb is larger than ub: {0}    lb: {1}    ub: {2}'.format(
                    v.name, lb, ub
                )
            )
        elif abs(ub - lb) <= tol:
            v.fix(0.5 * (lb + ub))


class MethanolModel(object):
    def __init__(self):

        self.model = m = pe.ConcreteModel()  # main model

        self.alpha = 0.72  # compressor coefficient
        self.eta = 0.75  # compressor efficiency
        self.gamma = 0.23077  # ratio of constant pressure heat capacity to constant volume heat capacity
        self.cp = 35.0  # heat capacity
        self.heat_of_reaction = -15
        self.volume_conversion = dict()
        self.volume_conversion[9] = 0.1
        self.volume_conversion[10] = 0.05
        self.reactor_volume = 100
        self.electricity_cost = 0.255
        self.cooling_cost = 700
        self.heating_cost = 8000
        self.purity_demand = 0.9  # purity demand in product stream
        self.demand = 1.0  # flowrate restriction on product flow
        self.flow_feed_lb = 0.5
        self.flow_feed_ub = 5
        self.flow_feed_temp = 3
        self.flow_feed_pressure = 1
        self.cost_flow_1 = 795.6
        self.cost_flow_2 = 1009.8
        self.price_of_product = 7650
        self.price_of_byproduct = 642.6
        self.cheap_reactor_fixed_cost = 100
        self.cheap_reactor_variable_cost = 5
        self.expensive_reactor_fixed_cost = 250
        self.expensive_reactor_variable_cost = 10
        self.heat_unit_match = 0.00306
        self.capacity_redundancy = 1.2
        self.antoine_unit_trans = 7500.6168
        self.K = 0.415
        self.delta_H = 26.25
        self.reactor_relation = 0.9
        self.purity_demand = 0.9
        self.fix_electricity_cost = 175
        self.two_stage_fix_cost = 50

        m.streams = pe.Set(initialize=list(range(1, 34)), ordered=True)
        m.components = pe.Set(initialize=['H2', 'CO', 'CH3OH', 'CH4'], ordered=True)
        m.flows = pe.Var(m.streams, bounds=(0, 20))
        m.temps = pe.Var(m.streams, bounds=(3, 9))
        m.pressures = pe.Var(m.streams, bounds=(0.1, 15))
        m.component_flows = pe.Var(m.streams, m.components, bounds=(0, 20))

        flow_1 = dict()
        flow_1['H2'] = 0.6
        flow_1['CO'] = 0.25
        flow_1['CH4'] = 0.15
        m.flow_1_composition = pe.Param(m.components, initialize=flow_1, default=0)
        flow_2 = dict()
        flow_2['H2'] = 0.65
        flow_2['CO'] = 0.30
        flow_2['CH4'] = 0.05
        m.flow_2_composition = pe.Param(m.components, initialize=flow_2, default=0)

        m.pressures[13].setlb(2.5)
        m.temps[13].setlb(4.23)
        m.temps[13].setub(8.73)
        m.temps[18].setlb(5.23)
        m.temps[18].setub(8.73)
        m.flows[22].setlb(0.1)
        m.flows[22].setub(1.0)
        m.temps[23].fix(4)
        m.temps[25].fix(4)

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
        self.inlet_streams['feed_mixer'] = (1, 2)
        self.inlet_streams['feed_splitter'] = 3
        self.inlet_streams['compressed_feed_mixer'] = (6, 9)
        self.inlet_streams['recycle_feed_mixer'] = (10, 33)
        self.inlet_streams['reactor_feed_splitter'] = 13
        self.inlet_streams['reactor_product_mixer'] = (16, 17)
        self.inlet_streams['purge_splitter'] = 21
        self.inlet_streams['recycle_compressor_splitter'] = 26
        self.inlet_streams['recycle_compressor_mixer'] = (29, 32)

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
        self.outlet_streams['feed_mixer'] = 3
        self.outlet_streams['feed_splitter'] = (4, 5)
        self.outlet_streams['compressed_feed_mixer'] = 10
        self.outlet_streams['recycle_feed_mixer'] = 11
        self.outlet_streams['reactor_feed_splitter'] = (14, 15)
        self.outlet_streams['reactor_product_mixer'] = 18
        self.outlet_streams['purge_splitter'] = (26, 24)
        self.outlet_streams['recycle_compressor_splitter'] = (27, 28)
        self.outlet_streams['recycle_compressor_mixer'] = 33

        self.vapor_outlets[13] = 21
        self.liquid_outlets[13] = 22

        def _total_flow(_m, _s):
            return _m.flows[_s] == sum(
                _m.component_flows[_s, _c] for _c in _m.components
            )

        m.total_flow_con = pe.Constraint(m.streams, rule=_total_flow)

        m.purity_con = pe.Constraint(
            expr=m.component_flows[23, 'CH3OH'] >= self.purity_demand * m.flows[23]
        )

        # ************************************
        # Feed
        # ************************************
        m.cheap_feed_disjunct = gdp.Disjunct()
        self.build_equal_streams(m.cheap_feed_disjunct, 1, 3)
        self.build_stream_doesnt_exist_con(m.cheap_feed_disjunct, 2)
        m.cheap_feed_disjunct.feed_cons = c = pe.ConstraintList()
        c.add(m.component_flows[1, 'H2'] == m.flow_1_composition['H2'] * m.flows[1])
        c.add(m.component_flows[1, 'CO'] == m.flow_1_composition['CO'] * m.flows[1])
        c.add(m.component_flows[1, 'CH4'] == m.flow_1_composition['CH4'] * m.flows[1])
        c.add(m.flows[1] >= self.flow_feed_lb)
        c.add(m.flows[1] <= self.flow_feed_ub)
        c.add(m.temps[1] == self.flow_feed_temp)
        c.add(m.pressures[1] == self.flow_feed_pressure)

        m.expensive_feed_disjunct = gdp.Disjunct()
        self.build_equal_streams(m.expensive_feed_disjunct, 2, 3)
        self.build_stream_doesnt_exist_con(m.expensive_feed_disjunct, 1)
        m.expensive_feed_disjunct.feed_cons = c = pe.ConstraintList()
        c.add(m.component_flows[2, 'H2'] == m.flow_2_composition['H2'] * m.flows[2])
        c.add(m.component_flows[2, 'CO'] == m.flow_2_composition['CO'] * m.flows[2])
        c.add(m.component_flows[2, 'CH4'] == m.flow_2_composition['CH4'] * m.flows[2])
        c.add(m.flows[2] >= self.flow_feed_lb)
        c.add(m.flows[2] <= self.flow_feed_ub)
        c.add(m.temps[2] == self.flow_feed_temp)
        c.add(m.pressures[2] == self.flow_feed_pressure)

        m.feed_disjunctions = gdp.Disjunction(
            expr=[m.cheap_feed_disjunct, m.expensive_feed_disjunct]
        )

        # ************************************
        # Feed compressors
        # ************************************
        m.single_stage_feed_compressor_disjunct = gdp.Disjunct()
        self.build_equal_streams(m.single_stage_feed_compressor_disjunct, 3, 4)
        self.build_stream_doesnt_exist_con(m.single_stage_feed_compressor_disjunct, 5)
        self.build_stream_doesnt_exist_con(m.single_stage_feed_compressor_disjunct, 7)
        self.build_stream_doesnt_exist_con(m.single_stage_feed_compressor_disjunct, 8)
        self.build_stream_doesnt_exist_con(m.single_stage_feed_compressor_disjunct, 9)
        self.build_equal_streams(m.single_stage_feed_compressor_disjunct, 6, 10)
        self.build_compressor(m.single_stage_feed_compressor_disjunct, 3)

        m.two_stage_feed_compressor_disjunct = gdp.Disjunct()
        self.build_equal_streams(m.two_stage_feed_compressor_disjunct, 3, 5)
        self.build_equal_streams(m.two_stage_feed_compressor_disjunct, 9, 10)
        self.build_stream_doesnt_exist_con(m.two_stage_feed_compressor_disjunct, 4)
        self.build_stream_doesnt_exist_con(m.two_stage_feed_compressor_disjunct, 6)
        self.build_compressor(m.two_stage_feed_compressor_disjunct, 4)
        self.build_cooler(m.two_stage_feed_compressor_disjunct, 5)
        self.build_compressor(m.two_stage_feed_compressor_disjunct, 6)
        m.two_stage_feed_compressor_disjunct.equal_electric_requirements = pe.Constraint(
            expr=m.two_stage_feed_compressor_disjunct.compressor_4.electricity_requirement
            == m.two_stage_feed_compressor_disjunct.compressor_6.electricity_requirement
        )
        m.two_stage_feed_compressor_disjunct.exists = pe.Var(bounds=(0, 1))
        m.two_stage_feed_compressor_disjunct.exists_con = pe.Constraint(
            expr=m.two_stage_feed_compressor_disjunct.exists == 1
        )

        m.feed_compressor_disjunction = gdp.Disjunction(
            expr=[
                m.single_stage_feed_compressor_disjunct,
                m.two_stage_feed_compressor_disjunct,
            ]
        )

        self.build_mixer(m, 'recycle_feed_mixer')
        self.build_cooler(m, 7)
        self.build_heater(m, 8)

        # ************************************
        # Reactors
        # ************************************
        m.expensive_reactor = gdp.Disjunct()
        self.build_equal_streams(m.expensive_reactor, 13, 15)
        self.build_equal_streams(m.expensive_reactor, 17, 18)
        self.build_stream_doesnt_exist_con(m.expensive_reactor, 14)
        self.build_stream_doesnt_exist_con(m.expensive_reactor, 16)
        self.build_reactor(m.expensive_reactor, 9)
        m.expensive_reactor.exists = pe.Var(bounds=(0, 1))
        m.expensive_reactor.exists_con = pe.Constraint(
            expr=m.expensive_reactor.exists == 1
        )
        m.expensive_reactor.composition_cons = c = pe.ConstraintList()
        for _comp in m.components:
            c.add(m.component_flows[17, _comp] >= 0.01)

        m.cheap_reactor = gdp.Disjunct()
        self.build_equal_streams(m.cheap_reactor, 13, 14)
        self.build_equal_streams(m.cheap_reactor, 16, 18)
        self.build_stream_doesnt_exist_con(m.cheap_reactor, 15)
        self.build_stream_doesnt_exist_con(m.cheap_reactor, 17)
        self.build_reactor(m.cheap_reactor, 10)
        m.cheap_reactor.exists = pe.Var(bounds=(0, 1))
        m.cheap_reactor.exists_con = pe.Constraint(expr=m.cheap_reactor.exists == 1)
        m.cheap_reactor.composition_cons = c = pe.ConstraintList()
        for _comp in m.components:
            c.add(m.component_flows[16, _comp] >= 0.01)

        m.reactor_disjunction = gdp.Disjunction(
            expr=[m.expensive_reactor, m.cheap_reactor]
        )

        self.build_expansion_valve(m, 11)
        self.build_cooler(m, 12)
        self.build_flash(m, 13)
        self.build_heater(m, 14)
        self.build_splitter(m, 'purge_splitter')
        self.build_heater(m, 15)

        # ************************************
        # Recycle compressors
        # ************************************
        m.single_stage_recycle_compressor_disjunct = gdp.Disjunct()
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
        self.build_equal_streams(m.two_stage_recycle_compressor_disjunct, 26, 28)
        self.build_equal_streams(m.two_stage_recycle_compressor_disjunct, 32, 33)
        self.build_stream_doesnt_exist_con(m.two_stage_recycle_compressor_disjunct, 27)
        self.build_stream_doesnt_exist_con(m.two_stage_recycle_compressor_disjunct, 29)
        self.build_compressor(m.two_stage_recycle_compressor_disjunct, 17)
        self.build_cooler(m.two_stage_recycle_compressor_disjunct, 18)
        self.build_compressor(m.two_stage_recycle_compressor_disjunct, 19)
        m.two_stage_recycle_compressor_disjunct.equal_electric_requirements = pe.Constraint(
            expr=m.two_stage_recycle_compressor_disjunct.compressor_17.electricity_requirement
            == m.two_stage_recycle_compressor_disjunct.compressor_19.electricity_requirement
        )
        m.two_stage_recycle_compressor_disjunct.exists = pe.Var(bounds=(0, 1))
        m.two_stage_recycle_compressor_disjunct.exists_con = pe.Constraint(
            expr=m.two_stage_recycle_compressor_disjunct.exists == 1
        )

        m.recycle_compressor_disjunction = gdp.Disjunction(
            expr=[
                m.single_stage_recycle_compressor_disjunct,
                m.two_stage_recycle_compressor_disjunct,
            ]
        )

        # ************************************
        # Objective
        # ************************************

        e = 0
        e -= self.cost_flow_1 * m.flows[1]
        e -= self.cost_flow_2 * m.flows[2]
        e += self.price_of_product * m.flows[23]
        e += self.price_of_byproduct * m.flows[25]
        e -= (
            self.cheap_reactor_variable_cost
            * self.reactor_volume
            * m.cheap_reactor.exists
        )
        e -= self.cheap_reactor_fixed_cost * m.cheap_reactor.exists
        e -= (
            self.expensive_reactor_variable_cost
            * self.reactor_volume
            * m.expensive_reactor.exists
        )
        e -= self.expensive_reactor_fixed_cost * m.expensive_reactor.exists
        e -= (
            (self.fix_electricity_cost + self.electricity_cost)
            * m.single_stage_feed_compressor_disjunct.compressor_3.electricity_requirement
        )
        e -= self.two_stage_fix_cost * m.two_stage_feed_compressor_disjunct.exists
        e -= (
            (self.fix_electricity_cost + self.electricity_cost)
            * m.two_stage_feed_compressor_disjunct.compressor_4.electricity_requirement
        )
        e -= (
            (self.fix_electricity_cost + self.electricity_cost)
            * m.two_stage_feed_compressor_disjunct.compressor_6.electricity_requirement
        )
        e -= self.cooling_cost * m.two_stage_feed_compressor_disjunct.cooler_5.heat_duty
        e -= (
            (self.fix_electricity_cost + self.electricity_cost)
            * m.single_stage_recycle_compressor_disjunct.compressor_16.electricity_requirement
        )
        e -= self.two_stage_fix_cost * m.two_stage_recycle_compressor_disjunct.exists
        e -= (
            (self.fix_electricity_cost + self.electricity_cost)
            * m.two_stage_recycle_compressor_disjunct.compressor_17.electricity_requirement
        )
        e -= (
            (self.fix_electricity_cost + self.electricity_cost)
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
        m.objective = pe.Objective(expr=-e)

    def build_compressor(self, block, unit_number):
        u = unit_number
        m = self.model
        t = m.temps
        p = m.pressures
        in_stream = self.inlet_streams[u]
        out_stream = self.outlet_streams[u]

        b = pe.Block()
        setattr(block, 'compressor_' + str(u), b)
        b.p_ratio = pe.Var(bounds=(0, 1.74))
        b.electricity_requirement = pe.Var(bounds=(0, 50))

        def _component_balances(_b, _c):
            return m.component_flows[out_stream, _c] == m.component_flows[in_stream, _c]

        b.component_balances = pe.Constraint(m.components, rule=_component_balances)
        b.t_ratio_con = pe.Constraint(expr=t[out_stream] == b.p_ratio * t[in_stream])
        b.electricity_requirement_con = pe.Constraint(
            expr=(
                b.electricity_requirement
                == self.alpha
                * (b.p_ratio - 1)
                * t[in_stream]
                * m.flows[in_stream]
                / (10.0 * self.eta * self.gamma)
            )
        )
        b.p_ratio_con = pe.Constraint(
            expr=p[out_stream] ** self.gamma == b.p_ratio * p[in_stream] ** self.gamma
        )

    def build_expansion_valve(self, block, unit_number):
        u = unit_number
        m = self.model
        t = m.temps
        p = m.pressures
        in_stream = self.inlet_streams[u]
        out_stream = self.outlet_streams[u]
        b = pe.Block()
        setattr(block, 'expansion_valve_' + str(u), b)

        def _component_balances(_b, _c):
            return m.component_flows[out_stream, _c] == m.component_flows[in_stream, _c]

        b.component_balances = pe.Constraint(m.components, rule=_component_balances)
        b.ratio_con = pe.Constraint(
            expr=t[out_stream] * p[in_stream] ** self.gamma
            == t[in_stream] * p[out_stream] ** self.gamma
        )
        b.expansion_con = pe.Constraint(expr=p[out_stream] <= p[in_stream])

    def build_cooler(self, block, unit_number):
        u = unit_number
        m = self.model
        t = m.temps
        p = m.pressures
        f = m.flows
        in_stream = self.inlet_streams[u]
        out_stream = self.outlet_streams[u]
        b = pe.Block()
        setattr(block, 'cooler_' + str(u), b)
        b.heat_duty = pe.Var(bounds=(0, 50))

        def _component_balances(_b, _c):
            return m.component_flows[out_stream, _c] == m.component_flows[in_stream, _c]

        b.component_balances = pe.Constraint(m.components, rule=_component_balances)
        b.heat_duty_con = pe.Constraint(
            expr=b.heat_duty
            == self.heat_unit_match
            * self.cp
            * (f[in_stream] * t[in_stream] - f[out_stream] * t[out_stream])
        )
        b.pressure_con = pe.Constraint(expr=p[out_stream] == p[in_stream])

    def build_heater(self, block, unit_number):
        u = unit_number
        m = self.model
        t = m.temps
        p = m.pressures
        f = m.flows
        in_stream = self.inlet_streams[u]
        out_stream = self.outlet_streams[u]
        b = pe.Block()
        setattr(block, 'heater_' + str(u), b)
        b.heat_duty = pe.Var(bounds=(0, 50))

        def _component_balances(_b, _c):
            return m.component_flows[out_stream, _c] == m.component_flows[in_stream, _c]

        b.component_balances = pe.Constraint(m.components, rule=_component_balances)
        b.heat_duty_con = pe.Constraint(
            expr=b.heat_duty
            == self.heat_unit_match
            * self.cp
            * (f[out_stream] * t[out_stream] - f[in_stream] * t[in_stream])
        )
        b.pressure_con = pe.Constraint(expr=p[out_stream] == p[in_stream])

    def build_mixer(self, block, unit_number):
        u = unit_number
        m = self.model
        t = m.temps
        p = m.pressures
        f = m.flows
        in_stream1, in_stream2 = self.inlet_streams[u]
        out_stream = self.outlet_streams[u]
        b = pe.Block()
        setattr(block, 'mixer_' + str(u), b)

        def _component_balances(_b, _c):
            return (
                m.component_flows[out_stream, _c]
                == m.component_flows[in_stream1, _c] + m.component_flows[in_stream2, _c]
            )

        b.component_balances = pe.Constraint(m.components, rule=_component_balances)
        b.average_temp = pe.Constraint(
            expr=(
                t[out_stream] * f[out_stream]
                == (t[in_stream1] * f[in_stream1] + t[in_stream2] * f[in_stream2])
            )
        )
        b.pressure_con1 = pe.Constraint(expr=p[in_stream1] == p[out_stream])
        b.pressure_con2 = pe.Constraint(expr=p[in_stream2] == p[out_stream])

    def build_splitter(self, block, unit_number):
        u = unit_number
        m = self.model
        t = m.temps
        p = m.pressures
        in_stream = self.inlet_streams[u]
        out_stream1, out_stream2 = self.outlet_streams[u]
        b = pe.Block()
        setattr(block, 'splitter_' + str(u), b)
        b.split_fraction = pe.Var(bounds=(0, 1))
        if unit_number == 'purge_splitter':
            b.split_fraction.setlb(0.01)
            b.split_fraction.setub(0.99)

        def _split_frac_rule(_b, _c):
            return (
                m.component_flows[out_stream1, _c]
                == b.split_fraction * m.component_flows[in_stream, _c]
            )

        b.split_frac_con = pe.Constraint(m.components, rule=_split_frac_rule)

        def _component_balances(_b, _c):
            return (
                m.component_flows[in_stream, _c]
                == m.component_flows[out_stream1, _c]
                + m.component_flows[out_stream2, _c]
            )

        b.component_balances = pe.Constraint(m.components, rule=_component_balances)
        b.temp_con1 = pe.Constraint(expr=t[in_stream] == t[out_stream1])
        b.temp_con2 = pe.Constraint(expr=t[in_stream] == t[out_stream2])
        b.pressure_con1 = pe.Constraint(expr=p[in_stream] == p[out_stream1])
        b.pressure_con2 = pe.Constraint(expr=p[in_stream] == p[out_stream2])

    def build_equal_streams(self, block, stream1, stream2):
        m = self.model
        t = m.temps
        p = m.pressures
        b = pe.Block()
        setattr(block, 'equal_streams_' + str(stream1) + '_' + str(stream2), b)

        def _component_balances(_b, _c):
            return m.component_flows[stream2, _c] == m.component_flows[stream1, _c]

        b.component_balances = pe.Constraint(m.components, rule=_component_balances)
        b.temp_con = pe.Constraint(expr=t[stream1] == t[stream2])
        b.pressure_con = pe.Constraint(expr=p[stream1] == p[stream2])

    def build_reactor(self, block, unit_number):
        u = unit_number
        m = self.model
        t = m.temps
        p = m.pressures
        f = m.flows
        component_f = m.component_flows
        in_stream = self.inlet_streams[u]
        out_stream = self.outlet_streams[u]

        b = pe.Block()
        setattr(block, 'reactor_' + str(u), b)
        b.consumption_rate = pe.Var(bounds=(0, 5))
        b.conversion = pe.Var(bounds=(0, 0.42))
        b.equilibrium_conversion = pe.Var(bounds=(0, 0.42))
        b.temp = pe.Var(bounds=(3, 8.73))
        b.pressure = pe.Var(bounds=(1, 15))
        b.p_sq_inv = pe.Var()
        b.t_inv = pe.Var()

        key = 'H2'

        b.p_sq_inv_con = pe.Constraint(expr=b.pressure**2 * b.p_sq_inv == 1)
        b.t_inv_con = pe.Constraint(expr=b.temp * b.t_inv == 1)
        fbbt(b.p_sq_inv_con)  # just getting bounds on p_sq_inv
        fbbt(b.t_inv_con)  # just getting bounds on t_inv
        b.conversion_consumption_con = pe.Constraint(
            expr=b.consumption_rate == b.conversion * component_f[in_stream, key]
        )
        b.energy_balance = pe.Constraint(
            expr=(f[in_stream] * t[in_stream] - f[out_stream] * t[out_stream]) * self.cp
            == 0.01 * self.heat_of_reaction * b.consumption_rate
        )
        b.H2_balance = pe.Constraint(
            expr=component_f[out_stream, 'H2']
            == component_f[in_stream, 'H2'] - b.consumption_rate
        )
        b.CO_balance = pe.Constraint(
            expr=component_f[out_stream, 'CO']
            == component_f[in_stream, 'CO'] - 0.5 * b.consumption_rate
        )
        b.CH3OH_balance = pe.Constraint(
            expr=component_f[out_stream, 'CH3OH']
            == component_f[in_stream, 'CH3OH'] + 0.5 * b.consumption_rate
        )
        b.CH4_balance = pe.Constraint(
            expr=component_f[out_stream, 'CH4'] == component_f[in_stream, 'CH4']
        )
        b.eq_conversion_con = pe.Constraint(
            expr=b.equilibrium_conversion
            == self.K * (1 - (self.delta_H * pe.exp(-18 * b.t_inv) * b.p_sq_inv))
        )
        b.conversion_con = pe.Constraint(
            expr=b.conversion * f[in_stream]
            == b.equilibrium_conversion
            * (1 - pe.exp(-self.volume_conversion[u] * self.reactor_volume))
            * (
                component_f[in_stream, 'H2']
                + component_f[in_stream, 'CO']
                + component_f[in_stream, 'CH3OH']
            )
        )
        b.pressure_con1 = pe.Constraint(expr=b.pressure == p[in_stream])
        b.pressure_con2 = pe.Constraint(
            expr=p[out_stream] == self.reactor_relation * b.pressure
        )
        b.temp_con = pe.Constraint(expr=b.temp == t[out_stream])

    def build_flash(self, block, unit_number):
        u = unit_number
        m = self.model
        t = m.temps
        p = m.pressures
        f = m.flows
        in_stream = self.inlet_streams[u]
        vapor_stream = self.vapor_outlets[u]
        liquid_stream = self.liquid_outlets[u]
        b = pe.Block()
        setattr(block, 'flash_' + str(u), b)

        b.vapor_pressure = pe.Var(m.components, bounds=(0.001, 80))
        b.flash_t = pe.Var(bounds=(3, 5))
        b.flash_p = pe.Var(bounds=(0.25, 15))
        b.vapor_recovery = pe.Var(m.components, bounds=(0.01, 0.9999))

        b.antoine_A = pe.Param(m.components, mutable=True)
        b.antoine_B = pe.Param(m.components, mutable=True)
        b.antoine_C = pe.Param(m.components, mutable=True)
        b.antoine_A['H2'] = 13.6333
        b.antoine_A['CO'] = 14.3686
        b.antoine_A['CH3OH'] = 18.5875
        b.antoine_A['CH4'] = 15.2243
        b.antoine_B['H2'] = 164.9
        b.antoine_B['CO'] = 530.22
        b.antoine_B['CH3OH'] = 3626.55
        b.antoine_B['CH4'] = 897.84
        b.antoine_C['H2'] = 3.19
        b.antoine_C['CO'] = -13.15
        b.antoine_C['CH3OH'] = -34.29
        b.antoine_C['CH4'] = -7.16

        def _component_balances(_b, _c):
            return (
                m.component_flows[in_stream, _c]
                == m.component_flows[vapor_stream, _c]
                + m.component_flows[liquid_stream, _c]
            )

        b.component_balances = pe.Constraint(m.components, rule=_component_balances)

        def _antoine(_b, _c):
            return (
                _b.antoine_A[_c]
                - pe.log(self.antoine_unit_trans * _b.vapor_pressure[_c])
            ) * (100 * _b.flash_t - _b.antoine_C[_c]) == _b.antoine_B[_c]

        b.antoine_con = pe.Constraint(m.components, rule=_antoine)

        def _vle(_b, _c):
            return (
                _b.vapor_recovery['H2']
                * (
                    _b.vapor_recovery[_c] * _b.vapor_pressure['H2']
                    + (1 - _b.vapor_recovery[_c]) * _b.vapor_pressure[_c]
                )
                == _b.vapor_pressure['H2'] * _b.vapor_recovery[_c]
            )

        b.vle_set = pe.Set(initialize=['CO', 'CH3OH', 'CH4'], ordered=True)
        b.vle_con = pe.Constraint(b.vle_set, rule=_vle)

        def _vapor_recovery(_b, _c):
            return (
                m.component_flows[vapor_stream, _c]
                == _b.vapor_recovery[_c] * m.component_flows[in_stream, _c]
            )

        b.vapor_recovery_con = pe.Constraint(m.components, rule=_vapor_recovery)

        b.total_p_con = pe.Constraint(
            expr=b.flash_p * f[liquid_stream]
            == sum(
                b.vapor_pressure[_c] * m.component_flows[liquid_stream, _c]
                for _c in m.components
            )
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
        setattr(block, 'stream_doesnt_exist_con_' + str(stream), b)
        b.zero_flow_con = pe.Constraint(expr=m.flows[stream] == 0)

        def _zero_component_flows(_b, _c):
            return m.component_flows[stream, _c] == 0

        b.zero_component_flows_con = pe.Constraint(
            m.components, rule=_zero_component_flows
        )
        b.fixed_temp_con = pe.Constraint(expr=m.temps[stream] == 3)
        b.fixed_pressure_con = pe.Constraint(expr=m.pressures[stream] == 1)


def enumerate_solutions():
    import time

    feed_choices = ['cheap', 'expensive']
    feed_compressor_choices = ['single_stage', 'two_stage']
    reactor_choices = ['cheap', 'expensive']
    recycle_compressor_choices = ['single_stage', 'two_stage']

    print(
        '{0:<20}{1:<20}{2:<20}{3:<20}{4:<20}{5:<20}'.format(
            'feed choice',
            'feed compressor',
            'reactor choice',
            'recycle compressor',
            'termination cond',
            'profit',
        )
    )
    since = time.time()
    for feed_choice in feed_choices:
        for feed_compressor_choice in feed_compressor_choices:
            for reactor_choice in reactor_choices:
                for recycle_compressor_choice in recycle_compressor_choices:
                    m = MethanolModel()
                    m = m.model
                    for _d in m.component_data_objects(
                        gdp.Disjunct, descend_into=True, active=True, sort=True
                    ):
                        _d.BigM = pe.Suffix()
                        for _c in _d.component_data_objects(
                            pe.Constraint, descend_into=True, active=True, sort=True
                        ):
                            lb, ub = compute_bounds_on_expr(_c.body)
                            _d.BigM[_c] = max(abs(lb), abs(ub))

                    if feed_choice == 'cheap':
                        m.cheap_feed_disjunct.indicator_var.fix(1)
                        m.expensive_feed_disjunct.indicator_var.fix(0)
                    else:
                        m.cheap_feed_disjunct.indicator_var.fix(0)
                        m.expensive_feed_disjunct.indicator_var.fix(1)

                    if feed_compressor_choice == 'single_stage':
                        m.single_stage_feed_compressor_disjunct.indicator_var.fix(1)
                        m.two_stage_feed_compressor_disjunct.indicator_var.fix(0)
                    else:
                        m.single_stage_feed_compressor_disjunct.indicator_var.fix(0)
                        m.two_stage_feed_compressor_disjunct.indicator_var.fix(1)

                    if reactor_choice == 'cheap':
                        m.cheap_reactor.indicator_var.fix(1)
                        m.expensive_reactor.indicator_var.fix(0)
                    else:
                        m.cheap_reactor.indicator_var.fix(0)
                        m.expensive_reactor.indicator_var.fix(1)

                    if recycle_compressor_choice == 'single_stage':
                        m.single_stage_recycle_compressor_disjunct.indicator_var.fix(1)
                        m.two_stage_recycle_compressor_disjunct.indicator_var.fix(0)
                    else:
                        m.single_stage_recycle_compressor_disjunct.indicator_var.fix(0)
                        m.two_stage_recycle_compressor_disjunct.indicator_var.fix(1)

                    pe.TransformationFactory('gdp.fix_disjuncts').apply_to(m)

                    fbbt(m, deactivate_satisfied_constraints=True)
                    fix_vars_with_equal_bounds(m)

                    for v in m.component_data_objects(pe.Var, descend_into=True):
                        if not v.is_fixed():
                            if v.has_lb() and v.has_ub():
                                v.value = 0.5 * (v.lb + v.ub)
                            else:
                                v.value = 1.0

                    opt = pe.SolverFactory('ipopt')
                    res = opt.solve(m, tee=False)
                    print(
                        '{0:<20}{1:<20}{2:<20}{3:<20}{4:<20}{5:<20}'.format(
                            feed_choice,
                            feed_compressor_choice,
                            reactor_choice,
                            recycle_compressor_choice,
                            str(res.solver.termination_condition),
                            str(-pe.value(m.objective)),
                        )
                    )
    time_elapsed = time.time() - since
    print('The code run {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return m


def solve_with_gdp_opt():
    m = MethanolModel().model
    for _d in m.component_data_objects(
        gdp.Disjunct, descend_into=True, active=True, sort=True
    ):
        _d.BigM = pe.Suffix()
        for _c in _d.component_data_objects(
            pe.Constraint, descend_into=True, active=True, sort=True
        ):
            lb, ub = compute_bounds_on_expr(_c.body)
            _d.BigM[_c] = max(abs(lb), abs(ub))
    opt = pe.SolverFactory('gdpopt')
    res = opt.solve(m, algorithm='LOA', mip_solver='gams', nlp_solver='gams', tee=True)
    for d in m.component_data_objects(
        ctype=gdp.Disjunct, active=True, sort=True, descend_into=True
    ):
        if d.indicator_var.value == 1:
            print(d.name)
    print(res)

    return m


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, filename='out.log')
    from pyomo.util.model_size import *

    # m = enumerate()
    m = solve_with_gdp_opt()
    print(build_model_size_report(m))
