"""Heat integration case study.

This is example 1 of the Yee & Grossmann, 1990 paper "Simultaneous optimization
models for heat integration--II".
DOI: 10.1016/0098-1354(90)85010-8

This file provides common modeling elements.

"""
from __future__ import division

from pyomo.environ import (
    ConcreteModel, Constraint, minimize, NonNegativeReals, Objective, Param, RangeSet, Set, Suffix, value, Var, )
from pyomo.gdp import Disjunct, Disjunction

from .cafaro_approx import calculate_cafaro_coefficients


def build_model(use_cafaro_approximation, num_stages):
    """Build the model."""
    m = ConcreteModel()
    m.hot_process_streams = Set(initialize=['H1', 'H2'])
    m.cold_process_streams = Set(initialize=['C1', 'C2'])
    m.process_streams = m.hot_process_streams | m.cold_process_streams
    m.hot_utility_streams = Set(initialize=['steam'])
    m.cold_utility_streams = Set(initialize=['water'])
    m.hot_streams = Set(
        initialize=m.hot_process_streams | m.hot_utility_streams)
    m.cold_streams = Set(
        initialize=m.cold_process_streams | m.cold_utility_streams)
    m.utility_streams = Set(
        initialize=m.hot_utility_streams | m.cold_utility_streams)
    m.streams = Set(
        initialize=m.process_streams | m.utility_streams)
    m.valid_matches = Set(
        initialize=(m.hot_process_streams * m.cold_streams) |
        (m.hot_utility_streams * m.cold_process_streams),
        doc="Match all hot streams to cold streams, but exclude "
        "matches between hot and cold utilities.")
    # m.EMAT = Param(doc="Exchanger minimum approach temperature [K]",
    #                initialize=1)
    # Unused right now, but could be used for variable bound tightening
    # in the LMTD calculation.

    m.stages = RangeSet(num_stages)

    m.T_in = Param(
        m.streams, doc="Inlet temperature of stream [K]",
        initialize={'H1': 443,
                    'H2': 423,
                    'C1': 293,
                    'C2': 353,
                    'steam': 450,
                    'water': 293})
    m.T_out = Param(
        m.streams, doc="Outlet temperature of stream [K]",
        initialize={'H1': 333,
                    'H2': 303,
                    'C1': 408,
                    'C2': 413,
                    'steam': 450,
                    'water': 313})

    m.heat_exchanged = Var(
        m.valid_matches, m.stages,
        domain=NonNegativeReals,
        doc="Heat exchanged from hot stream to cold stream in stage [kW]",
        initialize=1, bounds=(0, 5000))

    m.overall_FCp = Param(
        m.process_streams,
        doc="Flow times heat capacity of stream [kW / K]",
        initialize={'H1': 30,
                    'H2': 15,
                    'C1': 20,
                    'C2': 40})
    m.utility_usage = Var(
        m.utility_streams,
        doc="Hot or cold utility used [kW]",
        domain=NonNegativeReals, initialize=1, bounds=(0, 5000))
    m.stage_entry_T = Var(
        m.streams, m.stages,
        doc="Temperature of stream at stage entry.",
        initialize=350,
        bounds=(293, 450)  # TODO set to be equal to min and max temps
    )
    m.stage_exit_T = Var(
        m.streams, m.stages,
        doc="Temperature of stream at stage exit.",
        initialize=350,
        bounds=(293, 450)  # TODO set to be equal to min and max temps
    )
    # Improve bounds on stage entry and exit temperatures
    for strm, stg in m.process_streams * m.stages:
        m.stage_entry_T[strm, stg].setlb(
            min(value(m.T_in[strm]), value(m.T_out[strm])))
        m.stage_exit_T[strm, stg].setlb(
            min(value(m.T_in[strm]), value(m.T_out[strm])))
        m.stage_entry_T[strm, stg].setub(
            max(value(m.T_in[strm]), value(m.T_out[strm])))
        m.stage_exit_T[strm, stg].setub(
            max(value(m.T_in[strm]), value(m.T_out[strm])))
    for strm, stg in m.utility_streams * m.stages:
        _fix_and_bound(m.stage_entry_T[strm, stg], m.T_in[strm])
        _fix_and_bound(m.stage_exit_T[strm, stg], m.T_out[strm])
    for strm in m.hot_process_streams:
        _fix_and_bound(m.stage_entry_T[strm, 1], m.T_in[strm])
        _fix_and_bound(m.stage_exit_T[strm, num_stages], m.T_out[strm])
    for strm in m.cold_process_streams:
        _fix_and_bound(m.stage_exit_T[strm, 1], m.T_out[strm])
        _fix_and_bound(m.stage_entry_T[strm, num_stages], m.T_in[strm])

    m.BigM = Suffix(direction=Suffix.LOCAL)

    m.utility_unit_cost = Param(
        m.utility_streams,
        doc="Annual unit cost of utilities [$/kW]",
        initialize={'steam': 80, 'water': 20})

    m.module_sizes = Set(initialize=[10, 50, 100])
    m.max_num_modules = Param(m.module_sizes, initialize={
        # 5: 100,
        10: 50,
        50: 10,
        100: 5,
        # 250: 2
    }, doc="maximum number of each module size available.")

    m.exchanger_fixed_unit_cost = Param(
        m.valid_matches, default=2000)
    m.exchanger_area_cost_factor = Param(
        m.valid_matches, default=1000,
        initialize={
            ('steam', cold): 1200
            for cold in m.cold_process_streams},
        doc="1200 for heaters. 1000 for all other exchangers.")
    m.area_cost_exponent = Param(default=0.6)

    if use_cafaro_approximation:
        k, b = calculate_cafaro_coefficients(10, 500, m.area_cost_exponent)
        m.cafaro_k = Param(default=k)
        m.cafaro_b = Param(default=b)

    @m.Param(m.valid_matches, m.module_sizes,
             doc="Area cost factor for modular exchangers.")
    def module_area_cost_factor(m, hot, cold, area):
        if hot == 'steam':
            return 1300
        else:
            return 1100

    m.module_fixed_unit_cost = Param(default=0)
    m.module_area_cost_exponent = Param(default=0.6)

    @m.Param(m.valid_matches, m.module_sizes,
             doc="Cost of a module with a particular area.")
    def module_area_cost(m, hot, cold, area):
        return (m.module_area_cost_factor[hot, cold, area]
                * area ** m.module_area_cost_exponent)

    m.U = Param(
        m.valid_matches,
        default=0.8,
        initialize={
            ('steam', cold): 1.2
            for cold in m.cold_process_streams},
        doc="Overall heat transfer coefficient."
        "1.2 for heaters. 0.8 for everything else.")

    m.exchanger_hot_side_approach_T = Var(
        m.valid_matches, m.stages,
        doc="Temperature difference between the hot stream inlet and cold "
        "stream outlet of the exchanger.",
        bounds=(0.1, 500), initialize=10
    )
    m.exchanger_cold_side_approach_T = Var(
        m.valid_matches, m.stages,
        doc="Temperature difference between the hot stream outlet and cold "
        "stream inlet of the exchanger.",
        bounds=(0.1, 500), initialize=10
    )
    m.LMTD = Var(
        m.valid_matches, m.stages,
        doc="Log mean temperature difference across the exchanger.",
        bounds=(1, 500), initialize=10
    )
    # Improve LMTD bounds based on T values
    for hot, cold, stg in m.valid_matches * m.stages:
        hot_side_dT_LB = max(0, value(
            m.stage_entry_T[hot, stg].lb - m.stage_exit_T[cold, stg].ub))
        hot_side_dT_UB = max(0, value(
            m.stage_entry_T[hot, stg].ub - m.stage_exit_T[cold, stg].lb))
        cold_side_dT_LB = max(0, value(
            m.stage_exit_T[hot, stg].lb - m.stage_entry_T[cold, stg].ub))
        cold_side_dT_UB = max(0, value(
            m.stage_exit_T[hot, stg].ub - m.stage_entry_T[cold, stg].lb))
        m.LMTD[hot, cold, stg].setlb((
            hot_side_dT_LB * cold_side_dT_LB * (
                hot_side_dT_LB + cold_side_dT_LB) / 2) ** (1 / 3)
        )
        m.LMTD[hot, cold, stg].setub((
            hot_side_dT_UB * cold_side_dT_UB * (
                hot_side_dT_UB + cold_side_dT_UB) / 2) ** (1 / 3)
        )

    m.exchanger_fixed_cost = Var(
        m.stages, m.valid_matches,
        doc="Fixed cost for an exchanger between a hot and cold stream.",
        domain=NonNegativeReals, bounds=(0, 1E5), initialize=0)

    m.exchanger_area = Var(
        m.stages, m.valid_matches,
        doc="Area for an exchanger between a hot and cold stream.",
        domain=NonNegativeReals, bounds=(0, 500), initialize=5)
    m.exchanger_area_cost = Var(
        m.stages, m.valid_matches,
        doc="Capital cost contribution from exchanger area.",
        domain=NonNegativeReals, bounds=(0, 1E5), initialize=1000)

    @m.Constraint(m.hot_process_streams)
    def overall_hot_stream_heat_balance(m, strm):
        return (m.T_in[strm] - m.T_out[strm]) * m.overall_FCp[strm] == (
            sum(m.heat_exchanged[strm, cold, stg]
                for cold in m.cold_streams for stg in m.stages))

    @m.Constraint(m.cold_process_streams)
    def overall_cold_stream_heat_balance(m, strm):
        return (m.T_out[strm] - m.T_in[strm]) * m.overall_FCp[strm] == (
            sum(m.heat_exchanged[hot, strm, stg]
                for hot in m.hot_streams for stg in m.stages))

    @m.Constraint(m.utility_streams)
    def overall_utility_stream_usage(m, strm):
        return m.utility_usage[strm] == (
            sum(m.heat_exchanged[hot, strm, stg]
                for hot in m.hot_process_streams
                for stg in m.stages
                ) if strm in m.cold_utility_streams else 0 +
            sum(m.heat_exchanged[strm, cold, stg]
                for cold in m.cold_process_streams
                for stg in m.stages
                ) if strm in m.hot_utility_streams else 0
        )

    @m.Constraint(m.stages, m.hot_process_streams,
                  doc="Hot side overall heat balance for a stage.")
    def hot_stage_overall_heat_balance(m, stg, strm):
        return ((m.stage_entry_T[strm, stg] - m.stage_exit_T[strm, stg])
                * m.overall_FCp[strm]) == sum(
            m.heat_exchanged[strm, cold, stg]
            for cold in m.cold_streams)

    @m.Constraint(m.stages, m.cold_process_streams,
                  doc="Cold side overall heat balance for a stage.")
    def cold_stage_overall_heat_balance(m, stg, strm):
        return ((m.stage_exit_T[strm, stg] - m.stage_entry_T[strm, stg])
                * m.overall_FCp[strm]) == sum(
            m.heat_exchanged[hot, strm, stg]
            for hot in m.hot_streams)

    @m.Constraint(m.stages, m.hot_process_streams)
    def hot_stream_monotonic_T_decrease(m, stg, strm):
        return m.stage_exit_T[strm, stg] <= m.stage_entry_T[strm, stg]

    @m.Constraint(m.stages, m.cold_process_streams)
    def cold_stream_monotonic_T_increase(m, stg, strm):
        return m.stage_exit_T[strm, stg] >= m.stage_entry_T[strm, stg]

    @m.Constraint(m.stages, m.hot_process_streams)
    def hot_stream_stage_T_link(m, stg, strm):
        return (
            m.stage_exit_T[strm, stg] == m.stage_entry_T[strm, stg + 1]
        ) if stg < num_stages else Constraint.NoConstraint

    @m.Constraint(m.stages, m.cold_process_streams)
    def cold_stream_stage_T_link(m, stg, strm):
        return (
            m.stage_entry_T[strm, stg] == m.stage_exit_T[strm, stg + 1]
        ) if stg < num_stages else Constraint.NoConstraint

    @m.Expression(m.valid_matches, m.stages)
    def exchanger_capacity(m, hot, cold, stg):
        return m.exchanger_area[stg, hot, cold] * (
            m.U[hot, cold] * (
                m.exchanger_hot_side_approach_T[hot, cold, stg] *
                m.exchanger_cold_side_approach_T[hot, cold, stg] *
                (m.exchanger_hot_side_approach_T[hot, cold, stg] +
                 m.exchanger_cold_side_approach_T[hot, cold, stg]) / 2
            ) ** (1 / 3))

    def _exchanger_exists(disj, hot, cold, stg):
        disj.indicator_var.value = 1

        # Log mean temperature difference calculation
        disj.LMTD_calc = Constraint(
            doc="Log mean temperature difference",
            expr=m.LMTD[hot, cold, stg] == (
                m.exchanger_hot_side_approach_T[hot, cold, stg] *
                m.exchanger_cold_side_approach_T[hot, cold, stg] *
                (m.exchanger_hot_side_approach_T[hot, cold, stg] +
                 m.exchanger_cold_side_approach_T[hot, cold, stg]) / 2
            ) ** (1 / 3)
        )
        m.BigM[disj.LMTD_calc] = 160

        # disj.MTD_calc = Constraint(
        #     doc="Mean temperature difference",
        #     expr=m.LMTD[hot, cold, stg] <= (
        #         m.exchanger_hot_side_approach_T[hot, cold, stg] +
        #         m.exchanger_cold_side_approach_T[hot, cold, stg]) / 2
        # )

        # Calculation of the approach temperatures
        if hot in m.hot_utility_streams:
            disj.stage_hot_approach_temperature = Constraint(
                expr=m.exchanger_hot_side_approach_T[hot, cold, stg] <=
                m.T_in[hot] - m.stage_exit_T[cold, stg])
            disj.stage_cold_approach_temperature = Constraint(
                expr=m.exchanger_cold_side_approach_T[hot, cold, stg] <=
                m.T_out[hot] - m.stage_entry_T[cold, stg])
        elif cold in m.cold_utility_streams:
            disj.stage_hot_approach_temperature = Constraint(
                expr=m.exchanger_hot_side_approach_T[hot, cold, stg] <=
                m.stage_entry_T[hot, stg] - m.T_out[cold])
            disj.stage_cold_approach_temperature = Constraint(
                expr=m.exchanger_cold_side_approach_T[hot, cold, stg] <=
                m.stage_exit_T[hot, stg] - m.T_in[cold])
        else:
            disj.stage_hot_approach_temperature = Constraint(
                expr=m.exchanger_hot_side_approach_T[hot, cold, stg] <=
                m.stage_entry_T[hot, stg]
                - m.stage_exit_T[cold, stg])
            disj.stage_cold_approach_temperature = Constraint(
                expr=m.exchanger_cold_side_approach_T[hot, cold, stg] <=
                m.stage_exit_T[hot, stg]
                - m.stage_entry_T[cold, stg])

    def _exchanger_absent(disj, hot, cold, stg):
        disj.indicator_var.value = 0
        disj.no_match_exchanger_cost = Constraint(
            expr=m.exchanger_area_cost[stg, hot, cold] == 0)
        disj.no_match_exchanger_area = Constraint(
            expr=m.exchanger_area[stg, hot, cold] == 0)
        disj.no_match_exchanger_fixed_cost = Constraint(
            expr=m.exchanger_fixed_cost[stg, hot, cold] == 0)
        disj.no_heat_exchange = Constraint(
            expr=m.heat_exchanged[hot, cold, stg] == 0)

    m.exchanger_exists = Disjunct(
        m.valid_matches, m.stages,
        doc="Disjunct for the presence of an exchanger between a "
        "hot stream and a cold stream at a stage.", rule=_exchanger_exists)
    m.exchanger_absent = Disjunct(
        m.valid_matches, m.stages,
        doc="Disjunct for the absence of an exchanger between a "
        "hot stream and a cold stream at a stage.", rule=_exchanger_absent)

    def _exchanger_exists_or_absent(m, hot, cold, stg):
        return [m.exchanger_exists[hot, cold, stg],
                m.exchanger_absent[hot, cold, stg]]
    m.exchanger_exists_or_absent = Disjunction(
        m.valid_matches, m.stages,
        doc="Disjunction between presence or absence of an exchanger between "
        "a hot stream and a cold stream at a stage.",
        rule=_exchanger_exists_or_absent, xor=True)
    # Only hot utility matches in first stage and cold utility matches in last
    # stage
    for hot, cold in m.valid_matches:
        if hot not in m.utility_streams:
            m.exchanger_exists[hot, cold, 1].deactivate()
            m.exchanger_absent[hot, cold, 1].indicator_var.fix(1)
        if cold not in m.utility_streams:
            m.exchanger_exists[hot, cold, num_stages].deactivate()
            m.exchanger_absent[hot, cold, num_stages].indicator_var.fix(1)
    # Exclude utility-stream matches in middle stages
    for hot, cold, stg in m.valid_matches * (m.stages - [1, num_stages]):
        if hot in m.utility_streams or cold in m.utility_streams:
            m.exchanger_exists[hot, cold, stg].deactivate()
            m.exchanger_absent[hot, cold, stg].indicator_var.fix(1)

    @m.Expression(m.utility_streams)
    def utility_cost(m, strm):
        return m.utility_unit_cost[strm] * m.utility_usage[strm]

    m.total_cost = Objective(
        expr=sum(m.utility_cost[strm] for strm in m.utility_streams)
        + sum(m.exchanger_fixed_cost[stg, hot, cold]
              for stg in m.stages
              for hot, cold in m.valid_matches)
        + sum(m.exchanger_area_cost[stg, hot, cold]
              for stg in m.stages
              for hot, cold in m.valid_matches),
        sense=minimize
    )

    return m


def _fix_and_bound(var, val):
    var.fix(val)
    var.setlb(val)
    var.setub(val)
