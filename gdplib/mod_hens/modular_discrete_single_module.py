"""Heat integration case study.

This is example 1 of the Yee & Grossmann, 1990 paper "Simultaneous optimization
models for heat integration--II".
DOI: 10.1016/0098-1354(90)85010-8

This is a modification to support the incorporation of standardized exchanger
modules.

This is a further modification to support discretization of area simplifying
the nonlinear expressions, specialized to the case of allowing only a single
exchanger module type (size).

"""
from __future__ import division

from pyomo.environ import Binary, Constraint, RangeSet, Var
from pyomo.gdp import Disjunction

from . import common


def build_single_module(cafaro_approx, num_stages):
    return build_model(cafaro_approx, num_stages)


def build_model(use_cafaro_approximation, num_stages):
    """Build the model."""
    m = common.build_model(use_cafaro_approximation, num_stages)

    # list of tuples (num_modules, module_size)
    configurations_list = []
    for size in m.module_sizes:
        configs = [(i + 1, size) for i in range(m.max_num_modules[size])]
        configurations_list += configs

    # Map of config indx: (# modules, module size)
    m.configurations_map = {
        (k + 1): v for k, v in enumerate(configurations_list)}

    m.module_index_set = RangeSet(len(configurations_list))

    m.module_config_active = Var(
        m.valid_matches, m.stages, m.module_index_set,
        doc="Binary for if which module configuration is active for a match.",
        domain=Binary, initialize=0)

    @m.Param(m.module_index_set, doc="Area of each configuration")
    def module_area(m, indx):
        num_modules, size = m.configurations_map[indx]
        return num_modules * size

    @m.Param(m.valid_matches, m.module_index_set,
             doc="Area cost for each modular configuration.")
    def modular_size_cost(m, hot, cold, indx):
        num_modules, size = m.configurations_map[indx]
        return num_modules * m.module_area_cost[hot, cold, size]

    @m.Param(m.valid_matches, m.module_index_set,
             doc="Fixed cost for each modular exchanger size.")
    def modular_fixed_cost(m, hot, cold, indx):
        num_modules, size = m.configurations_map[indx]
        return num_modules * m.module_fixed_unit_cost

    m.LMTD_discretize = Var(
        m.hot_streams, m.cold_streams, m.stages, m.module_index_set,
        doc="Discretized log mean temperature difference",
        bounds=(0, 500), initialize=0
    )

    for hot, cold, stg in m.valid_matches * m.stages:
        disj = m.exchanger_exists[hot, cold, stg]
        disj.choose_one_config = Constraint(
            expr=sum(
                m.module_config_active[hot, cold, stg, indx]
                for indx in m.module_index_set) == 1
        )

        disj.exchanger_area_cost = Constraint(
            expr=m.exchanger_area_cost[stg, hot, cold] * 1E-3 ==
            sum(m.modular_size_cost[hot, cold, indx] * 1E-3 *
                m.module_config_active[hot, cold, stg, indx]
                for indx in m.module_index_set)
        )
        disj.exchanger_fixed_cost = Constraint(
            expr=m.exchanger_fixed_cost[stg, hot, cold] ==
            sum(m.modular_fixed_cost[hot, cold, indx] * 1E-3 *
                m.module_config_active[hot, cold, stg, indx]
                for indx in m.module_index_set))

        disj.discretize_area = Constraint(
            expr=m.exchanger_area[stg, hot, cold] == sum(
                m.module_area[indx] *
                m.module_config_active[hot, cold, stg, indx]
                for indx in m.module_index_set)
        )

        disj.discretized_LMTD = Constraint(
            expr=m.LMTD[hot, cold, stg] == sum(
                m.LMTD_discretize[hot, cold, stg, indx]
                for indx in m.module_index_set
            )
        )

        @disj.Constraint(m.module_index_set)
        def discretized_LMTD_LB(disj, indx):
            return (
                m.LMTD[hot, cold, stg].lb
                * m.module_config_active[hot, cold, stg, indx]
            ) <= m.LMTD_discretize[hot, cold, stg, indx]

        @disj.Constraint(m.module_index_set)
        def discretized_LMTD_UB(disj, indx):
            return m.LMTD_discretize[hot, cold, stg, indx] <= (
                m.LMTD[hot, cold, stg].ub
                * m.module_config_active[hot, cold, stg, indx]
            )

        disj.exchanger_required_area = Constraint(
            expr=m.U[hot, cold] * sum(
                m.module_area[indx] * m.LMTD_discretize[hot, cold, stg, indx]
                for indx in m.module_index_set) >=
            m.heat_exchanged[hot, cold, stg])

    @m.Disjunct(m.module_sizes)
    def module_type(disj, size):
        """Disjunct for selection of one module type."""
        @disj.Constraint(m.valid_matches, m.stages, m.module_index_set)
        def no_other_module_types(_, hot, cold, stg, indx):
            # num_modules, size = configurations_map[indx]
            if m.configurations_map[indx][1] != size:
                return m.module_config_active[hot, cold, stg, indx] == 0
            else:
                return Constraint.NoConstraint
        # disj.no_other_module_types = Constraint(
        #     expr=sum(
        #         m.module_config_active[hot, cold, stg, indx]
        #         for indx in m.module_index_set
        #         if m.configurations_map[indx][1] != size) == 0
        # )
    m.select_one_module_type = Disjunction(
        expr=[m.module_type[area] for area in m.module_sizes])

    return m
