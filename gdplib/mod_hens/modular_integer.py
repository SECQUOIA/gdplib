"""Heat integration case study.

This is example 1 of the Yee & Grossmann, 1990 paper "Simultaneous optimization
models for heat integration--II".
DOI: 10.1016/0098-1354(90)85010-8

This is a modification to support the incorporation of standardized exchanger
modules using integer variables for module selection.

"""
from __future__ import division

from pyomo.environ import (Constraint, Integers, log, Var)
from pyomo.gdp import Disjunct, Disjunction

from gdplib.mod_hens import common


def build_single_module(cafaro_approx, num_stages):
    m = build_model(cafaro_approx, num_stages)
    # Require modular
    for hot, cold, stg in m.valid_matches * m.stages:
        disj = m.exchanger_exists[hot, cold, stg]
        disj.modular.indicator_var.fix(1)
        disj.conventional.deactivate()

    # Must choose only one type of module
    @m.Disjunct(m.module_sizes)
    def module_type(disj, size):
        """Disjunct for selection of one module type."""
        disj.no_other_module_types = Constraint(
            expr=sum(m.num_modules[hot, cold, stage, area]
                     for hot, cold in m.valid_matches
                     for stage in m.stages
                     for area in m.module_sizes
                     if area != size) == 0)
    m.select_one_module_type = Disjunction(
        expr=[m.module_type[area] for area in m.module_sizes])

    return m


def build_require_modular(cafaro_approx, num_stages):
    m = build_model(cafaro_approx, num_stages)
    # Require modular
    for hot, cold, stg in m.valid_matches * m.stages:
        m.exchanger_exists[hot, cold, stg].conventional.deactivate()

    return m


def build_modular_option(cafaro_approx, num_stages):
    return build_model(cafaro_approx, num_stages)


def build_model(use_cafaro_approximation, num_stages):
    """Build the model."""
    m = common.build_model(use_cafaro_approximation, num_stages)

    m.num_modules = Var(
        m.valid_matches, m.stages, m.module_sizes,
        doc="The number of modules of each size at each exchanger.",
        domain=Integers, bounds=(0, 100), initialize=0)
    # improve quality of bounds
    for size in m.module_sizes:
        for var in m.num_modules[:, :, :, size]:
            var.setub(m.max_num_modules[size])

    for hot, cold, stg in m.valid_matches * m.stages:
        disj = m.exchanger_exists[hot, cold, stg]

        disj.conventional = Disjunct()
        if not use_cafaro_approximation:
            disj.conventional.exchanger_area_cost = Constraint(
                expr=m.exchanger_area_cost[stg, hot, cold] * 1E-3 >=
                m.exchanger_area_cost_factor[hot, cold] * 1E-3 *
                m.exchanger_area[stg, hot, cold] ** m.area_cost_exponent)
        else:
            disj.conventional.exchanger_area_cost = Constraint(
                expr=m.exchanger_area_cost[stg, hot, cold] * 1E-3 >=
                m.exchanger_area_cost_factor[hot, cold] * 1E-3 * m.cafaro_k
                * log(m.cafaro_b * m.exchanger_area[stg, hot, cold] + 1)
            )
        m.BigM[disj.conventional.exchanger_area_cost] = 100

        disj.conventional.exchanger_fixed_cost = Constraint(
            expr=m.exchanger_fixed_cost[stg, hot, cold] ==
            m.exchanger_fixed_unit_cost[hot, cold])

        @disj.conventional.Constraint(m.module_sizes)
        def no_modules(_, area):
            return m.num_modules[hot, cold, stg, area] == 0

        disj.modular = Disjunct()
        disj.modular.exchanger_area_cost = Constraint(
            expr=m.exchanger_area_cost[stg, hot, cold] * 1E-3 ==
            sum(m.module_area_cost[hot, cold, area]
                * m.num_modules[hot, cold, stg, area]
                for area in m.module_sizes)
            * 1E-3)
        disj.modular.exchanger_fixed_cost = Constraint(
            expr=m.exchanger_fixed_cost[stg, hot, cold] ==
            m.module_fixed_unit_cost * sum(m.num_modules[hot, cold, stg, area]
                                           for area in m.module_sizes))
        disj.modular.exchanger_area = Constraint(
            expr=m.exchanger_area[stg, hot, cold] ==
            sum(area * m.num_modules[hot, cold, stg, area]
                for area in m.module_sizes))
        disj.modular_or_not = Disjunction(
            expr=[disj.modular, disj.conventional])

        # Area requirement
        disj.exchanger_required_area = Constraint(
            expr=m.exchanger_area[stg, hot, cold] * (
                m.U[hot, cold] * m.LMTD[hot, cold, stg]) >=
            m.heat_exchanged[hot, cold, stg])
        m.BigM[disj.exchanger_required_area] = 5000

    return m
