"""Heat integration case study.

This is example 1 of the Yee & Grossmann, 1990 paper "Simultaneous optimization
models for heat integration--II".
DOI: 10.1016/0098-1354(90)85010-8

This is an implementation of the conventional problem.

"""
from __future__ import division

from pyomo.environ import Constraint, log, value

from . import common


def build_conventional(cafaro_approx, num_stages):
    return build_model(cafaro_approx, num_stages)


def build_model(use_cafaro_approximation, num_stages):
    """Build the model."""
    m = common.build_model(use_cafaro_approximation, num_stages)

    for hot, cold, stg in m.valid_matches * m.stages:
        disj = m.exchanger_exists[hot, cold, stg]
        if not use_cafaro_approximation:
            disj.exchanger_area_cost = Constraint(
                expr=m.exchanger_area_cost[stg, hot, cold] * 1E-3 >=
                m.exchanger_area_cost_factor[hot, cold] * 1E-3 *
                m.exchanger_area[stg, hot, cold] ** m.area_cost_exponent)
        else:
            disj.exchanger_area_cost = Constraint(
                expr=m.exchanger_area_cost[stg, hot, cold] * 1E-3 >=
                m.exchanger_area_cost_factor[hot, cold] * 1E-3 * m.cafaro_k
                * log(m.cafaro_b * m.exchanger_area[stg, hot, cold] + 1)
            )
        m.BigM[disj.exchanger_area_cost] = 100

        disj.exchanger_fixed_cost = Constraint(
            expr=m.exchanger_fixed_cost[stg, hot, cold] ==
            m.exchanger_fixed_unit_cost[hot, cold])

        # Area requirement
        disj.exchanger_required_area = Constraint(
            expr=m.exchanger_area[stg, hot, cold] * (
                m.U[hot, cold] * m.LMTD[hot, cold, stg]) >=
            m.heat_exchanged[hot, cold, stg])
        m.BigM[disj.exchanger_required_area] = 5000

    return m
