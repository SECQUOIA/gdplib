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
    """
    Builds a conventional heat integration model based on specified parameters, delegating to the common build_model function.

    Parameters
    ----------
    cafaro_approx : bool
        Specifies whether to use the Cafaro approximation in the model.
    num_stages : int
        The number of stages in the heat integration model.

    Returns
    -------
    Pyomo.ConcreteModel
        The constructed Pyomo concrete model for heat integration.
    """
    return build_model(cafaro_approx, num_stages)


def build_model(use_cafaro_approximation, num_stages):
    """
    Builds and configures a heat integration model using either standard calculations or the Cafaro approximation for specific costs and heat exchange calculations, supplemented by constraints specific to the conventional scenario.

    Parameters
    ----------
    use_cafaro_approximation : bool
        Flag to determine whether to use the Cafaro approximation for cost calculations.
    num_stages : int
        Number of stages in the heat exchange model.

    Returns
    -------
    Pyomo.ConcreteModel
        A fully configured heat integration model with additional conventional-specific constraints.
    """
    m = common.build_model(use_cafaro_approximation, num_stages)

    for hot, cold, stg in m.valid_matches * m.stages:
        disj = m.exchanger_exists[hot, cold, stg]
        if not use_cafaro_approximation:
            disj.exchanger_area_cost = Constraint(
                expr=m.exchanger_area_cost[stg, hot, cold] * 1e-3
                >= m.exchanger_area_cost_factor[hot, cold]
                * 1e-3
                * m.exchanger_area[stg, hot, cold] ** m.area_cost_exponent,
                doc="Ensures area cost meets the standard cost scaling.",
            )
        else:
            disj.exchanger_area_cost = Constraint(
                expr=m.exchanger_area_cost[stg, hot, cold] * 1e-3
                >= m.exchanger_area_cost_factor[hot, cold]
                * 1e-3
                * m.cafaro_k
                * log(m.cafaro_b * m.exchanger_area[stg, hot, cold] + 1),
                doc="Applies Cafaro's logarithmic cost scaling to area cost.",
            )
        m.BigM[disj.exchanger_area_cost] = 100

        disj.exchanger_fixed_cost = Constraint(
            expr=m.exchanger_fixed_cost[stg, hot, cold]
            == m.exchanger_fixed_unit_cost[hot, cold],
            doc="Sets fixed cost for the exchanger based on unit costs.",
        )

        # Area requirement
        disj.exchanger_required_area = Constraint(
            expr=m.exchanger_area[stg, hot, cold]
            * (m.U[hot, cold] * m.LMTD[hot, cold, stg])
            >= m.heat_exchanged[hot, cold, stg],
            doc="Calculates the required area based on heat exchanged and LMTD.",
        )
        m.BigM[disj.exchanger_required_area] = 5000

    return m
