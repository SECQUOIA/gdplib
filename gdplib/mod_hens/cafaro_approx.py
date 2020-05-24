"""Cafaro approximation parameter estimation.

Rather than use the cost relation (1), Cafaro & Grossmann, 2014 (DOI:
10.1016/j.compchemeng.2013.10.001) proposes using (2), which has much better
behaved derivative values near x=0. However, we need to use parameter
estimation in order to derive the correct values of k and b.

This file performs the parameter estimation.

(1) cost = factor * (area) ^ 0.6
(2) cost = factor * k * ln(bx + 1)

"""
from __future__ import division

from pyomo.environ import (ConcreteModel, Constraint, log, NonNegativeReals, SolverFactory, value, Var)


def calculate_cafaro_coefficients(area1, area2, exponent):
    """Calculate the coefficients for the Cafaro approximation.

    Gives the coefficients k and b to approximate a function x^exponent such
    that at the given areas, the following relations apply:

    area1 ^ exponent = k * ln(b * area1 + 1)
    area2 ^ exponent = k * ln(b * area2 + 1)

    Args:
        area1 (float): area to use as the first regression point
        area2 (float): area to use as the second regression point
        exponent (float): exponent to approximate
    """
    m = ConcreteModel()
    m.k = Var(domain=NonNegativeReals)
    m.b = Var(domain=NonNegativeReals)

    m.c1 = Constraint(
        expr=area1 ** exponent == m.k * log(m.b * area1 + 1))
    m.c2 = Constraint(
        expr=area2 ** exponent == m.k * log(m.b * area2 + 1))

    SolverFactory('ipopt').solve(m)

    return value(m.k), value(m.b)
