# coding: utf-8

# # [Pyomo.GDP](./index.ipynb) Logical Expression System Demo - Optimal Positioning
#
# This is a reproduction of the optimal positioning problem found as Example 4 in:
#
# > Duran, M. A., & Grossmann, I. E. (1986).
# > An outer-approximation algorithm for a class of mixed-integer nonlinear programs.
# > Mathematical Programming, 36(3), 307. https://doi.org/10.1007/BF02592064
#
# The formulation below was adapted from an implementation in MIPSYN.
#
# This code relies on the logic-v1 branch at https://github.com/qtothec/pyomo/tree/logic-v1
# Optimal: -8.06

from pyomo.environ import *
from pyomo.gdp import *
from pyomo.core.expr.logical_expr import *
from pyomo.core.plugins.transform.logical_to_linear import update_boolean_vars_from_binary

from six import StringIO
import pandas as pd


def build_model():
    m = ConcreteModel()
    m.locations = RangeSet(5)
    m.consumers = RangeSet(25)
    m.products = RangeSet(10)

    fixed_profit_data = {
        1: 1,
        2: 0.2,
        3: 1,
        4: 0.2,
        5: 0.9,
        6: 0.9,
        7: 0.1,
        8: 0.8,
        9: 1,
        10: 0.4,
        11: 1,
        12: 0.3,
        13: 0.1,
        14: 0.3,
        15: 0.5,
        16: 0.9,
        17: 0.8,
        18: 0.1,
        19: 0.9,
        20: 1,
        21: 1,
        22: 1,
        23: 0.2,
        24: 0.7,
        25: 0.7,
    }
    m.fixed_profit = Param(m.consumers, initialize=fixed_profit_data)

    minimum_weights_data = {
        1: 77.84,
        2: 175.971,
        3: 201.823,
        4: 143.953,
        5: 154.39,
        6: 433.318,
        7: 109.076,
        8: 41.596,
        9: 144.062,
        10: 99.834,
        11: 149.179,
        12: 123.807,
        13: 27.222,
        14: 89.927,
        15: 293.077,
        16: 174.317,
        17: 125.103,
        18: 222.842,
        19: 50.486,
        20: 361.197,
        21: 40.326,
        22: 161.852,
        23: 66.858,
        24: 340.581,
        25: 407.52,
    }
    m.minimum_weights = Param(m.consumers, initialize=minimum_weights_data)
    location_bounds = {
        1: (2, 4.5),
        2: (0, 8.0),
        3: (3, 9.0),
        4: (0, 5.0),
        5: (4, 10),
    }

    ideal_points_data = StringIO("""
            1       2       3       4       5
    1       2.26    5.15    4.03    1.74    4.74
    2       5.51    9.01    3.84    1.47    9.92
    3       4.06    1.8     0.71    9.09    8.13
    4       6.3     0.11    4.08    7.29    4.24
    5       2.81    1.65    8.08    3.99    3.51
    6       4.29    9.49    2.24    9.78    1.52
    7       9.76    3.64    6.62    3.66    9.08
    8       1.37    6.99    7.19    3.03    3.39
    9       8.89    8.29    6.05    7.48    4.09
    10      7.42    4.60    0.3     0.97    8.77
    11      1.54    7.06    0.01    1.23    3.11
    12      7.74    4.4     7.93    5.95    4.88
    13      9.94    5.21    8.58    0.13    4.57
    14      9.54    1.57    9.66    5.24    7.9
    15      7.46    8.81    1.67    6.47    1.81
    16      0.56    8.1     0.19    6.11    6.4
    17      3.86    6.68    6.42    7.29    4.66
    18      2.98    2.98    3.03    0.02    0.67
    19      3.61    7.62    1.79    7.8     9.81
    20      5.68    4.24    4.17    6.75    1.08
    21      5.48    3.74    3.34    6.22    7.94
    22      8.13    8.72    3.93    8.8     8.56
    23      1.37    0.54    1.55    5.56    5.85
    24      8.79    5.04    4.83    6.94    0.38
    25      2.66    4.19    6.49    8.04    1.66 
    """)
    ideal_points_table = pd.read_csv(ideal_points_data, delimiter=r'\s+')
    ideal_points_dict = {(k[0], int(k[1])): v for k, v in ideal_points_table.stack().to_dict().items()}
    m.ideal_points = Param(m.consumers, m.locations, initialize=ideal_points_dict)

    weights_data = StringIO("""
            1       2       3       4       5
    1       9.57    2.74    9.75    3.96    8.67
    2       8.38    3.93    5.18    5.2     7.82
    3       9.81    0.04    4.21    7.38    4.11
    4       7.41    6.08    5.46    4.86    1.48
    5       9.96    9.13    2.95    8.25    3.58
    6       9.39    4.27    5.09    1.81    7.58
    7       1.88    7.2     6.65    1.74    2.86
    8       4.01    2.67    4.86    2.55    6.91
    9       4.18    1.92    2.60    7.15    2.86
    10      7.81    2.14    9.63    7.61    9.17
    11      8.96    3.47    5.49    4.73    9.43
    12      9.94    1.63    1.23    4.33    7.08
    13      0.31    5       0.16    2.52    3.08
    14      6.02    0.92    7.47    9.74    1.76
    15      5.06    4.52    1.89    1.22    9.05
    16      5.92    2.56    7.74    6.96    5.18
    17      6.45    1.52    0.06    5.34    8.47
    18      1.04    1.36    5.99    8.10    5.22
    19      1.40    1.35    0.59    8.58    1.21
    20      6.68    9.48    1.6     6.74    8.92
    21      1.95    0.46    2.9     1.79    0.99
    22      5.18    5.1     8.81    3.27    9.63
    23      1.47    5.71    6.95    1.42    3.49
    24      5.4     3.12    5.37    6.1     3.71
    25      6.32    0.81    6.12    6.73    7.93
    """)
    weights_table = pd.read_csv(weights_data, delimiter=r'\s+')
    weights_dict = {(k[0], int(k[1])): v for k, v in weights_table.stack().to_dict().items()}
    m.weights = Param(m.consumers, m.locations, initialize=weights_dict)

    existing_products_data = StringIO("""
            1       2       3       4       5
    1       0.62    5.06    7.82    0.22    4.42
    2       5.21    2.66    9.54    5.03    8.01
    3       5.27    7.72    7.97    3.31    6.56
    4       1.02    8.89    8.77    3.1     6.66
    5       1.26    6.8     2.3     1.75    6.65
    6       3.74    9.06    9.8     3.01    9.52
    7       4.64    7.99    6.69    5.88    8.23
    8       8.35    3.79    1.19    1.96    5.88
    9       6.44    0.17    9.93    6.8     9.75
    10      6.49    1.92    0.05    4.89    6.43
    """)
    existing_products_table = pd.read_csv(existing_products_data, delimiter=r'\s+')
    existing_products_dict = {(k[0], int(k[1])): v for k, v in existing_products_table.stack().to_dict().items()}
    m.existing_products = Param(m.products, m.locations, initialize=existing_products_dict)

    # m.consumers * m.products
    rr = {(i, j): sum(m.weights[i, k] * (m.existing_products[j, k] - m.ideal_points[i, k]) * (
                m.existing_products[j, k] - m.ideal_points[i, k])
                      for k in m.locations)
          for (i, j) in m.consumers * m.products}
    r = {i: min(rr[i, j] for j in m.products) for i in m.consumers}

    m.x = Var(m.locations)
    m.Y = BooleanVar(m.consumers)
    m.H = Param(initialize=1000)
    m.U = Var(bounds=(0, 5000))

    @m.Disjunction(m.consumers)
    def d(m, i):
        return [
            [sum(m.weights[i, k] * (m.x[k] - m.ideal_points[i, k]) ** 2 for k in m.locations) - r[i] <= m.U],
            []
        ]
    for i in m.consumers:
        m.Y[i].set_binary_var(m.d[i].disjuncts[0].indicator_var)
    for k in m.locations:
        lb, ub = location_bounds[k]
        m.x[k].setlb(lb)
        m.x[k].setub(ub)

    m.c1 = Constraint(expr=m.x[1] - m.x[2] + m.x[3] + m.x[4] + m.x[5] <= 10)
    m.c2 = Constraint(expr=0.6 * m.x[1] - 0.9 * m.x[2] - 0.5 * m.x[3] + 0.1 * m.x[4] + m.x[5] <= -0.64)
    m.c3 = Constraint(expr=m.x[1] - m.x[2] + m.x[3] - m.x[4] + m.x[5] >= 0.69)
    m.c4 = Constraint(expr=0.157 * m.x[1] + 0.05 * m.x[2] <= 1.5)
    m.c5 = Constraint(expr=0.25 * m.x[2] + 1.05 * m.x[4] - 0.3 * m.x[5] >= 4.5)

    m.obj = Objective(
        expr=10 * m.U - sum(m.fixed_profit[i] * m.Y[i].as_binary() for i in m.consumers) + 0.6 * m.x[1] ** 2 - 0.9 * m.x[
            2] - 0.5 * m.x[3] + 0.1 * m.x[4] ** 2 + m.x[5])

    return m


if __name__ == "__main__":
    m = build_model()
    TransformationFactory('core.logical_to_linear').apply_to(m)
    # res = SolverFactory('gdpopt').solve(m, tee=True, nlp_solver='gams')
    TransformationFactory('gdp.bigm').apply_to(m)
    SolverFactory('gams').solve(m, tee=True, solver='baron', add_options=['option optcr=0;'])
    update_boolean_vars_from_binary(m)
    m.Y.display()
