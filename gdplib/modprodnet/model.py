from __future__ import division

import os

import pandas as pd
from pyomo.environ import (
    ConcreteModel, Constraint, Integers, maximize, Objective, Param, RangeSet, SolverFactory, summation,
    TransformationFactory, value, Var, )
from pyomo.gdp import Disjunct, Disjunction


def build_model(case="Growth"):
    m = ConcreteModel()
    m.months = RangeSet(0, 120, doc="10 years")
    m.discount_rate = Param(initialize=0.08, doc="8%")
    m.conv_setup_time = Param(initialize=12)
    m.modular_setup_time = Param(initialize=3)
    m.modular_teardown_time = Param(initialize=3)
    m.conventional_salvage_value = Param(initialize=0.05, doc="5%")
    m.module_salvage_value = Param(initialize=0.3, doc="30%")

    @m.Param(m.months, doc="Discount factor")
    def discount_factor(m, mo):
        return (1 + m.discount_rate / 12) ** (-mo / 12)

    @m.Param(m.months, doc="Value of each unit of production")
    def production_value(m, mo):
        return (7 / 12) * m.discount_factor[mo]

    xls_data = pd.read_excel(
        os.path.join(os.path.dirname(__file__), "market_size.xlsx"), sheet_name="single_market", index_col=0)

    @m.Param(m.months)
    def market_demand(m, mo):
        return float(xls_data[case][mo])

    m.conv_size = Var(bounds=(25, 350), initialize=25,
                      doc="Size of conventional plant")
    m.conv_base_cost = Param(initialize=1000, doc="Cost for size 25")
    m.conv_exponent = Param(initialize=0.6)
    m.conv_cost = Var(bounds=(0, value(m.conv_base_cost * (m.conv_size.ub / 25) ** m.conv_exponent)))
    m.module_base_cost = Param(initialize=1000, doc="Cost for size 25")

    m.production = Var(m.months, bounds=(0, 350))
    m.num_modules = Var(m.months, domain=Integers, bounds=(0, 35))
    m.modules_purchased = Var(m.months, domain=Integers, bounds=(0, 35))
    m.modules_sold = Var(m.months, domain=Integers, bounds=(0, 35))

    @m.Param(m.months)
    def module_unit_cost(m, mo):
        return m.module_base_cost * m.discount_factor[mo]

    @m.Constraint(m.months)
    def demand_limit(m, mo):
        return m.production[mo] <= m.market_demand[mo]

    @m.Expression(m.months)
    def revenue(m, mo):
        return m.production_value[mo] * m.production[mo]

    @m.Expression()
    def conv_salvage_val(m):
        return m.conv_cost * m.conventional_salvage_value * m.discount_factor[120]

    @m.Expression(m.months)
    def module_buy_cost(m, mo):
        return m.module_base_cost * m.discount_factor[mo] * m.modules_purchased[mo]

    @m.Expression(m.months)
    def module_sell_value(m, mo):
        return m.module_base_cost * m.discount_factor[mo] * m.module_salvage_value * m.modules_sold[mo]

    @m.Expression()
    def module_final_salvage(m):
        mo = max(m.months)
        return m.module_base_cost * m.discount_factor[mo] * m.module_salvage_value * m.num_modules[mo]

    m.profit = Objective(
        expr=sum(m.revenue[:])
        - m.conv_cost
        + m.conv_salvage_val
        - summation(m.module_buy_cost)
        + summation(m.module_sell_value)
        + m.module_final_salvage,
        sense=maximize)

    _build_conventional_disjunct(m)
    _build_modular_disjunct(m)
    m.conventional_or_modular = Disjunction(expr=[m.conventional, m.modular])

    return m


def _build_conventional_disjunct(m):
    m.conventional = Disjunct()

    m.conventional.cost_calc = Constraint(
        expr=m.conv_cost == (
            m.conv_base_cost * (m.conv_size / 25) ** m.conv_exponent))

    @m.conventional.Constraint(m.months)
    def conv_production_limit(conv_disj, mo):
        if mo < m.conv_setup_time:
            return m.production[mo] == 0
        else:
            return m.production[mo] <= m.conv_size

    @m.conventional.Constraint()
    def no_modules(conv_disj):
        return sum(m.num_modules[:]) + sum(m.modules_purchased[:]) + sum(m.modules_sold[:]) == 0


def _build_modular_disjunct(m):
    m.modular = Disjunct()

    @m.modular.Constraint(m.months)
    def module_balance(disj, mo):
        existing_modules = 0 if mo == 0 else m.num_modules[mo - 1]
        new_modules = 0 if mo < m.modular_setup_time else m.modules_purchased[mo - m.modular_setup_time]
        sold_modules = m.modules_sold[mo]
        return m.num_modules[mo] == existing_modules + new_modules - sold_modules

    @m.modular.Constraint(m.months)
    def modular_production_limit(mod_disj, mo):
        return m.production[mo] <= 25 * m.num_modules[mo]


def display_conventional(m, writer, sheet_name):
    df = pd.DataFrame(
        list([
            mo,
            m.production[mo].value,
            m.market_demand[mo],
            m.conv_size.value if mo >= m.conv_setup_time else 0
        ] for mo in m.months),
        columns=("Month", "Production", "Demand", "Capacity")
    ).set_index('Month').round(2)
    df.to_excel(writer, sheet_name)
    print('Conventional Profit', round(value(m.profit)))
    print('Conventional Revenue', round(value(sum(m.revenue[:]))))
    print('Conventional Size', round(value(m.conv_size)))
    print('Conventional Build Cost', round(value(m.conv_cost)))
    print('Conventional Salvage Value', round(value(m.conv_salvage_val)))
    print()


def display_modular(m, writer, sheet_name):
    df = pd.DataFrame(
        list([
            mo,
            m.production[mo].value,
            m.market_demand[mo],
            m.num_modules[mo].value * 25,
            m.num_modules[mo].value,
            m.modules_purchased[mo].value,
            m.modules_sold[mo].value] for mo in m.months
        ),
        columns=("Month", "Production", "Demand", "Capacity",
                 "Num Modules", "Add Modules", "Sold Modules")
    ).set_index('Month').round(2)
    df.to_excel(writer, sheet_name)
    print('Modular Profit', round(value(m.profit)))
    print('Modular Revenue', round(value(sum(m.revenue[:]))))
    print('Modular Revenue before conventional startup', round(value(sum(m.revenue[mo] for mo in m.months if mo < 12))))
    print('Modular Build Cost', round(value(sum(m.module_buy_cost[:]))))
    print('Modules Purchased', round(value(sum(m.modules_purchased[:]))))
    print('Modular Nondiscount Cost', round(value(m.module_base_cost * sum(m.modules_purchased[:]))))
    print('Modular Sale Credit', round(value(sum(m.module_sell_value[:]))))
    print('Modular Final Salvage Credit', round(value(m.module_final_salvage)))
    print()


if __name__ == "__main__":
    cases = ['Growth', 'Dip', 'Decay']
    conv_size_vals = {}
    with pd.ExcelWriter('cap_expand_config.xlsx') as writer:
        for case in cases:
            print(case)
            m = build_model(case)
            m.conventional.indicator_var.fix(1)
            m.modular.deactivate()
            TransformationFactory('gdp.bigm').apply_to(m, bigM=7000)
            SolverFactory('gams').solve(m, solver='baron')
            conv_size_vals[case] = value(m.conv_size)
            display_conventional(m, writer, 'conv_%s' % case)
            del m

            m = build_model(case)
            m.modular.indicator_var.fix(1)
            m.conventional.deactivate()
            TransformationFactory('gdp.chull').apply_to(m)
            SolverFactory('gurobi').solve(m)
            display_modular(m, writer, 'mod_%s' % case)

    # Run conventional case with fixed values
