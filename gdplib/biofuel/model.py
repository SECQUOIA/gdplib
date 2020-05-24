from __future__ import division

import os
from math import fabs

import pandas as pd
from pyomo.environ import (
    ConcreteModel, Constraint, Integers, minimize, NonNegativeReals, Objective, Param, RangeSet, SolverFactory, sqrt,
    Suffix, summation, TransformationFactory, value, Var, )
from pyomo.gdp import Disjunct


def build_model():
    m = ConcreteModel()
    m.bigM = Suffix(direction=Suffix.LOCAL)
    m.time = RangeSet(0, 120, doc="months in 10 years")
    m.suppliers = RangeSet(10)
    m.markets = RangeSet(10)
    m.potential_sites = RangeSet(12)
    m.discount_rate = Param(initialize=0.08, doc="8%")
    m.conv_setup_time = Param(initialize=12)
    m.modular_setup_time = Param(initialize=3)
    m.modular_teardown_time = Param(initialize=3)
    m.teardown_value = Param(initialize=0.30, doc="30%")
    m.conventional_salvage_value = Param(initialize=0.05, doc="5%")

    @m.Param(m.time)
    def discount_factor(m, t):
        return (1 + m.discount_rate / 12) ** (-t / 12)

    xls_data = pd.read_excel(
        os.path.join(os.path.dirname(__file__), "problem_data.xlsx"),
        sheet_name=["sources", "markets", "sites", "growth", "decay"],
        index_col=0)

    @m.Param(m.markets, m.time, doc="Market demand [thousand ton/month]")
    def market_demand(m, mkt, t):
        if m.conv_setup_time <= t <= max(m.time) - 3:
            return float(xls_data["markets"]["demand"][mkt]) / 1000 / 12
        else:
            return 0

    @m.Param(m.suppliers, m.time, doc="Raw material supply [thousand ton/month]")
    def available_supply(m, sup, t):
        # If t is before supply available or after supply decayed, then no
        # supply
        if t < float(xls_data["sources"]["growth"][sup]):
            return 0
        elif t > float(xls_data["sources"]["decay"][sup]):
            return 0
        else:
            return float(xls_data["sources"]["avail"][sup]) / 1000 / 12

    @m.Param(m.suppliers)
    def supplier_x(m, sup):
        return float(xls_data["sources"]["x"][sup])

    @m.Param(m.suppliers)
    def supplier_y(m, sup):
        return float(xls_data["sources"]["y"][sup])

    @m.Param(m.markets)
    def market_x(m, mkt):
        return float(xls_data["markets"]["x"][mkt])

    @m.Param(m.markets)
    def market_y(m, mkt):
        return float(xls_data["markets"]["y"][mkt])

    @m.Param(m.potential_sites)
    def site_x(m, site):
        return float(xls_data["sites"]["x"][site])

    @m.Param(m.potential_sites)
    def site_y(m, site):
        return float(xls_data["sites"]["y"][site])

    @m.Param(m.suppliers, m.potential_sites, doc="Miles")
    def dist_supplier_to_site(m, sup, site):
        return sqrt((m.supplier_x[sup] - m.site_x[site]) ** 2 +
                    (m.supplier_y[sup] - m.site_y[site]) ** 2)

    @m.Param(m.potential_sites, m.markets, doc="Miles")
    def dist_site_to_market(m, site, mkt):
        return sqrt((m.site_x[site] - m.market_x[mkt]) ** 2 +
                    (m.site_y[site] - m.market_y[mkt]) ** 2)

    m.conversion = Param(initialize=0.26, doc="overall conversion to product")

    m.conv_site_size = Var(
        m.potential_sites,
        bounds=(120 / 12 / 10, 120 / 12), initialize=1,
        doc="Product capacity of site [thousand ton/mo]")

    m.conv_base_cost = Param(initialize=268.4, doc="Cost for size 120k per year [million $]")
    m.module_base_cost = Param(initialize=268.4, doc="Cost for size 120k per year [million $]")
    m.conv_exponent = Param(initialize=0.7)

    m.supply = Var(m.potential_sites, m.time, bounds=(0, 120 / 12 / 0.26 * 10), doc="thousand ton/mo")
    m.production = Var(m.potential_sites, m.time, bounds=(0, 120 / 12 * 10), doc="thousand ton/mo")
    m.num_modules = Var(m.potential_sites, m.time, domain=Integers, bounds=(0, 10))
    m.modules_purchased = Var(m.potential_sites, m.time, domain=Integers, bounds=(0, 10))
    m.modules_sold = Var(m.potential_sites, m.time, domain=Integers, bounds=(0, 10))

    m.conv_build_cost = Var(
        m.potential_sites,
        doc="Cost of building conventional facility [milllion $]",
        bounds=(0, 1350 * 10), initialize=0)

    @m.Param(m.suppliers, m.time)
    def raw_material_unit_cost(m, sup, t):
        return float(xls_data["sources"]["cost"][sup]) * m.discount_factor[t]

    @m.Param(m.time)
    def module_unit_cost(m, t):
        return m.module_base_cost * m.discount_factor[t]

    @m.Param(m.time, doc="$/ton")
    def unit_production_cost(m, t):
        return 300 * m.discount_factor[t]

    @m.Param(doc="thousand $")
    def transport_fixed_cost(m):
        return 125

    @m.Param(m.time, doc="$/ton-mile")
    def unit_product_transport_cost(m, t):
        return 0.13 * m.discount_factor[t]

    @m.Param(m.time, doc="$/ton-mile")
    def unit_raw_material_transport_cost(m, t):
        return 2 * m.discount_factor[t]

    m.supply_shipments = Var(
        m.suppliers, m.potential_sites, m.time, domain=NonNegativeReals,
        bounds=(0, 120 / 12 / 0.26), doc="thousand ton/mo")
    m.product_shipments = Var(
        m.potential_sites, m.markets, m.time, domain=NonNegativeReals,
        bounds=(0, 120 / 12), doc="thousand ton/mo")

    @m.Constraint(m.suppliers, m.time)
    def supply_limits(m, sup, t):
        return sum(m.supply_shipments[sup, site, t]
                   for site in m.potential_sites) <= m.available_supply[sup, t]

    @m.Constraint(m.markets, m.time)
    def demand_satisfaction(m, mkt, t):
        return sum(m.product_shipments[site, mkt, t]
                   for site in m.potential_sites) == m.market_demand[mkt, t]

    @m.Constraint(m.potential_sites, m.time)
    def product_balance(m, site, t):
        return m.production[site, t] == sum(m.product_shipments[site, mkt, t]
                                            for mkt in m.markets)

    @m.Constraint(m.potential_sites, m.time)
    def require_raw_materials(m, site, t):
        return m.production[site, t] <= m.conversion * m.supply[site, t]

    m.modular = Disjunct(m.potential_sites, rule=_build_modular_disjunct)
    m.conventional = Disjunct(m.potential_sites, rule=_build_conventional_disjunct)
    m.site_inactive = Disjunct(m.potential_sites, rule=_build_site_inactive_disjunct)

    m.supply_route_active = Disjunct(m.suppliers, m.potential_sites, rule=_build_supply_route_active)
    m.supply_route_inactive = Disjunct(m.suppliers, m.potential_sites, rule=_build_supply_route_inactive)

    m.product_route_active = Disjunct(m.potential_sites, m.markets, rule=_build_product_route_active)
    m.product_route_inactive = Disjunct(m.potential_sites, m.markets, rule=_build_product_route_inactive)

    @m.Disjunction(m.potential_sites)
    def site_type(m, site):
        return [m.modular[site], m.conventional[site], m.site_inactive[site]]

    @m.Disjunction(m.suppliers, m.potential_sites)
    def supply_route_active_or_not(m, sup, site):
        return [m.supply_route_active[sup, site], m.supply_route_inactive[sup, site]]

    @m.Disjunction(m.potential_sites, m.markets)
    def product_route_active_or_not(m, site, mkt):
        return [m.product_route_active[site, mkt], m.product_route_inactive[site, mkt]]

    @m.Expression(m.suppliers, m.potential_sites, doc="million $")
    def raw_material_transport_cost(m, sup, site):
        return sum(
            m.supply_shipments[sup, site, t]
            * m.unit_raw_material_transport_cost[t]
            * m.dist_supplier_to_site[sup, site] / 1000
            for t in m.time)

    @m.Expression(doc="million $")
    def raw_material_fixed_transport_cost(m):
        return (
            sum(m.supply_route_active[sup, site].indicator_var
                for sup in m.suppliers for site in m.potential_sites)
            * m.transport_fixed_cost / 1000)

    @m.Expression(m.potential_sites, m.markets, doc="million $")
    def product_transport_cost(m, site, mkt):
        return sum(
            m.product_shipments[site, mkt, t]
            * m.unit_product_transport_cost[t]
            * m.dist_site_to_market[site, mkt] / 1000
            for t in m.time)

    @m.Expression(doc="million $")
    def product_fixed_transport_cost(m):
        return (
            sum(m.product_route_active[site, mkt].indicator_var
                for site in m.potential_sites for mkt in m.markets)
            * m.transport_fixed_cost / 1000)

    @m.Expression(m.potential_sites, m.time, doc="Cost of module setups in each month [million $]")
    def module_setup_cost(m, site, t):
        return m.modules_purchased[site, t] * m.module_unit_cost[t]

    @m.Expression(m.potential_sites, m.time, doc="Value of module teardowns in each month [million $]")
    def module_teardown_credit(m, site, t):
        return m.modules_sold[site, t] * m.module_unit_cost[t] * m.teardown_value

    @m.Expression(m.potential_sites, doc="Conventional site salvage value")
    def conv_salvage_value(m, site):
        return m.conv_build_cost[site] * m.discount_factor[m.time.last()] * m.conventional_salvage_value

    m.total_cost = Objective(
        expr=0
        + summation(m.conv_build_cost)
        + summation(m.module_setup_cost)
        - summation(m.conv_salvage_value)
        - summation(m.module_teardown_credit)
        + summation(m.raw_material_transport_cost)
        + summation(m.raw_material_fixed_transport_cost)
        + summation(m.product_transport_cost)
        + summation(m.product_fixed_transport_cost)
        + 0,
        sense=minimize)

    return m


def _build_site_inactive_disjunct(disj, site):
    m = disj.model()

    @disj.Constraint()
    def no_modules(disj):
        return sum(m.num_modules[...]) + sum(m.modules_purchased[...]) + sum(m.modules_sold[...]) == 0

    @disj.Constraint()
    def no_production(disj):
        return sum(m.production[site, t] for t in m.time) == 0

    @disj.Constraint()
    def no_supply(disj):
        return sum(m.supply[site, t] for t in m.time) == 0


def _build_conventional_disjunct(disj, site):
    m = disj.model()

    disj.cost_calc = Constraint(
        expr=m.conv_build_cost[site] == (
            m.conv_base_cost * (m.conv_site_size[site] / 10) ** m.conv_exponent))
    # m.bigM[disj.cost_calc] = 7000

    @disj.Constraint(m.time)
    def supply_balance(disj, t):
        return m.supply[site, t] == sum(
            m.supply_shipments[sup, site, t] for sup in m.suppliers)

    @disj.Constraint(m.time)
    def conv_production_limit(conv_disj, t):
        if t < m.conv_setup_time:
            return m.production[site, t] == 0
        else:
            return m.production[site, t] <= m.conv_site_size[site]

    @disj.Constraint()
    def no_modules(disj):
        return sum(m.num_modules[...]) + sum(m.modules_purchased[...]) + sum(m.modules_sold[...]) == 0


def _build_modular_disjunct(disj, site):
    m = disj.model()

    @disj.Constraint(m.time)
    def supply_balance(disj, t):
        return m.supply[site, t] == sum(
            m.supply_shipments[sup, site, t] for sup in m.suppliers)

    @disj.Constraint(m.time)
    def module_balance(disj, t):
        existing_modules = 0 if t == m.time.first() else m.num_modules[site, t - 1]
        new_modules = 0 if t < m.modular_setup_time else m.modules_purchased[site, t - m.modular_setup_time]
        sold_modules = m.modules_sold[site, t]
        return m.num_modules[site, t] == existing_modules + new_modules - sold_modules

    for t in range(value(m.modular_setup_time)):
        m.num_modules[site, t].fix(0)

    @disj.Constraint(m.time)
    def modular_production_limit(mod_disj, t):
        return m.production[site, t] <= 10 * m.num_modules[site, t]


def _build_supply_route_active(disj, sup, site):
    m = disj.model()


def _build_supply_route_inactive(disj, sup, site):
    m = disj.model()

    @disj.Constraint()
    def no_supply(disj):
        return sum(m.supply_shipments[sup, site, t] for t in m.time) == 0


def _build_product_route_active(disj, site, mkt):
    m = disj.model()


def _build_product_route_inactive(disj, site, mkt):
    m = disj.model()

    @disj.Constraint()
    def no_product(disj):
        return sum(m.product_shipments[site, mkt, t] for t in m.time) == 0


def print_nonzeros(var):
    for i in var:
        if var[i].value != 0:
            print("%7s : %10f : %10f : %10f" % (i, var[i].lb, var[i].value, var[i].ub))


if __name__ == "__main__":
    m = build_model()
    m.modular[:].deactivate()
    # m.conventional[:].deactivate()
    # for site, t in m.potential_sites * m.time:
    #     if t <= max(m.time) - 4:
    #         m.modular[site].removing_modules[t].deactivate()
    TransformationFactory('gdp.bigm').apply_to(m, bigM=7000)
    # res = SolverFactory('gurobi').solve(m, tee=True)
    res = SolverFactory('gams').solve(
        m, tee=True,
        solver='scip',
        # solver='gurobi',
        # add_options=['option reslim = 1200;', 'option optcr=0.0001;'],
        add_options=[
            'option reslim = 1200;',
            'OPTION threads=4;',
            'option optcr=0.01',
            ],
        )
    # res = SolverFactory('gdpopt').solve(
    #     m, tee=True,
    #     iterlim=2,
    #     mip_solver='gams',
    #     mip_solver_args=dict(add_options=['option reslim = 30;']))
    results = pd.DataFrame([
        ['Total Cost', value(m.total_cost)],
        ['Conv Build Cost', value(summation(m.conv_build_cost))],
        ['Conv Salvage Value', value(summation(m.conv_salvage_value))],
        ['Module Build Cost', value(summation(m.module_setup_cost))],
        ['Module Salvage Value', value(summation(m.module_teardown_credit))],
        ['Raw Material Transport', value(summation(m.raw_material_transport_cost) + summation(m.raw_material_fixed_transport_cost))],
        ['Product Transport', value(summation(m.product_transport_cost) + summation(m.product_fixed_transport_cost))]
    ], columns=['Quantity', 'Value [million $]']).set_index('Quantity').round(0)
    print(results)

    df = pd.DataFrame([
        [
            site, t,
            value(m.num_modules[site, t]),
            value(m.modules_purchased[site, t]),
            value(m.modules_sold[site, t]),
            value(m.module_setup_cost[site, t]),
            value(m.module_teardown_credit[site, t]),
            value(m.production[site, t])] for site, t in m.potential_sites * m.time
        ],
        columns=("Site", "Month", "Num Modules", "Buy Modules",
                 "Sell Modules",
                 "Setup Cost", "Teardown Credit", "Production")
    )
    df.to_excel("facility_config.xlsx")

    # if res.solver.termination_condition is not TerminationCondition.optimal:
    #     exit()
    import matplotlib.pyplot as plt

    plt.plot([x for x in m.site_x.values()],
             [y for y in m.site_y.values()], 'k.', markersize=12)
    plt.plot([x for x in m.market_x.values()],
             [y for y in m.market_y.values()], 'b.', markersize=12)
    plt.plot([x for x in m.supplier_x.values()],
             [y for y in m.supplier_y.values()], 'r.', markersize=12)
    for mkt in m.markets:
        plt.annotate('m%s' % mkt, (m.market_x[mkt], m.market_y[mkt]),
                     (m.market_x[mkt] + 2, m.market_y[mkt] + 2),
                     fontsize='x-small')
    for site in m.potential_sites:
        if m.site_inactive[site].indicator_var.value == 0:
            plt.annotate(
                'p%s' % site, (m.site_x[site], m.site_y[site]),
                (m.site_x[site] + 2, m.site_y[site] + 2),
                fontsize='x-small')
        else:
            plt.annotate(
                'x%s' % site, (m.site_x[site], m.site_y[site]),
                (m.site_x[site] + 2, m.site_y[site] + 2),
                fontsize='x-small')
    for sup in m.suppliers:
        plt.annotate(
            's%s' % sup, (m.supplier_x[sup], m.supplier_y[sup]),
            (m.supplier_x[sup] + 2, m.supplier_y[sup] + 2),
            fontsize='x-small')
    for sup, site in m.suppliers * m.potential_sites:
        if fabs(m.supply_route_active[sup, site].indicator_var.value - 1) <= 1E-3:
            plt.arrow(m.supplier_x[sup], m.supplier_y[sup],
                      m.site_x[site] - m.supplier_x[sup],
                      m.site_y[site] - m.supplier_y[sup],
                      width=0.8, length_includes_head=True, color='b')
    for site, mkt in m.potential_sites * m.markets:
        if fabs(m.product_route_active[site, mkt].indicator_var.value - 1) <= 1E-3:
            plt.arrow(m.site_x[site], m.site_y[site],
                      m.market_x[mkt] - m.site_x[site],
                      m.market_y[mkt] - m.site_y[site],
                      width=0.8, length_includes_head=True, color='r')
    plt.show()
