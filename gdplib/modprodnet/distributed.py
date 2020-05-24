from __future__ import division

import os

import matplotlib.pyplot as plt
import pandas as pd

from pyomo.environ import *


def build_model():
    m = ConcreteModel()
    m.months = RangeSet(0, 120, doc="10 years")
    m.discount_rate = Param(initialize=0.08, doc="8%")
    m.conv_setup_time = Param(initialize=12)
    m.modular_setup_time = Param(initialize=3)
    m.modular_move_time = Param(initialize=3)
    m.markets = RangeSet(5)

    xls_data = pd.read_excel(
        os.path.join(os.path.dirname(__file__), "multiple_market_size.xlsx"),
        sheet_name=["demand", "locations"], index_col=0)

    @m.Param(m.markets, m.months)
    def market_demand(m, mkt, mo):
        return float(xls_data["demand"]["market%s" % mkt][mo])

    @m.Param(m.months, doc="Cost of transporting unit of product one mile.")
    def transport_cost(m, mo):
        return 0.2 / 12 * (1 + m.discount_rate / 12) ** (-mo / 12)

    m.route_fixed_cost = Param(
        initialize=100,
        doc="Cost of establishing a route from a modular site to a market")

    m.conv_x = Var(bounds=(0, 300), doc="x-coordinate of centralized plant.")
    m.conv_y = Var(bounds=(0, 300), doc="y-coordinate of centralized plant.")

    m.conv_size = Var(bounds=(10, 500), initialize=10,
                      doc="Size of conventional plant.")
    m.conv_cost = Var()
    m.conv_base_cost = Param(initialize=1000, doc="Cost for size 20")
    m.conv_exponent = Param(initialize=0.6)

    m.cost_calc = Constraint(
        expr=m.conv_cost == (
            m.conv_base_cost * (m.conv_size / 20) ** m.conv_exponent))

    @m.Param(m.markets)
    def mkt_x(m, mkt):
        return float(xls_data["locations"]["x"]["market%s" % mkt])

    @m.Param(m.markets)
    def mkt_y(m, mkt):
        return float(xls_data["locations"]["y"]["market%s" % mkt])

    m.dist_to_mkt = Var(m.markets, bounds=(0, sqrt(300**2 + 300**2)))

    @m.Constraint(m.markets)
    def distance_calculation(m, mkt):
        return m.dist_to_mkt[mkt] == sqrt(
            (m.conv_x / 300 - m.mkt_x[mkt] / 300)**2 +
            (m.conv_y / 300 - m.mkt_y[mkt] / 300)**2) * 300

    m.shipments_to_mkt = Var(m.markets, m.months, bounds=(0, 100))
    m.production = Var(m.months, bounds=(0, 500))

    @m.Constraint(m.months)
    def production_satisfaction(m, mo):
        return m.production[mo] == sum(m.shipments_to_mkt[mkt, mo]
                                       for mkt in m.markets)

    @m.Constraint(m.months)
    def size_requirement(m, mo):
        if mo < m.conv_setup_time:
            return m.production[mo] == 0
        else:
            return m.production[mo] <= m.conv_size

    @m.Constraint(m.markets, m.months)
    def demand_satisfaction(m, mkt, mo):
        return m.market_demand[mkt, mo] <= m.shipments_to_mkt[mkt, mo]

    m.variable_shipment_cost = Expression(
        expr=sum(
            m.shipments_to_mkt[mkt, mo] * m.dist_to_mkt[mkt] *
            m.transport_cost[mo]
            for mkt in m.markets for mo in m.months))

    m.total_cost = Objective(
        expr=m.variable_shipment_cost + m.conv_cost + 5 * m.route_fixed_cost)

    return m


def build_modular_model():
    m = ConcreteModel()
    m.bigM = Suffix(direction=Suffix.LOCAL)
    m.months = RangeSet(0, 120, doc="10 years")
    m.discount_rate = Param(initialize=0.08, doc="8%")
    m.modular_setup_time = Param(initialize=3)
    m.modular_move_time = Param(initialize=3)
    m.markets = RangeSet(5)
    m.modular_sites = RangeSet(2)
    m.site_pairs = Set(
        initialize=m.modular_sites * m.modular_sites,
        filter=lambda _, x, y: not x == y)
    m.unique_site_pairs = Set(
        initialize=m.site_pairs, filter=lambda _, x, y: x < y)

    xls_data = pd.read_excel(
        os.path.join(os.path.dirname(__file__), "multiple_market_size.xlsx"),
        sheet_name=["demand", "locations"], index_col=0)

    @m.Param(m.markets, m.months)
    def market_demand(m, mkt, mo):
        return float(xls_data["demand"]["market%s" % mkt][mo])

    @m.Param(m.months, doc="Cost of transporting unit of product one mile.")
    def transport_cost(m, mo):
        return 0.2 / 12 * (1 + m.discount_rate / 12) ** (-mo / 12)

    m.route_fixed_cost = Param(
        initialize=100,
        doc="Cost of establishing a route from a modular site to a market")

    @m.Param(m.months, doc="Cost of transporting a module one mile")
    def modular_transport_cost(m, mo):
        return 1 * (1 + m.discount_rate / 12) ** (-mo / 12)
    m.modular_base_cost = Param(initialize=1000, doc="Cost for size 20")

    @m.Param(m.months)
    def modular_unit_cost(m, mo):
        return m.modular_base_cost * (1 + m.discount_rate / 12) ** (-mo / 12)

    @m.Param(m.markets)
    def mkt_x(m, mkt):
        return float(xls_data["locations"]["x"]["market%s" % mkt])

    @m.Param(m.markets)
    def mkt_y(m, mkt):
        return float(xls_data["locations"]["y"]["market%s" % mkt])

    m.site_x = Var(
        m.modular_sites, bounds=(0, 300), initialize={
            1: 50, 2: 225, 3: 250})
    m.site_y = Var(
        m.modular_sites, bounds=(0, 300), initialize={
            1: 300, 2: 275, 3: 50})

    m.num_modules = Var(
        m.modular_sites, m.months, domain=Integers, bounds=(0, 25))
    m.modules_transferred = Var(
        m.site_pairs, m.months,
        domain=Integers, bounds=(0, 25),
        doc="Number of modules moved from one site to another in a month.")
    # m.modules_transferred[...].fix(0)
    m.modules_added = Var(
        m.modular_sites, m.months, domain=Integers, bounds=(0, 25))

    m.dist_to_mkt = Var(
        m.modular_sites, m.markets, bounds=(0, sqrt(300**2 + 300**2)))
    m.sqr_scaled_dist_to_mkt = Var(
        m.modular_sites, m.markets, bounds=(0.001, 8), initialize=0.001)
    m.dist_to_site = Var(m.site_pairs, bounds=(0, sqrt(300**2 + 300**2)))
    m.sqr_scaled_dist_to_site = Var(
        m.site_pairs, bounds=(0.001, 8), initialize=0.001)
    m.shipments_to_mkt = Var(
        m.modular_sites, m.markets, m.months, bounds=(0, 100))
    m.production = Var(m.modular_sites, m.months, bounds=(0, 500))

    @m.Disjunct(m.modular_sites)
    def site_active(disj, site):
        @disj.Constraint(m.months)
        def production_satisfaction(site_disj, mo):
            return m.production[site, mo] == sum(
                m.shipments_to_mkt[site, mkt, mo] for mkt in m.markets)

        @disj.Constraint(m.months)
        def production_limit(site_disj, mo):
            return m.production[site, mo] <= m.num_modules[site, mo] * 20

        @disj.Constraint(m.months)
        def module_balance(site_disj, mo):
            existing_modules = m.num_modules[site, mo - 1] if mo >= 1 else 0
            new_modules = (m.modules_added[site, mo - m.modular_setup_time]
                           if mo > m.modular_setup_time else 0)
            xfrd_in_modules = sum(
                m.modules_transferred[from_site,
                                      site, mo - m.modular_move_time]
                for from_site in m.modular_sites
                if (not from_site == site) and mo > m.modular_move_time)
            xfrd_out_modules = sum(
                m.modules_transferred[site, to_site, mo]
                for to_site in m.modular_sites if not to_site == site)
            return m.num_modules[site, mo] == (
                new_modules + xfrd_in_modules - xfrd_out_modules +
                existing_modules)

    @m.Disjunct(m.modular_sites)
    def site_inactive(disj, site):
        disj.no_modules = Constraint(
            expr=sum(m.num_modules[site, mo] for mo in m.months) == 0)

        disj.no_module_transfer = Constraint(
            expr=sum(m.modules_transferred[site1, site2, mo]
                     for site1, site2 in m.site_pairs
                     for mo in m.months
                     if site1 == site or site2 == site) == 0)

        disj.no_shipments = Constraint(
            expr=sum(m.shipments_to_mkt[site, mkt, mo]
                     for mkt in m.markets for mo in m.months) == 0)

    @m.Disjunction(m.modular_sites)
    def site_active_or_not(m, site):
        return [m.site_active[site], m.site_inactive[site]]

    @m.Constraint(m.modular_sites, doc="Symmetry breaking for site activation")
    def site_active_ordering(m, site):
        if site + 1 <= max(m.modular_sites):
            return (m.site_active[site].indicator_var >=
                    m.site_active[site + 1].indicator_var)
        else:
            return Constraint.NoConstraint

    @m.Disjunct(m.unique_site_pairs)
    def pair_active(disj, site1, site2):
        disj.site1_active = Constraint(
            expr=m.site_active[site1].indicator_var == 1)
        disj.site2_active = Constraint(
            expr=m.site_active[site2].indicator_var == 1)
        disj.site_distance_calc = Constraint(
            expr=m.dist_to_site[site1, site2] == sqrt(
                m.sqr_scaled_dist_to_site[site1, site2]) * 150)
        disj.site_distance_symmetry = Constraint(
            expr=m.dist_to_site[site1, site2] == m.dist_to_site[site2, site1])
        disj.site_sqr_distance_calc = Constraint(
            expr=m.sqr_scaled_dist_to_site[site1, site2] == (
                (m.site_x[site1] / 150 - m.site_x[site2] / 150)**2 +
                (m.site_y[site1] / 150 - m.site_y[site2] / 150)**2))
        disj.site_sqr_distance_symmetry = Constraint(
            expr=m.sqr_scaled_dist_to_site[site1, site2] ==
            m.sqr_scaled_dist_to_site[site2, site1])

    @m.Disjunct(m.unique_site_pairs)
    def pair_inactive(disj, site1, site2):
        disj.site1_inactive = Constraint(
            expr=m.site_active[site1].indicator_var == 0)
        disj.site2_inactive = Constraint(
            expr=m.site_active[site2].indicator_var == 0)

        disj.no_module_transfer = Constraint(
            expr=sum(m.modules_transferred[site1, site2, mo]
                     for mo in m.months) == 0)

    @m.Disjunction(m.unique_site_pairs)
    def site_pair_active_or_not(m, site1, site2):
        return [m.pair_active[site1, site2], m.pair_inactive[site1, site2]]

    @m.Constraint(m.markets, m.months)
    def demand_satisfaction(m, mkt, mo):
        return m.market_demand[mkt, mo] <= sum(
            m.shipments_to_mkt[site, mkt, mo] for site in m.modular_sites)

    @m.Disjunct(m.modular_sites, m.markets)
    def product_route_active(disj, site, mkt):
        disj.site_active = Constraint(
            expr=m.site_active[site].indicator_var == 1)

        @disj.Constraint()
        def market_distance_calculation(site_disj):
            return m.dist_to_mkt[site, mkt] == sqrt(
                m.sqr_scaled_dist_to_mkt[site, mkt]) * 150

        @disj.Constraint()
        def market_sqr_distance_calc(site_disj):
            return m.sqr_scaled_dist_to_mkt[site, mkt] == (
                (m.site_x[site] / 150 - m.mkt_x[mkt] / 150)**2 +
                (m.site_y[site] / 150 - m.mkt_y[mkt] / 150)**2)

    @m.Disjunct(m.modular_sites, m.markets)
    def product_route_inactive(disj, site, mkt):
        disj.no_shipments = Constraint(
            expr=sum(m.shipments_to_mkt[site, mkt, mo]
                     for mo in m.months) == 0)

    @m.Disjunction(m.modular_sites, m.markets)
    def product_route_active_or_not(m, site, mkt):
        return [m.product_route_active[site, mkt],
                m.product_route_inactive[site, mkt]]

    m.variable_shipment_cost = Expression(
        expr=sum(
            m.shipments_to_mkt[site, mkt, mo] * m.dist_to_mkt[site, mkt] *
            m.transport_cost[mo]
            for site in m.modular_sites for mkt in m.markets
            for mo in m.months))

    m.fixed_shipment_cost = Expression(
        expr=sum(m.product_route_active[site, mkt].indicator_var *
                 m.route_fixed_cost
                 for site in m.modular_sites for mkt in m.markets))

    m.module_purchase_cost = Expression(
        expr=sum(m.modules_added[site, mo] * m.modular_unit_cost[mo]
                 for site in m.modular_sites for mo in m.months))

    m.module_transfer_cost = Expression(expr=sum(
        m.modules_transferred[site1, site2, mo]
        * m.dist_to_site[site1, site2] * m.modular_transport_cost[mo]
        for site1, site2 in m.site_pairs for mo in m.months))

    m.total_cost = Objective(
        expr=m.variable_shipment_cost
        + m.fixed_shipment_cost
        + m.module_purchase_cost
        + m.module_transfer_cost)

    return m


if __name__ == "__main__":
    # m = build_model()
    # SolverFactory('ipopt').solve(m)
    # print("Plant at ({conv_x:3.0f}, {conv_y:3.0f}) size {conv_size:3.0f} "
    #       "cost {conv_cost:4.0f}.".format(
    #           conv_x=m.conv_x.value,
    #           conv_y=m.conv_y.value,
    #           conv_size=m.conv_size.value,
    #           conv_cost=m.conv_cost.value))
    # print("Total cost: %s" % value(m.total_cost.expr))
    # print("  Variable ship cost: %s" % value(m.variable_shipment_cost))
    # print("  Fix ship cost: %s" % value(500))
    # print("  Conv buy cost: %s" % value(m.conv_cost))
    # exit()

    m = build_modular_model()
    # SolverFactory('gdpopt').solve(
    #     m, tee=True, mip_solver="gams", nlp_solver="ipopt",
    #     mip_solver_args={
    #                      'io_options': {'add_options': [
    #                          'option reslim=30;']}})
    TransformationFactory('gdp.bigm').apply_to(m, bigM=10000)
    # TransformationFactory('gdp.chull').apply_to(m)
    res = SolverFactory('gams').solve(m, tee=True, io_options={
        'add_options': ['option reslim = 300;']})
    # SolverFactory('gams').solve(
    #     m, tee=True,
    #     io_options={
    #         'solver': 'baron',
    #         'add_options': ['option solver=baron;',
    #                         'option reslim = 600;']})
    # from pyomo.util.infeasible import log_infeasible_constraints
    # log_infeasible_constraints(m)

    def record_generator():
        for mo in m.months:
            yield (
                (mo,)
                + tuple(m.num_modules[site, mo].value
                        for site in m.modular_sites)
                + tuple(m.production[site, mo].value
                        for site in m.modular_sites)
                + tuple(m.shipments_to_mkt[site, mkt, mo].value
                        for site in m.modular_sites
                        for mkt in m.markets)
            )

    df = pd.DataFrame.from_records(
        record_generator(),
        columns=("Month",)
        + tuple("Num Site%s" % site for site in m.modular_sites)
        + tuple("Prod Site%s" % site for site in m.modular_sites)
        + tuple("Ship Site%s to Mkt%s" % (site, mkt)
                for site in m.modular_sites for mkt in m.markets))
    df.to_excel("multiple_modular_config.xlsx")
    print("Total cost: %s" % value(m.total_cost.expr))
    print("  Variable ship cost: %s" % value(m.variable_shipment_cost))
    print("  Fix ship cost: %s" % value(m.fixed_shipment_cost))
    print("  Module buy cost: %s" % value(m.module_purchase_cost))
    print("  Module xfer cost: %s" % value(m.module_transfer_cost))
    for site in m.modular_sites:
        print("Site {:1.0f} at ({:3.0f}, {:3.0f})".format(
            site, m.site_x[site].value, m.site_y[site].value))
        print("  Supplies markets {}".format(tuple(
            mkt for mkt in m.markets
            if m.product_route_active[site, mkt].indicator_var.value == 1)))


    if res.solver.termination_condition is not TerminationCondition.optimal:
        exit()

    plt.plot([x.value for x in m.site_x.values()],
             [y.value for y in m.site_y.values()], 'k.', markersize=12)
    plt.plot([x for x in m.mkt_x.values()],
             [y for y in m.mkt_y.values()], 'bo', markersize=12)
    for mkt in m.markets:
        plt.annotate('mkt%s' % mkt, (m.mkt_x[mkt], m.mkt_y[mkt]),
                     (m.mkt_x[mkt] + 2, m.mkt_y[mkt] + 0))
    for site in m.modular_sites:
        plt.annotate(
            'site%s' % site, (m.site_x[site].value, m.site_y[site].value),
             (m.site_x[site].value + 2, m.site_y[site].value + 0))
    for site, mkt in m.modular_sites * m.markets:
        if m.product_route_active[site, mkt].indicator_var.value == 1:
            plt.arrow(m.site_x[site].value, m.site_y[site].value,
                      m.mkt_x[mkt] - m.site_x[site].value,
                      m.mkt_y[mkt] - m.site_y[site].value,
                      width=0.8, length_includes_head=True, color='r')
    for site1, site2 in m.site_pairs:
        if sum(m.modules_transferred[site1, site2, mo].value
               for mo in m.months) > 0:
            plt.arrow(m.site_x[site1].value, m.site_y[site1].value,
                      m.site_x[site2].value - m.site_x[site1].value,
                      m.site_y[site2].value - m.site_y[site1].value,
                      width=0.9, length_includes_head=True,
                      linestyle='dotted', color='k')
    plt.show()
