from __future__ import division

import os

import pandas as pd

from pyomo.environ import (
    ConcreteModel, Constraint, Integers, NonNegativeReals, Objective, Param,
    RangeSet, Set, SolverFactory, Suffix, TransformationFactory, Var, exp, log,
    sqrt, summation, value
)
from .util import alphanum_sorted
from pyomo.environ import TerminationCondition as tc


def build_model():
    m = ConcreteModel()
    m.BigM = Suffix(direction=Suffix.LOCAL)

    m.periods_per_year = Param(initialize=4, doc="Quarters per year")
    m.project_life = Param(initialize=15, doc="Years")
    m.time = RangeSet(0, m.periods_per_year *
                      m.project_life - 1, doc="Time periods")
    m.discount_rate = Param(initialize=0.08, doc="8%")
    m.learning_rate = Param(initialize=0.1, doc="Fraction discount for doubling of quantity")

    m.module_setup_time = Param(
        initialize=1, doc="1 quarter for module transfer")

    @m.Param(m.time)
    def discount_factor(m, t):
        return (1 + m.discount_rate / m.periods_per_year) ** (-t / m.periods_per_year)

    xlsx_data = pd.read_excel(os.path.join(os.path.dirname(__file__), "data.xlsx"), sheet_name=None)
    module_sheet = xlsx_data['modules'].set_index('Type')
    m.module_types = Set(initialize=module_sheet.columns.tolist(),)

    @m.Param(m.module_types)
    def module_base_cost(m, mtype):
        return float(module_sheet[mtype]['Capital Cost [MM$]'])

    @m.Param(m.module_types, doc="Natural gas consumption per module of this type [MMSCF/d]")
    def unit_gas_consumption(m, mtype):
        return float(module_sheet[mtype]['Nat Gas [MMSCF/d]'])

    @m.Param(m.module_types, doc="Gasoline production per module of this type [kBD]")
    def gasoline_production(m, mtype):
        return float(module_sheet[mtype]['Gasoline [kBD]'])

    @m.Param(m.module_types, doc="Overall conversion of natural gas into gasoline per module of this type [kB/MMSCF]")
    def module_conversion(m, mtype):
        return float(module_sheet[mtype]['Conversion [kB/MMSCF]'])

    site_sheet = xlsx_data['sites'].set_index('Potential site')
    m.potential_sites = Set(initialize=site_sheet.index.tolist())
    m.site_pairs = Set(
        doc="Pairs of potential sites",
        initialize=m.potential_sites * m.potential_sites,
        filter=lambda _, x, y: not x == y)

    @m.Param(m.potential_sites)
    def site_x(m, site):
        return float(site_sheet['x'][site])

    @m.Param(m.potential_sites)
    def site_y(m, site):
        return float(site_sheet['y'][site])

    well_sheet = xlsx_data['wells'].set_index('Well')
    m.well_clusters = Set(initialize=well_sheet.index.tolist())

    @m.Param(m.well_clusters)
    def well_x(m, well):
        return float(well_sheet['x'][well])

    @m.Param(m.well_clusters)
    def well_y(m, well):
        return float(well_sheet['y'][well])

    sched_sheet = xlsx_data['well-schedule']
    decay_curve = [1] + [3.69 * exp(-1.31 * (t + 1) ** 0.292) for t in range(m.project_life * 12)]
    well_profiles = {well: [0 for _ in decay_curve] for well in m.well_clusters}
    for _, well_info in sched_sheet.iterrows():
        start_time = int(well_info['Month'])
        prod = [0] * start_time + decay_curve[:len(decay_curve) - start_time]
        prod = [x * float(well_info['max prod [MMSCF/d]']) for x in prod]
        current_profile = well_profiles[well_info['well-cluster']]
        well_profiles[well_info['well-cluster']] = [val + prod[i] for i, val in enumerate(current_profile)]

    @m.Param(m.well_clusters, m.time, doc="Supply of gas from well cluster [MMSCF/day]")
    def gas_supply(m, well, t):
        return sum(well_profiles[well][t * 3:t * 3 + 2]) / 3

    mkt_sheet = xlsx_data['markets'].set_index('Market')
    m.markets = Set(initialize=mkt_sheet.index.tolist())

    @m.Param(m.markets)
    def mkt_x(m, mkt):
        return float(mkt_sheet['x'][mkt])

    @m.Param(m.markets)
    def mkt_y(m, mkt):
        return float(mkt_sheet['y'][mkt])

    @m.Param(m.markets, doc="Gasoline demand [kBD]")
    def mkt_demand(m, mkt):
        return float(mkt_sheet['demand [kBD]'][mkt])

    m.sources = Set(initialize=m.well_clusters | m.potential_sites)
    m.destinations = Set(initialize=m.potential_sites | m.markets)

    @m.Param(m.sources, m.destinations, doc="Distance [mi]")
    def distance(m, src, dest):
        if src in m.well_clusters:
            src_x = m.well_x[src]
            src_y = m.well_y[src]
        else:
            src_x = m.site_x[src]
            src_y = m.site_y[src]
        if dest in m.markets:
            dest_x = m.mkt_x[dest]
            dest_y = m.mkt_y[dest]
        else:
            dest_x = m.site_x[dest]
            dest_y = m.site_y[dest]
        return sqrt((src_x - dest_x) ** 2 + (src_y - dest_y) ** 2)

    m.num_modules = Var(
        m.module_types, m.potential_sites, m.time,
        doc="Number of active modules of each type at a site in a period",
        domain=Integers, bounds=(0, 50), initialize=1)
    m.modules_transferred = Var(
        m.module_types, m.site_pairs, m.time,
        doc="Number of module transfers initiated from one site to another in a period.",
        domain=Integers, bounds=(0, 15), initialize=0)
    m.modules_purchased = Var(
        m.module_types, m.potential_sites, m.time,
        doc="Number of modules of each type purchased for a site in a period",
        domain=Integers, bounds=(0, 30), initialize=1)

    m.pipeline_unit_cost = Param(doc="MM$/mile", initialize=2)

    @m.Param(m.time, doc="Module transport cost per mile [M$/100 miles]")
    def module_transport_distance_cost(m, t):
        return 50 * m.discount_factor[t]

    @m.Param(m.time, doc="Module transport cost per unit [MM$/module]")
    def module_transport_unit_cost(m, t):
        return 3 * m.discount_factor[t]

    @m.Param(m.time, doc="Stranded gas price [$/MSCF]")
    def nat_gas_price(m, t):
        return 5 * m.discount_factor[t]

    @m.Param(m.time, doc="Gasoline price [$/gal]")
    def gasoline_price(m, t):
        return 2.5 * m.discount_factor[t]

    @m.Param(m.time, doc="Gasoline transport cost [$/gal/100 miles]")
    def gasoline_tranport_cost(m, t):
        return 0.045 * m.discount_factor[t]

    m.gal_per_bbl = Param(initialize=42, doc="Gallons per barrel")
    m.days_per_period = Param(initialize=90, doc="Days in a production period")

    m.learning_factor = Var(
        m.module_types,
        doc="Fraction of cost due to economies of mass production",
        domain=NonNegativeReals, bounds=(0, 1), initialize=1)

    @m.Disjunct(m.module_types)
    def mtype_exists(disj, mtype):
        disj.learning_factor_calc = Constraint(
            expr=m.learning_factor[mtype] == (1 - m.learning_rate) ** (
                log(sum(m.modules_purchased[mtype, :, :])) / log(2)))
        m.BigM[disj.learning_factor_calc] = 1
        disj.require_module_purchases = Constraint(
            expr=sum(m.modules_purchased[mtype, :, :]) >= 1)

    @m.Disjunct(m.module_types)
    def mtype_absent(disj, mtype):
        disj.constant_learning_factor = Constraint(
            expr=m.learning_factor[mtype] == 1)

    @m.Disjunction(m.module_types)
    def mtype_existence(m, mtype):
        return [m.mtype_exists[mtype], m.mtype_absent[mtype]]

    @m.Expression(m.module_types, m.time, doc="Module unit cost [MM$/module]")
    def module_unit_cost(m, mtype, t):
        return m.module_base_cost[mtype] * m.learning_factor[mtype] * m.discount_factor[t]

    m.production = Var(
        m.potential_sites, m.time,
        doc="Production of gasoline in a time period [kBD]",
        domain=NonNegativeReals, bounds=(0, 30), initialize=10)
    m.gas_consumption = Var(
        m.potential_sites, m.module_types, m.time,
        doc="Consumption of natural gas by each module type "
        "at each site in a time period [MMSCF/d]",
        domain=NonNegativeReals, bounds=(0, 250), initialize=50)
    m.gas_flows = Var(
        m.well_clusters, m.potential_sites, m.time,
        doc="Flow of gas from a well cluster to a site [MMSCF/d]",
        domain=NonNegativeReals, bounds=(0, 200), initialize=15)
    m.product_flows = Var(
        m.potential_sites, m.markets, m.time,
        doc="Product shipments from a site to a market in a period [kBD]",
        domain=NonNegativeReals, bounds=(0, 30), initialize=10)

    @m.Constraint(m.potential_sites, m.module_types, m.time)
    def consumption_capacity(m, site, mtype, t):
        return m.gas_consumption[site, mtype, t] <= (
            m.num_modules[mtype, site, t] * m.unit_gas_consumption[mtype])

    @m.Constraint(m.potential_sites, m.time)
    def production_limit(m, site, t):
        return m.production[site, t] <= sum(
            m.gas_consumption[site, mtype, t] * m.module_conversion[mtype]
            for mtype in m.module_types)

    @m.Expression(m.potential_sites, m.time)
    def capacity(m, site, t):
        return sum(
            m.num_modules[mtype, site, t] * m.unit_gas_consumption[mtype]
            * m.module_conversion[mtype] for mtype in m.module_types)

    @m.Constraint(m.potential_sites, m.time)
    def gas_supply_meets_consumption(m, site, t):
        return sum(m.gas_consumption[site, :, t]) == sum(m.gas_flows[:, site, t])

    @m.Constraint(m.well_clusters, m.time)
    def gas_supply_limit(m, well, t):
        return sum(m.gas_flows[well, site, t]
                   for site in m.potential_sites) <= m.gas_supply[well, t]

    @m.Constraint(m.potential_sites, m.time)
    def gasoline_production_requirement(m, site, t):
        return sum(m.product_flows[site, mkt, t]
                   for mkt in m.markets) == m.production[site, t]

    @m.Constraint(m.potential_sites, m.module_types, m.time)
    def module_balance(m, site, mtype, t):
        if t >= m.module_setup_time:
            modules_added = m.modules_purchased[
                mtype, site, t - m.module_setup_time]
            modules_transferred_in = sum(
                m.modules_transferred[
                    mtype, from_site, to_site, t - m.module_setup_time]
                for from_site, to_site in m.site_pairs if to_site == site)
        else:
            modules_added = 0
            modules_transferred_in = 0
        if t >= 1:
            existing_modules = m.num_modules[mtype, site, t - 1]
        else:
            existing_modules = 0
        modules_transferred_out = sum(
            m.modules_transferred[mtype, from_site, to_site, t]
            for from_site, to_site in m.site_pairs if from_site == site)

        return m.num_modules[mtype, site, t] == (
            existing_modules + modules_added
            + modules_transferred_in - modules_transferred_out)

    @m.Disjunct(m.potential_sites)
    def site_active(disj, site):
        pass

    @m.Disjunct(m.potential_sites)
    def site_inactive(disj, site):
        disj.no_production = Constraint(
            expr=sum(m.production[site, :]) == 0)
        disj.no_gas_consumption = Constraint(
            expr=sum(m.gas_consumption[site, :, :]) == 0)
        disj.no_gas_flows = Constraint(
            expr=sum(m.gas_flows[:, site, :]) == 0)
        disj.no_product_flows = Constraint(
            expr=sum(m.product_flows[site, :, :]) == 0)
        disj.no_modules = Constraint(
            expr=sum(m.num_modules[:, site, :]) == 0)
        disj.no_modules_transferred = Constraint(
            expr=sum(
                m.modules_transferred[mtypes, from_site, to_site, t]
                for mtypes in m.module_types
                for from_site, to_site in m.site_pairs
                for t in m.time
                if from_site == site or to_site == site) == 0)
        disj.no_modules_purchased = Constraint(
            expr=sum(
                m.modules_purchased[mtype, site, t]
                for mtype in m.module_types for t in m.time) == 0)

    @m.Disjunction(m.potential_sites)
    def site_active_or_not(m, site):
        return [m.site_active[site], m.site_inactive[site]]

    @m.Disjunct(m.well_clusters, m.potential_sites)
    def pipeline_exists(disj, well, site):
        pass

    @m.Disjunct(m.well_clusters, m.potential_sites)
    def pipeline_absent(disj, well, site):
        disj.no_natural_gas_flow = Constraint(
            expr=sum(m.gas_flows[well, site, t] for t in m.time) == 0)

    @m.Disjunction(m.well_clusters, m.potential_sites)
    def pipeline_existence(m, well, site):
        return [m.pipeline_exists[well, site], m.pipeline_absent[well, site]]

    # Objective Function Construnction
    @m.Expression(m.potential_sites, doc="MM$")
    def product_revenue(m, site):
        return sum(
            m.product_flows[site, mkt, t]  # kBD
            * 1000  # bbl/kB
            / 1E6  # $ to MM$
            * m.days_per_period
            * m.gasoline_price[t] * m.gal_per_bbl
            for mkt in m.markets
            for t in m.time)

    @m.Expression(m.potential_sites, doc="MM$")
    def raw_material_cost(m, site):
        return sum(
            m.gas_consumption[site, mtype, t] * m.days_per_period
            / 1E6  # $ to MM$
            * m.nat_gas_price[t]
            * 1000  # MMSCF to MSCF
            for mtype in m.module_types for t in m.time)

    @m.Expression(
        m.potential_sites, m.markets,
        doc="Aggregate cost to transport gasoline from a site to market [MM$]")
    def product_transport_cost(m, site, mkt):
        return sum(
            m.product_flows[site, mkt, t] * m.gal_per_bbl
            * 1000  # bbl/kB
            / 1E6  # $ to MM$
            * m.distance[site, mkt] / 100 * m.gasoline_tranport_cost[t]
            for t in m.time)

    @m.Expression(m.well_clusters, m.potential_sites, doc="MM$")
    def pipeline_construction_cost(m, well, site):
        return (m.pipeline_unit_cost * m.distance[well, site]
                * m.pipeline_exists[well, site].indicator_var)

    # Module transport cost
    @m.Expression(m.site_pairs, doc="MM$")
    def module_relocation_cost(m, from_site, to_site):
        return sum(
            m.modules_transferred[mtype, from_site, to_site, t]
            * m.distance[from_site, to_site] / 100
            * m.module_transport_distance_cost[t]
            / 1E3  # M$ to MM$
            + m.modules_transferred[mtype, from_site, to_site, t]
            * m.module_transport_unit_cost[t]
            for mtype in m.module_types
            for t in m.time)

    @m.Expression(m.potential_sites, doc="MM$")
    def module_purchase_cost(m, site):
        return sum(
            m.module_unit_cost[mtype, t] * m.modules_purchased[mtype, site, t]
            for mtype in m.module_types
            for t in m.time)

    @m.Expression(doc="MM$")
    def profit(m):
        return (
            summation(m.product_revenue)
            - summation(m.raw_material_cost)
            - summation(m.product_transport_cost)
            - summation(m.pipeline_construction_cost)
            - summation(m.module_relocation_cost)
            - summation(m.module_purchase_cost)
        )

    m.neg_profit = Objective(expr=-m.profit)

    # Tightening constraints
    @m.Constraint(doc="Limit total module purchases over project span.")
    def restrict_module_purchases(m):
        return sum(m.modules_purchased[...]) <= 5

    @m.Constraint(m.site_pairs, doc="Limit transfers between any two sites")
    def restrict_module_transfers(m, from_site, to_site):
        return sum(m.modules_transferred[:, from_site, to_site, :]) <= 5

    return m


if __name__ == "__main__":
    m = build_model()

    # Restrict number of module types; A, R, S, U
    # valid_modules = ['A500', 'A1000', 'A2000', 'A5000']
    # valid_modules = ['A500', 'R500', 'A5000', 'R5000']
    # valid_modules = ['U500', 'U5000']
    # valid_modules = ['U100', 'U250']
    # valid_modules = ['U1000']
    # valid_modules = ['U500']
    valid_modules = ['U250']
    # valid_modules = ['U100']
    for mtype in m.module_types - valid_modules:
        m.gas_consumption[:, mtype, :].fix(0)
        m.num_modules[mtype, :, :].fix(0)
        m.modules_transferred[mtype, :, :, :].fix(0)
        m.modules_purchased[mtype, :, :].fix(0)
        m.mtype_exists[mtype].deactivate()
        m.mtype_absent[mtype].indicator_var.fix(1)
