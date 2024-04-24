from __future__ import division

import os
from math import fabs

import pandas as pd
from pyomo.environ import (
    ConcreteModel, Constraint, Integers, minimize, NonNegativeReals, Objective, Param, RangeSet, SolverFactory, sqrt,
    Suffix, summation, TransformationFactory, value, Var, )
from pyomo.gdp import Disjunct


def build_model():
    """_summary_

    Returns
    -------
    Pyomo.ConcreteModel
        The Pyomo concrete model which descibes the multiperiod location-allocation optimization model designed to determine the most cost-effective network layout and production allocation to meet market demands.

    References
    ----------
    [1] Lara, C. L., Trespalacios, F., & Grossmann, I. E. (2018). Global optimization algorithm for capacitated multi-facility continuous location-allocation problems. Journal of Global Optimization, 71(4), 871-889. https://doi.org/10.1007/s10898-018-0621-6
    [2] Chen, Q., & Grossmann, I. E. (2019). Effective generalized disjunctive programming models for modular process synthesis. Industrial & Engineering Chemistry Research, 58(15), 5873-5886. https://doi.org/10.1021/acs.iecr.8b04600
    """
    m = ConcreteModel()
    m.bigM = Suffix(direction=Suffix.LOCAL)
    m.time = RangeSet(0, 120, doc="months in 10 years")
    m.suppliers = RangeSet(10) # 10 suppliers
    m.markets = RangeSet(10) # 10 markets
    m.potential_sites = RangeSet(12) # 12 facility sites
    m.discount_rate = Param(initialize=0.08, doc="discount rate [8%]")
    m.conv_setup_time = Param(initialize=12)
    m.modular_setup_time = Param(initialize=3)
    m.modular_teardown_time = Param(initialize=3)
    m.teardown_value = Param(initialize=0.30, doc="tear down value [30%]")
    m.conventional_salvage_value = Param(initialize=0.05, doc="salvage value [5%]")

    @m.Param(m.time)
    def discount_factor(m, t):
        """
        Calculate the discount factor for a given time period 't', based on a monthly compounding interest rate.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        t : int
            Index of time in months from 0 to 120 (10 years)

        Returns
        -------
        Pyomo.Parameter
            The discount factor for month 't', calculated using the formula (1 + r/12)**(-t/12) where 'r' is the annual discount rate.
        """
        return (1 + m.discount_rate / 12) ** (-t / 12)

    xls_data = pd.read_excel(
        os.path.join(os.path.dirname(__file__), "problem_data.xlsx"),
        sheet_name=["sources", "markets", "sites", "growth", "decay"],
        index_col=0)

    @m.Param(m.markets, m.time, doc="Market demand [thousand ton/month]")
    def market_demand(m, mkt, t):
        """
        Calculate the market demand for a given market 'mkt' at time 't', based on the demand data provided in the Excel file.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        mkt : int
            Index of the market from 1 to 10
        t : int
            Index of time in months from 0 to 120 (10 years)

        Returns
        -------
        Pyomo.Parameter
            If the conversion setup time is less than or equal to 't' and 't' is less than the maximum time period minus 3 months, return the market demand in thousand tons per month, otherwise return 0.
        """
        if m.conv_setup_time <= t <= max(m.time) - 3:
            return float(xls_data["markets"]["demand"][mkt]) / 1000 / 12
        else:
            return 0

    @m.Param(m.suppliers, m.time, doc="Raw material supply [thousand ton/month]")
    def available_supply(m, sup, t):
        """
        Calculate the available supply of raw materials for a given supplier 'sup' at time 't', based on the supply data provided in the Excel file.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        sup : int
            Index of the supplier from 1 to 10
        t : int
            Index of time in months from 0 to 120 (10 years)

        Returns
        -------
        Pyomo.Parameter
            If 't' is before the growth period or after the decay period, return 0, otherwise return the available supply in thousand tons per month.
        """
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
        """
        Get the x-coordinate of the supplier location in miles from the Excel data.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        sup : int
            Index of the supplier from 1 to 10

        Returns
        -------
        Pyomo.Parameter
            x-coordinate of the supplier location in miles
        """
        return float(xls_data["sources"]["x"][sup])

    @m.Param(m.suppliers)
    def supplier_y(m, sup):
        """
        Get the y-coordinate of the supplier location in miles from the Excel data.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        sup : int
            Index of the supplier from 1 to 10

        Returns
        -------
        Pyomo.Parameter
            y-coordinate of the supplier location in miles
        """
        return float(xls_data["sources"]["y"][sup])

    @m.Param(m.markets)
    def market_x(m, mkt):
        """
        Get the x-coordinate of the market location in miles from the Excel data.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        mkt : int
            Index of the market from 1 to 10

        Returns
        -------
        Pyomo.Parameter
            x-coordinate of the market location in miles
        """
        return float(xls_data["markets"]["x"][mkt])

    @m.Param(m.markets)
    def market_y(m, mkt):
        """
        Get the y-coordinate of the market location in miles from the Excel data.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        mkt : int
            Index of the market from 1 to 10

        Returns
        -------
        Pyomo.Parameter
            y-coordinate of the market location in miles
        """
        return float(xls_data["markets"]["y"][mkt])

    @m.Param(m.potential_sites)
    def site_x(m, site):
        """
        Get the x-coordinate of the facility site location in miles from the Excel data.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        site : int
            Index of the facility site from 1 to 12

        Returns
        -------
        Pyomo.Parameter
            x-coordinate of the facility site location in miles
        """
        return float(xls_data["sites"]["x"][site])

    @m.Param(m.potential_sites)
    def site_y(m, site):
        """
        Get the y-coordinate of the facility site location in miles from the Excel data.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        site : int
            Index of the facility site from 1 to 12

        Returns
        -------
        Pyomo.Parameter
            y-coordinate of the facility site location in miles
        """
        return float(xls_data["sites"]["y"][site])

    @m.Param(m.suppliers, m.potential_sites, doc="Miles")
    def dist_supplier_to_site(m, sup, site):
        """
        Calculate the distance in miles between a supplier 'sup' and a facility site 'site' using the Euclidean distance formula.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        sup : int
            Index of the supplier from 1 to 10
        site : int
            Index of the facility site from 1 to 12

        Returns
        -------
        Pyomo.Parameter
            The distance in miles between the supplier and the facility site
        """
        return sqrt((m.supplier_x[sup] - m.site_x[site]) ** 2 +
                    (m.supplier_y[sup] - m.site_y[site]) ** 2)

    @m.Param(m.potential_sites, m.markets, doc="Miles")
    def dist_site_to_market(m, site, mkt):
        """
        Calculate the distance in miles between a facility site 'site' and a market 'mkt' using the Euclidean distance formula.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        site : int
            Index of the facility site from 1 to 12
        mkt : int
            Index of the market from 1 to 10

        Returns
        -------
        Pyomo.Parameter
            The distance in miles between the facility site and the market
        """
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
    m.num_modules = Var(m.potential_sites, m.time, domain=Integers, bounds=(0, 10), doc="Number of modules")
    m.modules_purchased = Var(m.potential_sites, m.time, domain=Integers, bounds=(0, 10), doc="Modules purchased")
    m.modules_sold = Var(m.potential_sites, m.time, domain=Integers, bounds=(0, 10), doc="Modules sold")

    m.conv_build_cost = Var(
        m.potential_sites,
        doc="Cost of building conventional facility [milllion $]",
        bounds=(0, 1350 * 10), initialize=0)

    @m.Param(m.suppliers, m.time)
    def raw_material_unit_cost(m, sup, t):
        """
        Calculate the unit cost of raw materials for a given supplier 'sup' at time 't', based on the cost data provided in the Excel file.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        sup : int
            Index of the supplier from 1 to 10
        t : int
            Index of time in months from 0 to 120 (10 years)

        Returns
        -------
        Pyomo.Parameter
            The unit cost of raw materials for the supplier at time 't', calculated as the cost from the Excel data multiplied by the discount factor for time 't'.
        """
        return float(xls_data["sources"]["cost"][sup]) * m.discount_factor[t]

    @m.Param(m.time)
    def module_unit_cost(m, t):
        """
        Calculate the unit cost of modules at time 't', based on the cost data provided in the Excel file.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        t : int
            Index of time in months from 0 to 120 (10 years)

        Returns
        -------
        Pyomo.Parameter
            The unit cost of modules at time 't', calculated as the cost from the Excel data multiplied by the discount factor for time 't'.
        """
        return m.module_base_cost * m.discount_factor[t]

    @m.Param(m.time, doc="$/ton")
    def unit_production_cost(m, t):
        """
        Calculate the unit production cost at time 't', the production cost is 300 $/ton multiplied by the discount factor for time 't'.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        t : int
            Index of time in months from 0 to 120 (10 years)

        Returns
        -------
        Pyomo.Parameter
            The unit production cost at time 't', calculated as 300 $/ton multiplied by the discount factor for time 't'.
        """
        return 300 * m.discount_factor[t]

    @m.Param(doc="thousand $")
    def transport_fixed_cost(m):
        """
        Fixed cost of transportation in thousand dollars.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model

        Returns
        -------
        Pyomo.Parameter
            The fixed cost of transportation in thousand dollars, the cost is 125 thousand dollars.
        """
        return 125

    @m.Param(m.time, doc="$/ton-mile")
    def unit_product_transport_cost(m, t):
        """
        Calculate the unit product transport cost at time 't', the cost is 0.13 $/ton-mile multiplied by the discount factor for time 't'.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        t : int
            Index of time in months from 0 to 120 (10 years)

        Returns
        -------
        Pyomo.Parameter
            The unit product transport cost at time 't', calculated as 0.13 $/ton-mile multiplied by the discount factor for time 't'.
        """
        return 0.13 * m.discount_factor[t]

    @m.Param(m.time, doc="$/ton-mile")
    def unit_raw_material_transport_cost(m, t):
        """
        Calculate the unit raw material transport cost at time 't', the cost is 2 $/ton-mile multiplied by the discount factor for time 't'.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        t : int
            Index of time in months from 0 to 120 (10 years)

        Returns
        -------
        Pyomo.Parameter
            The unit raw material transport cost at time 't', calculated as 2 $/ton-mile multiplied by the discount factor for time 't'.
        """
        return 2 * m.discount_factor[t]

    m.supply_shipments = Var(
        m.suppliers, m.potential_sites, m.time, domain=NonNegativeReals,
        bounds=(0, 120 / 12 / 0.26), doc="thousand ton/mo")
    m.product_shipments = Var(
        m.potential_sites, m.markets, m.time, domain=NonNegativeReals,
        bounds=(0, 120 / 12), doc="thousand ton/mo")

    @m.Constraint(m.suppliers, m.time)
    def supply_limits(m, sup, t):
        """
        Ensure that the total supply from a supplier 'sup' at time 't' does not exceed the available supply from the supplier.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        sup : int
            Index of the supplier from 1 to 10
        t : int
            Index of time in months from 0 to 120 (10 years)

        Returns
        -------
        Pyomo.Constraint
            The total supply from the supplier 'sup' at time 't' should not exceed the available supply from the supplier.
        """
        return sum(m.supply_shipments[sup, site, t]
                   for site in m.potential_sites) <= m.available_supply[sup, t]

    @m.Constraint(m.markets, m.time)
    def demand_satisfaction(m, mkt, t):
        """
        Ensure that the total product shipments to a market 'mkt' at time 't' meets the market demand for the product.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        mkt : int
            Index of the market from 1 to 10
        t : int
            Index of time in months from 0 to 120 (10 years)

        Returns
        -------
        Pyomo.Constraint
            The total product shipments to the market 'mkt' at time 't' should meet the market demand for the product.
        """
        return sum(m.product_shipments[site, mkt, t]
                   for site in m.potential_sites) == m.market_demand[mkt, t]

    @m.Constraint(m.potential_sites, m.time)
    def product_balance(m, site, t):
        """
        Ensure that the total product shipments from a facility site 'site' at time 't' meets the production from the site.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        site : int
            Index of the facility site from 1 to 12
        t : int
            Index of time in months from 0 to 120 (10 years)

        Returns
        -------
        Pyomo.Constraint
            The total product shipments from the facility site 'site' at time 't' should meet the production from the site.
        """
        return m.production[site, t] == sum(m.product_shipments[site, mkt, t]
                                            for mkt in m.markets)

    @m.Constraint(m.potential_sites, m.time)
    def require_raw_materials(m, site, t):
        """
        Ensure that the raw materials required for production at a facility site 'site' at time 't' are available from the suppliers.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        site : int
            Index of the facility site from 1 to 12
        t : int
            Index of time in months from 0 to 120 (10 years)

        Returns
        -------
        Pyomo.Constraint
            The production at the facility site 'site' at time 't' should not exceed the raw materials available from the suppliers which is the supply multiplied by the conversion factor.
        """
        return m.production[site, t] <= m.conversion * m.supply[site, t]

    m.modular = Disjunct(m.potential_sites, rule=_build_modular_disjunct, doc="Disjunct for modular site")
    m.conventional = Disjunct(m.potential_sites, rule=_build_conventional_disjunct, doc="Disjunct for conventional site")
    m.site_inactive = Disjunct(m.potential_sites, rule=_build_site_inactive_disjunct, doc="Disjunct for inactive site")

    m.supply_route_active = Disjunct(m.suppliers, m.potential_sites, rule=_build_supply_route_active, doc="Disjunct for active supply route")
    m.supply_route_inactive = Disjunct(m.suppliers, m.potential_sites, rule=_build_supply_route_inactive, doc="Disjunct for inactive supply route")

    m.product_route_active = Disjunct(m.potential_sites, m.markets, rule=_build_product_route_active, doc="Disjunct for active product route")
    m.product_route_inactive = Disjunct(m.potential_sites, m.markets, rule=_build_product_route_inactive, doc="Disjunct for inactive product route")

    @m.Disjunction(m.potential_sites)
    def site_type(m, site):
        """
        Define the disjunction for the facility site type, which can be modular, conventional, or inactive.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        site : int
            Index of the facility site from 1 to 12

        Returns
        -------
        Pyomo.Disjunction
            The disjunction for the facility site type, which can be modular, conventional, or inactive.
        """
        return [m.modular[site], m.conventional[site], m.site_inactive[site]]

    @m.Disjunction(m.suppliers, m.potential_sites)
    def supply_route_active_or_not(m, sup, site):
        """
        Define the disjunction for the supply route between a supplier and a facility site, which can be active or inactive.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        sup : int
            Index of the supplier from 1 to 10
        site : int
            Index of the facility site from 1 to 12

        Returns
        -------
        Pyomo.Disjunction
            The disjunction for the supply route between a supplier and a facility site, which can be active or inactive.
        """
        return [m.supply_route_active[sup, site], m.supply_route_inactive[sup, site]]

    @m.Disjunction(m.potential_sites, m.markets)
    def product_route_active_or_not(m, site, mkt):
        """
        Define the disjunction for the product route between a facility site and a market, which can be active or inactive.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        site : int
            Index of the facility site from 1 to 12
        mkt : int
            Index of the market from 1 to 10

        Returns
        -------
        Pyomo.Disjunction
            The disjunction for the product route between a facility site and a market, which can be active or inactive.
        """
        return [m.product_route_active[site, mkt], m.product_route_inactive[site, mkt]]

    @m.Expression(m.suppliers, m.potential_sites, doc="million $")
    def raw_material_transport_cost(m, sup, site):
        """
        Calculate the cost of transporting raw materials from a supplier 'sup' to a facility site 'site' at each time period using the unit raw material transport cost, the supply shipments, and the distance between the supplier and the site.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        sup : int
            _description_
        site : int
            Index of the facility site from 1 to 12

        Returns
        -------
        Pyomo.Expression
            Total transportation cost considering the quantity of shipments, unit cost per time period, and distance between suppliers and sites.
        """
        return sum(
            m.supply_shipments[sup, site, t] # [1000 ton/month]
            * m.unit_raw_material_transport_cost[t] # [$/ton-mile]
            * m.dist_supplier_to_site[sup, site] / 1000 # [mile], [million/1000]
            for t in m.time)

    @m.Expression(doc="million $")
    def raw_material_fixed_transport_cost(m):
        """
        Calculate the fixed cost of transporting raw materials to the facility sites based on the total number of active supply routes and the fixed transportation cost.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model

        Returns
        -------
        Pyomo.Expression
            Sum of fixed transport costs, accounting for the activation of each route.
        """
        return (
            sum(m.supply_route_active[sup, site].binary_indicator_var
                for sup in m.suppliers for site in m.potential_sites)
            * m.transport_fixed_cost / 1000) # [thousand $] [million/1000]

    @m.Expression(m.potential_sites, m.markets, doc="million $")
    def product_transport_cost(m, site, mkt):
        """
        Calculate the cost of transporting products from a facility site 'site' to a market 'mkt' at each time period using the unit product transport cost, the product shipments, and the distance between the site and the market.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        site : int
            Index of the facility site from 1 to 12
        mkt : int
            _description_

        Returns
        -------
        Pyomo.Expression
            Total transportation cost considering the quantity of shipments, unit cost per time period, and distance between sites and markets.
        """
        return sum(
            m.product_shipments[site, mkt, t] # [1000 ton/month]
            * m.unit_product_transport_cost[t] # [$/ton-mile]
            * m.dist_site_to_market[site, mkt] / 1000 # [mile], [million/1000]
            for t in m.time)

    @m.Expression(doc="million $")
    def product_fixed_transport_cost(m):
        """
        Calculate the fixed cost of transporting products to the markets based on the total number of active product routes and the fixed transportation cost.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model

        Returns
        -------
        Pyomo.Expression
            Sum of fixed transport costs, accounting for the activation of each route.
        """
        return (
            sum(m.product_route_active[site, mkt].binary_indicator_var
                for site in m.potential_sites for mkt in m.markets)
            * m.transport_fixed_cost / 1000) # [thousand $] [million/1000]

    @m.Expression(m.potential_sites, m.time, doc="Cost of module setups in each month [million $]")
    def module_setup_cost(m, site, t):
        """
        Calculate the cost of setting up modules at a facility site 'site' at each time period using the unit module cost and the number of modules purchased.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        site : int
            Index of the facility site from 1 to 12
        t : int
            Index of time in months from 0 to 120 (10 years)

        Returns
        -------
        Pyomo.Expression
            Total setup cost considering the quantity of modules purchased and the unit cost per time period.
        """
        return m.modules_purchased[site, t] * m.module_unit_cost[t]

    @m.Expression(m.potential_sites, m.time, doc="Value of module teardowns in each month [million $]")
    def module_teardown_credit(m, site, t):
        """
        Calculate the value of tearing down modules at a facility site 'site' at each time period using the unit module cost and the number of modules sold.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        site : int
            Index of the facility site from 1 to 12
        t : int
            Index of time in months from 0 to 120 (10 years)

        Returns
        -------
        Pyomo.Expression
            Total teardown value considering the quantity of modules sold and the unit cost per time period.
        """
        return m.modules_sold[site, t] * m.module_unit_cost[t] * m.teardown_value

    @m.Expression(m.potential_sites, doc="Conventional site salvage value")
    def conv_salvage_value(m, site):
        """
        Calculate the salvage value of a conventional facility site 'site' using the build cost, the discount factor for the last time period, and the conventional salvage value.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            Pyomo concrete model which descibes the multiperiod location-allocation optimization model
        site : int
            Index of the facility site from 1 to 12

        Returns
        -------
        Pyomo.Expression
            Salvage value of the conventional facility site 'site' considering the build cost, discount factor, and salvage value.
        """
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
        sense=minimize, doc="Total cost [million $]")

    return m


def _build_site_inactive_disjunct(disj, site):
    """
    Configure the disjunct for a facility site marked as inactive.

    Parameters
    ----------
    disj : Pyomo.Disjunct
        Pyomo disjunct for inactive site
    site : int
        Index of the facility site from 1 to 12

    Returns
    -------
    None
        None, but adds constraints to the disjunct
    """
    m = disj.model()

    @disj.Constraint()
    def no_modules(disj):
        """
        Ensure that there are no modules at the inactive site.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            The disjunct object defining constraints for the inactive site

        Returns
        -------
        Pyomo.Constraint
            The constraint that there are no modules at the inactive site
        """
        return sum(m.num_modules[...]) + sum(m.modules_purchased[...]) + sum(m.modules_sold[...]) == 0

    @disj.Constraint()
    def no_production(disj):
        """
        Ensure that there is no production at the inactive site.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            The disjunct object defining constraints for the inactive site

        Returns
        -------
        Pyomo.Constraint
            The constraint that there is no production at the inactive site.
        """
        return sum(m.production[site, t] for t in m.time) == 0

    @disj.Constraint()
    def no_supply(disj):
        """
        Ensure that there is no supply at the inactive site.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            The disjunct object defining constraints for the inactive site

        Returns
        -------
        Pyomo.Constraint
            The constraint that there is no supply at the inactive site.
        """
        return sum(m.supply[site, t] for t in m.time) == 0


def _build_conventional_disjunct(disj, site):
    """
    Configure the disjunct for a conventional facility site.

    Parameters
    ----------
    disj : Pyomo.Disjunct
        The disjunct object associated with a conventional site.
    site : int
        Index of the facility site from 1 to 12

    Returns
    -------
    None
        None, but adds constraints to the disjunct
    """
    m = disj.model()

    disj.cost_calc = Constraint(
        expr=m.conv_build_cost[site] == (
            m.conv_base_cost * (m.conv_site_size[site] / 10) ** m.conv_exponent), doc="the build cost for the conventional facility")
    # m.bigM[disj.cost_calc] = 7000

    @disj.Constraint(m.time)
    def supply_balance(disj, t):
        """
        Ensure that the supply at the conventional site meets the supply shipments from the suppliers.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            The disjunct object for a conventional site.
        t : int
            Index of time in months from 0 to 120 (10 years)

        Returns
        -------
        Pyomo.Constraint
            A constraint that the total supply at the site during each time period equals the total shipments received.
        """
        return m.supply[site, t] == sum(
            m.supply_shipments[sup, site, t] for sup in m.suppliers)

    @disj.Constraint(m.time)
    def conv_production_limit(conv_disj, t):
        """
        Limit the production at the site based on its capacity. No production is allowed before the setup time.

        Parameters
        ----------
        conv_disj : Pyomo.Disjunct
            The disjunct object for a conventional site.
        t : int
            Index of time in months from 0 to 120 (10 years)

        Returns
        -------
        Pyomo.Constraint
            A constraint that limits production to the site's capacity after setup and prohibits production before setup.
        """
        if t < m.conv_setup_time:
            return m.production[site, t] == 0
        else:
            return m.production[site, t] <= m.conv_site_size[site]

    @disj.Constraint()
    def no_modules(disj):
        """
        Ensure no modular units are present, purchased, or sold at the conventional site.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            The disjunct object for a conventional site.

        Returns
        -------
        Pyomo.Constraint
            A constraint that the number of modules (present, purchased, sold) at the site is zero.
        """
        return sum(m.num_modules[...]) + sum(m.modules_purchased[...]) + sum(m.modules_sold[...]) == 0


def _build_modular_disjunct(disj, site):
    """
    Configure the disjunct for a modular facility site.

    Parameters
    ----------
    disj : Pyomo.Disjunct
        The disjunct object associated with a modular site.
    site : int
        Index of the facility site from 1 to 12

    Returns
    -------
    None
        None, but adds constraints to the disjunct
    """
    m = disj.model()

    @disj.Constraint(m.time)
    def supply_balance(disj, t):
        """
        Ensure that the supply at the modular site meets the supply shipments from the suppliers.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            The disjunct object for a modular site.
        t : int
            Index of time in months from 0 to 120 (10 years)

        Returns
        -------
        Pyomo.Constraint
            A constraint that the total supply at the site during each time period equals the total shipments received.
        """
        return m.supply[site, t] == sum(
            m.supply_shipments[sup, site, t] for sup in m.suppliers)

    @disj.Constraint(m.time)
    def module_balance(disj, t):
        """
        Ensure that the number of modules at the site is consistent with the number of modules purchased and sold.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            The disjunct object for a modular site.
        t : int
            Index of time in months from 0 to 120 (10 years)

        Returns
        -------
        Pyomo.Constraint
            A constraint that maintains the number of modules based on previous balances, new purchases, and modules sold.
        """
        existing_modules = 0 if t == m.time.first() else m.num_modules[site, t - 1]
        new_modules = 0 if t < m.modular_setup_time else m.modules_purchased[site, t - m.modular_setup_time]
        sold_modules = m.modules_sold[site, t]
        return m.num_modules[site, t] == existing_modules + new_modules - sold_modules

    # Fix the number of modules to zero during the setup time
    for t in range(value(m.modular_setup_time)):
        m.num_modules[site, t].fix(0)

    @disj.Constraint(m.time)
    def modular_production_limit(mod_disj, t):
        """
        Limit the production at the site based on the number of modules present. No production is allowed before the setup time.

        Parameters
        ----------
        mod_disj : Pyomo.Disjunct
            The disjunct object for a modular site.
        t : int
            Index of time in months from 0 to 120 (10 years)

        Returns
        -------
        Pyomo.Constraint
            A constraint that limits production to the site's capacity after setup and prohibits production before setup.
        """
        return m.production[site, t] <= 10 * m.num_modules[site, t]


def _build_supply_route_active(disj, sup, site):
    """
    Build the disjunct for an active supply route from a supplier to a facility site.

    Parameters
    ----------
    disj : Pyomo.Disjunct
        The disjunct object for an active supply route
    sup : int
        Index of the supplier from 1 to 10
    site : int
        Index of the facility site from 1 to 12
    """
    m = disj.model()


def _build_supply_route_inactive(disj, sup, site):
    """
    Build the disjunct for an inactive supply route from a supplier to a facility site.

    Parameters
    ----------
    disj : Pyomo.Disjunct
        The disjunct object for an inactive supply route
    sup : int
        Index of the supplier from 1 to 10
    site : int
        Index of the facility site from 1 to 12

    Returns
    -------
    None
        None, but adds constraints to the disjunct
    """
    m = disj.model()

    @disj.Constraint()
    def no_supply(disj):
        """
        Ensure that there are no supply shipments from the supplier to the site.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            The disjunct object for an inactive supply route

        Returns
        -------
        Pyomo.Constraint
            A constraint that there are no supply shipments from the supplier to the site.
        """
        return sum(m.supply_shipments[sup, site, t] for t in m.time) == 0


def _build_product_route_active(disj, site, mkt):
    """
    Build the disjunct for an active product route from a facility site to a market.

    Parameters
    ----------
    disj : Pyomo.Disjunct
        The disjunct object for an active product route
    site : int
        Index of the facility site from 1 to 12
    mkt : int
        Index of the market from 1 to 10
    """
    m = disj.model()


def _build_product_route_inactive(disj, site, mkt):
    """
    Build the disjunct for an inactive product route from a facility site to a market.

    Parameters
    ----------
    disj : Pyomo.Disjunct
        The disjunct object for an inactive product route
    site : int
        Index of the facility site from 1 to 12
    mkt : int
        Index of the market from 1 to 10

    Returns
    -------
    None
        None, but adds constraints to the disjunct
    """
    m = disj.model()

    @disj.Constraint()
    def no_product(disj):
        """
        Ensure that there are no product shipments from the site to the market.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            _description_

        Returns
        -------
        Pyomo.Constraint
            A constraint that there are no product shipments from the site to the market.
        """
        return sum(m.product_shipments[site, mkt, t] for t in m.time) == 0


def print_nonzeros(var):
    """
    Print the nonzero values of a Pyomo variable

    Parameters
    ----------
    var : pyomo.Var
        Pyomo variable of the model
    """
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
        if m.site_inactive[site].binary_indicator_var.value == 0:
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
        if fabs(m.supply_route_active[sup, site].binary_indicator_var.value - 1) <= 1E-3:
            plt.arrow(m.supplier_x[sup], m.supplier_y[sup],
                      m.site_x[site] - m.supplier_x[sup],
                      m.site_y[site] - m.supplier_y[sup],
                      width=0.8, length_includes_head=True, color='b')
    for site, mkt in m.potential_sites * m.markets:
        if fabs(m.product_route_active[site, mkt].binary_indicator_var.value - 1) <= 1E-3:
            plt.arrow(m.site_x[site], m.site_y[site],
                      m.market_x[mkt] - m.site_x[site],
                      m.market_y[mkt] - m.site_y[site],
                      width=0.8, length_includes_head=True, color='r')
    plt.show()
