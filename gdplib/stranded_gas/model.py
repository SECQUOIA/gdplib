"""
model.py

"""

import os

import pandas as pd

from pyomo.environ import (
    ConcreteModel, Constraint, Integers, NonNegativeReals, Objective, Param,
    RangeSet, Set, SolverFactory, Suffix, TransformationFactory, Var, exp, log,
    sqrt, summation, value
)
from gdplib.stranded_gas.util import alphanum_sorted
from pyomo.environ import TerminationCondition as tc


def build_model():
    """
    Constructs a Pyomo ConcreteModel for optimizing a modular stranded gas processing network. The model is designed to convert stranded gas into gasoline using a modular and intensified GTL process. It incorporates the economic dynamics of module investments, gas processing, and product transportation.

    Returns
    -------
    Pyomo.ConcreteModel
        A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL (Gas To Liquid) network. The model includes variables for the number of modules, their type and placement, as well as for production and transportation of gasoline. It aims to maximize the net present value (NPV) of the processing network.

    References
    ----------
    [1] Chen, Q., & Grossmann, I. E. (2019). Economies of numbers for a modular stranded gas processing network: Modeling and optimization. In Computer Aided Chemical Engineering (Vol. 47, pp. 257-262). Elsevier. DOI: 10.1016/B978-0-444-64241-7.50100-3
    """
    m = ConcreteModel('Stranded gas production')
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
        """
        Calculates the discount factor for a given time period in the model.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        t : int
            Set of time periods quarterly period within the 15 year project life.

        Returns
        -------
        Pyomo.Parameter
            A float representing the discount factor for the given time period.
        """
        return (1 + m.discount_rate / m.periods_per_year) ** (-t / m.periods_per_year)

    xlsx_data = pd.read_excel(os.path.join(os.path.dirname(__file__), "data.xlsx"), sheet_name=None)
    module_sheet = xlsx_data['modules'].set_index('Type')
    m.module_types = Set(initialize=module_sheet.columns.tolist(), doc="Module types")

    @m.Param(m.module_types)
    def module_base_cost(m, mtype):
        """
        Calculates the base cost of a module of a given type.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        mtype : str
            Index of the module type (A500, R500, S500, U500, A1000, R1000, S1000, U1000, A2000, R2000, S2000, U2000, A5000, R5000).

        Returns
        -------
        Pyomo.Parameter
            A float representing the base cost of a module of the given type.
        """
        return float(module_sheet[mtype]['Capital Cost [MM$]'])

    @m.Param(m.module_types, doc="Natural gas consumption per module of this type [MMSCF/d]")
    def unit_gas_consumption(m, mtype):
        """
        Calculates the natural gas consumption per module of a given type.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        mtype : str
             Index of the module type.

        Returns
        -------
        Pyomo.Parameter
            A float representing the natural gas consumption per module of the given type.
        """
        return float(module_sheet[mtype]['Nat Gas [MMSCF/d]'])

    @m.Param(m.module_types, doc="Gasoline production per module of this type [kBD]")
    def gasoline_production(m, mtype):
        """
        Calculates the gasoline production per module of a given type.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        mtype : str
            Index of the module type.

        Returns
        -------
        Pyomo.Parameter
            A float representing the gasoline production per module of the given type.
        """
        return float(module_sheet[mtype]['Gasoline [kBD]'])

    @m.Param(m.module_types, doc="Overall conversion of natural gas into gasoline per module of this type [kB/MMSCF]")
    def module_conversion(m, mtype):
        """
        Calculates the overall conversion of natural gas into gasoline per module of a given type.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        mtype : str
            Index of the module type.

        Returns
        -------
        Pyomo.Parameter
            A float representing the overall conversion of natural gas into gasoline per module of the given type.
        """
        return float(module_sheet[mtype]['Conversion [kB/MMSCF]'])

    site_sheet = xlsx_data['sites'].set_index('Potential site')
    m.potential_sites = Set(initialize=site_sheet.index.tolist(), doc="Potential sites")
    m.site_pairs = Set(
        doc="Pairs of potential sites",
        initialize=m.potential_sites * m.potential_sites,
        filter=lambda _, x, y: not x == y)

    @m.Param(m.potential_sites)
    def site_x(m, site):
        """
        Calculates the x-coordinate of a potential site.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        site : str
            The index for the potential site.

        Returns
        -------
        Pyomo.Parameter
            An integer representing the x-coordinate of the potential site.
        """
        return float(site_sheet['x'][site])

    @m.Param(m.potential_sites)
    def site_y(m, site):
        """
        Calculates the y-coordinate of a potential site.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        site : str
            The index for the potential site.

        Returns
        -------
        Pyomo.Parameter
            An integer representing the y-coordinate of the potential site.
        """
        return float(site_sheet['y'][site])

    well_sheet = xlsx_data['wells'].set_index('Well')
    m.well_clusters = Set(initialize=well_sheet.index.tolist(), doc="Well clusters")

    @m.Param(m.well_clusters)
    def well_x(m, well):
        """
        Calculates the x-coordinate of a well cluster.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        well : str
            The index for the well cluster. It starts from w1 and goes up to w12.

        Returns
        -------
        Pyomo.Parameter
            An integer representing the x-coordinate of the well cluster.
        """
        return float(well_sheet['x'][well])

    @m.Param(m.well_clusters)
    def well_y(m, well):
        """
        Calculates the y-coordinate of a well cluster.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        well : str
            The index for the well cluster.

        Returns
        -------
        Pyomo.Parameter
            An integer representing the y-coordinate of the well cluster.
        """
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
        """
        Calculates the supply of gas from a well cluster in a given time period.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        well : str
            The index for the well cluster.
        t : int
            Set of time periods quarterly period within the 15 year project life.

        Returns
        -------
        Pyomo.Parameter
            A float representing the supply of gas from the well cluster in the given time period.
        """
        return sum(well_profiles[well][t * 3:t * 3 + 2]) / 3

    mkt_sheet = xlsx_data['markets'].set_index('Market')
    m.markets = Set(initialize=mkt_sheet.index.tolist(), doc="Markets")

    @m.Param(m.markets)
    def mkt_x(m, mkt):
        """
        Calculates the x-coordinate of a market.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        mkt : str
            The index for the market. (m1, m2, m3)

        Returns
        -------
        Pyomo.Parameter
            An integer representing the x-coordinate of the market.
        """
        return float(mkt_sheet['x'][mkt])

    @m.Param(m.markets)
    def mkt_y(m, mkt):
        """
        Calculates the y-coordinate of a market.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        mkt : str
            The index for the market.

        Returns
        -------
        Pyomo.Parameter
            An integer representing the y-coordinate of the market.
        """
        return float(mkt_sheet['y'][mkt])

    @m.Param(m.markets, doc="Gasoline demand [kBD]")
    def mkt_demand(m, mkt):
        """
        Calculates the demand for gasoline in a market in a given time period.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        mkt : str
            The index for the market.

        Returns
        -------
        Pyomo.Parameter
            A float representing the demand for gasoline in the market in the given time period.
        """
        return float(mkt_sheet['demand [kBD]'][mkt])

    m.sources = Set(initialize=m.well_clusters | m.potential_sites, doc="Sources")
    m.destinations = Set(initialize=m.potential_sites | m.markets, doc="Destinations")

    @m.Param(m.sources, m.destinations, doc="Distance [mi]")
    def distance(m, src, dest):
        """
        Calculates the Euclidean distance between a source and a destination within the gas processing network.
        Assuming `src_x`, `src_y` for a source and `dest_x`, `dest_y` for a destination are defined within the model, the distance is calculated as follows:
    
        distance = sqrt((src_x - dest_x) ** 2 + (src_y - dest_y) ** 2)

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        src : str
            The identifier for the source, which could be a well cluster or a potential site.
        dest : str
            The identifier for the destination, which could be a potential site or a market.

        Returns
        -------
        Pyomo.Parameter
            A parameter representing the Euclidean distance between the source and destination.
        """
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
        """
        Calculates the module transport cost per mile in a given time period.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        t : int
            Set of time periods quarterly period within the 15 year project life.

        Returns
        -------
        Pyomo.Parameter
            A float representing the module transport cost per mile in the given time period.
        """
        return 50 * m.discount_factor[t]

    @m.Param(m.time, doc="Module transport cost per unit [MM$/module]")
    def module_transport_unit_cost(m, t):
        """
        Calculates the module transport cost per unit in a given time period.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        t : int
            Set of time periods quarterly period within the 15 year project life.

        Returns
        -------
        Pyomo.Parameter
            A float representing the module transport cost per unit in the given time period.
        """
        return 3 * m.discount_factor[t]

    @m.Param(m.time, doc="Stranded gas price [$/MSCF]")
    def nat_gas_price(m, t):
        """
        Calculates the price of stranded gas in a given time period.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        t : int
            Set of time periods quarterly period within the 15 year project life.

        Returns
        -------
        Pyomo.Parameter
            A float representing the price of stranded gas in the given time period.
        """
        return 5 * m.discount_factor[t]

    @m.Param(m.time, doc="Gasoline price [$/gal]")
    def gasoline_price(m, t):
        """
        Calculates the price of gasoline in a given time period.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        t : int
            Set of time periods quarterly period within the 15 year project life.

        Returns
        -------
        Pyomo.Parameter
            A float representing the price of gasoline in the given time period.
        """
        return 2.5 * m.discount_factor[t]

    @m.Param(m.time, doc="Gasoline transport cost [$/gal/100 miles]")
    def gasoline_tranport_cost(m, t):
        """
        Calculates the gasoline transport cost in a given time period.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        t : int
            Set of time periods quarterly period within the 15 year project life.

        Returns
        -------
        Pyomo.Parameter
            A float representing the gasoline transport cost in the given time period.
        """
        return 0.045 * m.discount_factor[t]

    m.gal_per_bbl = Param(initialize=42, doc="Gallons per barrel")
    m.days_per_period = Param(initialize=90, doc="Days in a production period")

    m.learning_factor = Var(
        m.module_types,
        doc="Fraction of cost due to economies of mass production",
        domain=NonNegativeReals, bounds=(0, 1), initialize=1)

    @m.Disjunct(m.module_types)
    def mtype_exists(disj, mtype):
        """
        Represents the scenario where a specific module type exists within the GTL network.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            A Pyomo Disjunct that represents the existence of a module type.
        mtype : str
            Index of the module type.

        Constraints
        ------------
        learning_factor_calc : Pyomo Constraint
            Captures the learning curve effect by adjusting the learning factor based on the total quantity of modules purchased.
        require_module_purchases : Pyomo Constraint
            Ensures that at least one module of this type is purchased, activating this disjunct.
        """
        disj.learning_factor_calc = Constraint(
            expr=m.learning_factor[mtype] == (1 - m.learning_rate) ** (
                log(sum(m.modules_purchased[mtype, :, :])) / log(2)), doc="Learning factor calculation")
        m.BigM[disj.learning_factor_calc] = 1
        disj.require_module_purchases = Constraint(
            expr=sum(m.modules_purchased[mtype, :, :]) >= 1, doc="At least one module purchase")

    @m.Disjunct(m.module_types)
    def mtype_absent(disj, mtype):
        """
        Represents the scenario where a specific module type does not exist within the GTL network.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            A Pyomo Disjunct that represents the absence of a module type.
        mtype : str
            Index of the module type.
        """
        disj.constant_learning_factor = Constraint(
            expr=m.learning_factor[mtype] == 1, doc="Constant learning factor")

    @m.Disjunction(m.module_types)
    def mtype_existence(m, mtype):
        """
        A disjunction that determines whether a module type exists or is absent within the GTL network.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        mtype : str
            Index of the module type.

        Returns
        -------
        list of Pyomo.Disjunct
            A list containing two disjuncts, one for the scenario where the module type exists and one for where it is absent.
        """
        return [m.mtype_exists[mtype], m.mtype_absent[mtype]]

    @m.Expression(m.module_types, m.time, doc="Module unit cost [MM$/module]")
    def module_unit_cost(m, mtype, t):
        """
        Computes the unit cost of a module type at a specific time period, considering the base cost, the learning factor due to economies of numbers, and the time-based discount factor.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        mtype : str
            Index of the module type.
        t : int
            Set of time periods quarterly period within the 15 year project life.

        Returns
        -------
        Pyomo.Expression
            A Pyomo Expression that calculates the total unit cost of a module for a given type and time period.
        """
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
        """
        Ensures that the natural gas consumption at any site for any module type does not exceed the production capacity of the modules present.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        site : str
            The index for the potential site.
        mtype : str
            Index of the module type.
        t : int
            Set of time periods quarterly period within the 15 year project life.

        Returns
        -------
        Pyomo.Constraint
            A constraint that limits the gas consumption per module type at each site, ensuring it does not exceed the capacity provided by the number of active modules of that type at the site during the time period.
        """
        return m.gas_consumption[site, mtype, t] <= (
            m.num_modules[mtype, site, t] * m.unit_gas_consumption[mtype])

    @m.Constraint(m.potential_sites, m.time)
    def production_limit(m, site, t):
        """
        Limits the production of gasoline at each site to the maximum possible based on the gas consumed and the conversion efficiency of the modules.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        site : str
            The index for the potential site.
        t : int
            Set of time periods quarterly period within the 15 year project life.

        Returns
        -------
        Pyomo.Constraint
            A constraint that ensures the production of gasoline at each site does not exceed the sum of the product of gas consumption and conversion rates for all module types at that site.
        """
        return m.production[site, t] <= sum(
            m.gas_consumption[site, mtype, t] * m.module_conversion[mtype]
            for mtype in m.module_types)

    @m.Expression(m.potential_sites, m.time)
    def capacity(m, site, t):
        """
        Calculates the total potential gasoline production capacity at each site for a given time period, based on the number of active modules, their gas consumption, and conversion efficiency.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        site : str
            The index for the potential site.
        t : int
            Set of time periods quarterly period within the 15 year project life.

        Returns
        -------
        Pyomo.Expression
            An expression that sums up the potential production capacity at a site, calculated as the product of the number of modules, their individual gas consumption rates, and their conversion efficiency.
        """
        return sum(
            m.num_modules[mtype, site, t] * m.unit_gas_consumption[mtype]
            * m.module_conversion[mtype] for mtype in m.module_types)

    @m.Constraint(m.potential_sites, m.time)
    def gas_supply_meets_consumption(m, site, t):
        """
        Ensures that the total gas consumed at a site is exactly matched by the gas supplied to it, reflecting a balance between supply and demand at any given time period.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        site : str
            The index for the potential site.
        t : int
            Set of time periods quarterly period within the 15 year project life.

        Returns
        -------
        Pyomo.Constraint
            A constraint that balances the gas supply with the gas consumption at each site, ensuring that the total gas flow to the site equals the total consumption.
        """
        return sum(m.gas_consumption[site, :, t]) == sum(m.gas_flows[:, site, t])

    @m.Constraint(m.well_clusters, m.time)
    def gas_supply_limit(m, well, t):
        """
        Ensures that the total gas supplied from a well cluster does not exceed the available gas supply for that cluster during any given time period.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        well : str
            The index for the well cluster.
        t : int
            Set of time periods quarterly period within the 15 year project life.

        Returns
        -------
        Pyomo.Constraint
            A constraint that limits the total gas flow from a well cluster to various sites to not exceed the gas supply available at that well cluster for the given time period.
        """
        return sum(m.gas_flows[well, site, t]
                   for site in m.potential_sites) <= m.gas_supply[well, t]

    @m.Constraint(m.potential_sites, m.time)
    def gasoline_production_requirement(m, site, t):
        """
        Ensures that the total amount of gasoline shipped from a site matches the production at that site for each time period.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        site : str
            The index for the potential site.
        t : int
            Set of time periods quarterly period within the 15 year project life.

        Returns
        -------
        Pyomo.Constraint
            A constraint that the sum of product flows (gasoline) from a site to various markets equals the total production at that site for the given period.
        """
        return sum(m.product_flows[site, mkt, t]
                   for mkt in m.markets) == m.production[site, t]

    @m.Constraint(m.potential_sites, m.module_types, m.time)
    def module_balance(m, site, mtype, t):
        """
        Balances the number of modules at a site across time periods by accounting for modules added, transferred, and previously existing. This ensures a consistent and accurate count of modules that reflects all transactions and changes over time.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        site : str
            The index for the potential site.
        mtype : str
            Index of the module type.
        t : int
            Set of time periods quarterly period within the 15 year project life.

        Returns
        -------
        Pyomo.Constraint
            A constraint that maintains an accurate balance of module counts at each site, considering new purchases, transfers in, existing inventory, and transfers out.
        """
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
        """
        Represents the active state of a potential site within the GTL network.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            A Pyomo Disjunct that represents the active state of a potential site.
        site : str
            The index for the potential site.
        """
        pass

    @m.Disjunct(m.potential_sites)
    def site_inactive(disj, site):
        """
        Represents the inactive state of a potential site within the GTL network.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            A Pyomo Disjunct that represents the inactive state of a potential site.
        site : str
            The index for the potential site.
        """
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
                if from_site == site or to_site == site) == 0, doc="No modules transferred")
        disj.no_modules_purchased = Constraint(
            expr=sum(
                m.modules_purchased[mtype, site, t]
                for mtype in m.module_types for t in m.time) == 0, doc="No modules purchased")

    @m.Disjunction(m.potential_sites)
    def site_active_or_not(m, site):
        """
        A disjunction that determines whether a potential site is active or inactive within the GTL network.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        site : str
            The index for the potential site.

        Returns
        -------
        list of Pyomo.Disjunct
            A list containing two disjuncts, one for the scenario where the site is active and one for where it is inactive.
        """
        return [m.site_active[site], m.site_inactive[site]]

    @m.Disjunct(m.well_clusters, m.potential_sites)
    def pipeline_exists(disj, well, site):
        """
        Represents the scenario where a pipeline exists between a well cluster and a potential site.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            _description_
        well : str
            The index for the well cluster.
        site : str
            The index for the potential site.
        """
        pass

    @m.Disjunct(m.well_clusters, m.potential_sites)
    def pipeline_absent(disj, well, site):
        """
        Represents the scenario where a pipeline does not exist between a well cluster and a potential site.

        Parameters
        ----------
        disj : Pyomo.Disjunct
            _description_
        well : str
            The index for the well cluster.
        site : str
            The index for the potential site.
        """
        disj.no_natural_gas_flow = Constraint(
            expr=sum(m.gas_flows[well, site, t] for t in m.time) == 0, doc="No natural gas flow")

    @m.Disjunction(m.well_clusters, m.potential_sites)
    def pipeline_existence(m, well, site):
        """
        A disjunction that determines whether a pipeline exists or is absent between a well cluster and a potential site.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        well : str
            The index for the well cluster.
        site : str
            The index for the potential site.

        Returns
        -------
        list of Pyomo.Disjunct
            A list containing two disjuncts, one for the scenario where a pipeline exists and one for where it is absent.
        """
        return [m.pipeline_exists[well, site], m.pipeline_absent[well, site]]

    # Objective Function Construnction
    @m.Expression(m.potential_sites, doc="MM$")
    def product_revenue(m, site):
        """
        Calculates the total revenue generated from the sale of gasoline produced at each site. This expression multiplies the volume of gasoline sold by the price per gallon, adjusted to millions of dollars for the entire production period.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        site : str
            The index for the potential site.

        Returns
        -------
        Pyomo.Expression
            An expression representing the total revenue in million dollars from selling the gasoline produced at the site.
        """
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
        """
        Calculates the total cost of natural gas consumed as a raw material at each site, converted to millions of dollars.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        site : str
            The index for the potential site.

        Returns
        -------
        Pyomo.Expression
            An expression calculating the total cost of natural gas used, taking into account the gas price and the conversion factor from MSCF to MMSCF.
        """
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
        """
        Computes the cost of transporting gasoline from each production site to different markets, expressed in million dollars.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        site : str
            The index for the potential site.
        mkt : str
            The index for the market.

        Returns
        -------
        Pyomo.Expression
            The total transportation cost for shipping gasoline from a site to a market, adjusted for the distance and transportation rate.
        """
        return sum(
            m.product_flows[site, mkt, t] * m.gal_per_bbl
            * 1000  # bbl/kB
            / 1E6  # $ to MM$
            * m.distance[site, mkt] / 100 * m.gasoline_tranport_cost[t]
            for t in m.time)

    @m.Expression(m.well_clusters, m.potential_sites, doc="MM$")
    def pipeline_construction_cost(m, well, site):
        """
        Calculates the cost of constructing pipelines from well clusters to potential sites, with costs dependent on the existence of a pipeline, distance, and unit cost per mile.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        well : str
            The index for the well cluster.
        site : str
            The index for the potential site.

        Returns
        -------
        Pyomo.Expression
            The cost of pipeline construction, in million dollars, if a pipeline is established between the well cluster and the site.
        """
        return (m.pipeline_unit_cost * m.distance[well, site]
                * m.pipeline_exists[well, site].binary_indicator_var)

    # Module transport cost
    @m.Expression(m.site_pairs, doc="MM$")
    def module_relocation_cost(m, from_site, to_site):
        """
        Calculates the cost of relocating modules from one site to another, considering the distance, transport cost per mile, and unit cost per module. This cost includes the transportation costs based on distance and per-unit transport costs, accounting for all modules transferred between specified sites over the entire project duration.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        from_site : str
            Index for the originating site of the module transfer.
        to_site : str
            Index for the destination site of the module transfer.

        Returns
        -------
        Pyomo.Expression
            An expression calculating the total relocation cost for modules moved between the two sites, factoring in the distance and both per-mile and per-unit costs, scaled to million dollars.
        """
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
        """
        Computes the total cost of purchasing new modules for a specific site, considering the unit costs of modules, which may vary over time due to discounts and other factors. This expression aggregates the costs for all modules purchased across the project's timeframe.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        site : str
            The index for the potential site.

        Returns
        -------
        Pyomo.Expression
            An expression representing the total cost of module purchases at the specified site, converted to million dollars.
        """
        return sum(
            m.module_unit_cost[mtype, t] * m.modules_purchased[mtype, site, t]
            for mtype in m.module_types
            for t in m.time)

    @m.Expression(doc="MM$")
    def profit(m):
        """
        Calculates the overall profit for the GTL network by subtracting all relevant costs from the total revenue. This is used as the objective function to be maximized (or minimize the negative profit).

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.

        Returns
        -------
        Pyomo.Expression
            The net profit expression, computed as the difference between total revenue and all accumulated costs across the network.
        """
        return (
            summation(m.product_revenue)
            - summation(m.raw_material_cost)
            - summation(m.product_transport_cost)
            - summation(m.pipeline_construction_cost)
            - summation(m.module_relocation_cost)
            - summation(m.module_purchase_cost)
        )

    m.neg_profit = Objective(expr=-m.profit, doc="Objective Function: Minimize Negative Profit")

    # Tightening constraints
    @m.Constraint(doc="Limit total module purchases over project span.")
    def restrict_module_purchases(m):
        """
        Enforces a limit on the total number of module purchases across all sites and module types throughout the entire project span. This constraint is crucial for controlling capital expenditure and ensuring that the module acquisition does not exceed a specified threshold, helping to maintain budget constraints.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.

        Returns
        -------
        Pyomo.Constraint
            A global constraint that limits the aggregate number of modules purchased across all sites to 5, ensuring that the total investment in module purchases remains within predefined limits.
        """
        return sum(m.modules_purchased[...]) <= 5

    @m.Constraint(m.site_pairs, doc="Limit transfers between any two sites")
    def restrict_module_transfers(m, from_site, to_site):
        """
        Imposes a limit on the number of modules that can be transferred between any two sites during the entire project timeline. This constraint helps manage logistics and ensures that module reallocation does not become overly frequent or excessive, which could lead to operational inefficiencies and increased costs.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            A Pyomo ConcreteModel that represents the multi-period optimization problem of a GTL network.
        from_site : str
            Index for the origin site from which modules are being transferred.
        to_site : str
            Index for the destination site to which modules are being transferred.

        Returns
        -------
        Pyomo.Constraint
            A constraint limiting the total number of modules transferred from one site to another to 5, providing a control mechanism on the frequency and volume of inter-site module movements.
        """
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
        m.mtype_absent[mtype].binary_indicator_var.fix(1)
