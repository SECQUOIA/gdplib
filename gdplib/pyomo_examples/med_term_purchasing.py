#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.environ import *
from pyomo.gdp import *
from pyomo.common.fileutils import this_file_dir
from os.path import join

# Medium-term Purchasing Contracts problem from http://minlp.org/library/lib.php?lib=GDP
# This model maximizes profit in a short-term horizon in which various contracts
# are available for purchasing raw materials. The model decides inventory levels,
# amounts to purchase, amount sold, and flows through the process nodes while
# maximizing profit. The four different contracts available are:
# FIXED PRICE CONTRACT: buy as much as you want at constant price
# DISCOUNT CONTRACT: quantities below minimum amount cost RegPrice. Any additional quantity
# above min amount costs DiscoutPrice.
# BULK CONTRACT: If more than min amount is purchased, whole purchase is at discount price.
# FIXED DURATION CONTRACT: Depending on length of time contract is valid, there is a purchase
# price during that time and min quantity that must be purchased


# This version of the model is a literal transcription of what is in
# ShortTermContractCH.gms from the website. Some data is hardcoded into this model,
# most notably the process structure itself and the mass balance information.

def build_model():
    """
    Build a Pyomo abstract model for the medium-term purchasing contracts problem.

    Returns
    -------
    Pyomo.AbstractModel
        Pyomo abstract model for medium-term purchasing contracts problem.

    Raises
    ------
    RuntimeError
        _description_
    RuntimeError
        _description_

    References
    ----------
    [1] Vecchietti, Aldo, and I. Grossmann. "Computational experience with logmip solving linear and nonlinear disjunctive programming problems." In Proc. of FOCAPD, pp. 587-590. 2004.
    [2] Vecchietti, A., S. Lee and I.E. Grossmann, “Modeling of Discrete/Continuous Optimization Problems: Characterization and Formulation of Disjunctions and their Relaxations,”
        Computers and Chemical Engineering 27, 433-448 (2003).
    """
    model = AbstractModel()

    # Constants (data that was hard-coded in GAMS model)
    AMOUNT_UB = 1000
    COST_UB = 1e4
    MAX_AMOUNT_FP = 1000
    MIN_AMOUNT_FD_1MONTH = 0

    RandomConst_Line264 = 0.17
    RandomConst_Line265 = 0.83

    ###################
    # Sets
    ###################

    # T
    # t in GAMS
    model.TimePeriods = Set(ordered=True, doc="Set of time periods")

    # Available length contracts
    # p in GAMS
    model.Contracts_Length = Set(doc="Set of available length contracts")

    # JP
    # final(j) in GAMS
    # Finished products
    model.Products = Set(doc="Set of finished products")

    # JM
    # rawmat(J) in GAMS
    # Set of Raw Materials-- raw materials, intermediate products, and final products partition J
    model.RawMaterials = Set(doc="Set of raw materials, intermediate products, and final products")

    # C
    # c in GAMS
    model.Contracts = Set(doc='Set of available contracts')

    # I
    # i in GAMS
    model.Processes = Set(doc='Set of processes in the network')

    # J
    # j in GAMS
    model.Streams = Set(doc='Set of streams in the network')


    ##################
    # Parameters
    ##################

    # Q_it
    # excap(i) in GAMS
    model.Capacity = Param(model.Processes, doc='Capacity of process i')

    # u_ijt
    # cov(i) in GAMS
    model.ProcessConstants = Param(model.Processes, doc='Process constants')

    # a_jt^U and d_jt^U
    # spdm(j,t) in GAMS
    model.SupplyAndDemandUBs = Param(model.Streams, model.TimePeriods, default=0, doc='Supply and demand upper bounds')

    # d_jt^L
    # lbdm(j, t) in GAMS
    model.DemandLB = Param(model.Streams, model.TimePeriods, default=0, doc='Demand lower bounds')

    # delta_it
    # delta(i, t) in GAMS
    # operating cost of process i at time t
    model.OperatingCosts = Param(model.Processes, model.TimePeriods, doc='Operating cost of process i at time t')

    # prices of raw materials under FP contract and selling prices of products
    # pf(j, t) in GAMS
    # omega_jt and pf_jt
    model.Prices = Param(model.Streams, model.TimePeriods, default=0, doc='Prices of raw materials under FP contract and selling prices of products')

    # Price for quantities less than min amount under discount contract
    # pd1(j, t) in GAMS
    model.RegPrice_Discount = Param(model.Streams, model.TimePeriod, doc='Price for quantities less than min amount under discount contract')

    # Discounted price for the quantity purchased exceeding the min amount
    # pd2(j,t0 in GAMS
    model.DiscountPrice_Discount = Param(model.Streams, model.TimePeriods, doc='Discounted price for the quantity purchased exceeding the min amount')

    # Price for quantities below min amount
    # pb1(j,t) in GAMS
    model.RegPrice_Bulk = Param(model.Streams, model.TimePeriods, doc='Price for quantities below min amount under bulk contract')

    # Price for quantities above min amount
    # pb2(j, t) in GAMS
    model.DiscountPrice_Bulk = Param(model.Streams, model.TimePeriods, doc='Price for quantities above minimum amount under bulk contract')

    # prices with length contract
    # pl(j, p, t) in GAMS
    model.Prices_Length = Param(model.Streams, model.Contracts_Length, model.TimePeriods, default=0, doc='Prices with length contract')

    # sigmad_jt
    # sigmad(j, t) in GAMS
    # Minimum quantity of chemical j that must be bought before recieving a Discount under discount contract
    model.MinAmount_Discount = Param(model.Streams, model.TimePeriods, default=0, doc='Minimum quantity of chemical j that must be bought before receiving a Discount under discount contract')

    # min quantity to recieve discount under bulk contract
    # sigmab(j, t) in GAMS
    model.MinAmount_Bulk = Param(model.Streams, model.TimePeriods, default=0, doc='Minimum quantity of chemical j that must be bought before receiving a Discount under bulk contract')

    # min quantity to recieve discount under length contract
    # sigmal(j, p) in GAMS
    model.MinAmount_Length = Param(model.Streams, model.Contracts_Length, default=0, doc='Minimum quantity of chemical j that must be bought before receiving a Discount under length contract')

    # main products of process i
    # These are 1 (true) if stream j is the main product of process i, false otherwise.
    # jm(j, i) in GAMS
    model.MainProducts = Param(model.Streams, model.Processes, default=0, doc='Main products of process i')

    # theta_jt
    # psf(j, t) in GAMS
    # Shortfall penalty of product j at time t
    model.ShortfallPenalty = Param(model.Products, model.TimePeriods, doc='Shortfall penalty of product j at time t')

    # shortfall upper bound
    # sfub(j, t) in GAMS
    model.ShortfallUB = Param(model.Products, model.TimePeriods, default=0, doc='Shortfall upper bound')

    # epsilon_jt
    # cinv(j, t) in GAMS
    # inventory cost of material j at time t
    model.InventoryCost = Param(model.Streams, model.TimePeriods, doc='Inventory cost of material j at time t')

    # invub(j, t) in GAMS
    # inventory upper bound
    model.InventoryLevelUB = Param(model.Streams, model.TimePeriods, default=0, doc='Inventory upper bound')

    ## UPPER BOUNDS HARDCODED INTO GAMS MODEL

    # All of these upper bounds are hardcoded. So I am just leaving them that way.
    # This means they all have to be the same as each other right now.
    def getAmountUBs(model, j, t):
        """
        Retrieves the upper bound for the amount that can be purchased or processed for a given material j and time period t.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
            Index of time period.

        Returns
        -------
        float
            The hardcoded upper bound on the amount for any material and time period, defined by the global variable `AMOUNT_UB`.
        """
        return AMOUNT_UB

    def getCostUBs(model, j, t):
        """
        Retrieves the upper bound for the cost associated with purchasing or processing a given material in a specific time period.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
            Index of time period.

        Returns
        -------
        float
            The hardcoded upper bound on costs for any material and time period, specified by the global variable `COST_UB`.
        """
        return COST_UB

    model.AmountPurchasedUB_FP = Param(model.Streams, model.TimePeriods,
        initialize=getAmountUBs, doc='Upper bound on amount purchased under fixed price contract')
    model.AmountPurchasedUB_Discount = Param(model.Streams, model.TimePeriods,
        initialize=getAmountUBs, doc='Upper bound on amount purchased under discount contract')
    model.AmountPurchasedBelowMinUB_Discount = Param(model.Streams, model.TimePeriods,
        initialize=getAmountUBs, doc='Upper bound on amount purchased below min amount for discount under discount contract')
    model.AmountPurchasedAboveMinUB_Discount = Param(model.Streams, model.TimePeriods,
        initialize=getAmountUBs, doc='Upper bound on amount purchased above min amount for discount under discount contract')
    model.AmountPurchasedUB_FD = Param(model.Streams, model.TimePeriods,
        initialize=getAmountUBs, doc='Upper bound on amount purchased under fixed duration contract')
    model.AmountPurchasedUB_Bulk = Param(model.Streams, model.TimePeriods,
        initialize=getAmountUBs, doc='Upper bound on amount purchased under bulk contract')

    model.CostUB_FP = Param(model.Streams, model.TimePeriods, initialize=getCostUBs, doc='Upper bound on cost of fixed price contract')
    model.CostUB_FD = Param(model.Streams, model.TimePeriods, initialize=getCostUBs, doc='Upper bound on cost of fixed duration contract')
    model.CostUB_Discount = Param(model.Streams, model.TimePeriods, initialize=getCostUBs, DOC='Upper bound on cost of discount contract')
    model.CostUB_Bulk = Param(model.Streams, model.TimePeriods, initialize=getCostUBs, doc='Upper bound on cost of bulk contract')


    ####################
    #VARIABLES
    ####################

    # prof in GAMS
    # will be objective
    model.Profit = Var(doc='Profit')

    # f(j, t) in GAMS
    # mass flow rates in tons per time interval t
    model.FlowRate = Var(model.Streams, model.TimePeriods, within=NonNegativeReals, doc='Mass flow rates in tons per time interval t')

    # V_jt
    # inv(j, t) in GAMS
    # inventory level of chemical j at time period t
    def getInventoryBounds(model, i, j):
        """
        Defines the lower and upper bounds for the inventory level of material j associated with process i.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        i : int
            Index of processes.
        j : int
            Index of materials.

        Returns
        -------
        tuple
            A tuple containing two floats: the lower bound (0) and the upper bound for the inventory level of material j. 
            The upper bound is retrieved from the model's 'InventoryLevelUB' parameter using indices i and j.
        """
        return (0, model.InventoryLevelUB[i,j])
    model.InventoryLevel = Var(model.Streams, model.TimePeriods,
        bounds=getInventoryBounds, doc='Inventory level of material j at time period t')

    # SF_jt
    # sf(j, t) in GAMS
    # Shortfall of demand for chemical j at time period t
    def getShortfallBounds(model, i, j):
        """
        Defines the lower and upper bounds for the shortfall in demand for material j during a specific time period, associated with process i.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        i : int
            Index of processes.
        j : int
            Index of materials.

        Returns
        -------
        tuple
            A tuple (0, upper_bound), where 0 is the lower bound (nonnegative shortfalls) and 'upper_bound' is extracted from 'model.ShortfallUB[i, j]'.
            'model.ShortfallUB[i, j]' represents the maximal permitted shortfall for material i in the context of process i.
        """
        return(0, model.ShortfallUB[i,j])
    model.Shortfall = Var(model.Products, model.TimePeriods,
        bounds=getShortfallBounds, doc='Shortfall of demand for material j at time period t')


    # amounts purchased under different contracts

    # spf(j, t) in GAMS
    # Amount of raw material j bought under fixed price contract at time period t
    def get_FP_bounds(model, j, t):
        """
        Determines the bounds for the amount of raw material 'j' that can be purchased under a fixed price contract during time period 't'.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
            Index of time period.

        Returns
        -------
        tuple
            A tuple (0, upper_bound), where '0' is the lower bound (non-negative constraint) and 'upper_bound' is sourced from 'model.AmountPurchasedUB_FP[j,t]'. 
            'model.AmountPurchasedUB_FP[j,t]' represents the maximum allowed purchase amount for material 'j' in period 't'
        """
        return (0, model.AmountPurchasedUB_FP[j,t])
    model.AmountPurchased_FP = Var(model.Streams, model.TimePeriods,
        bounds=get_FP_bounds, doc='Amount of raw material j bought under fixed price contract at time period t')

    # spd(j, t) in GAMS
    def get_Discount_Total_bounds(model, j, t):
        """
        Determines the lower and upper bounds for the total amount of material j that can be purchased under discount contracts during time period t.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
            Index of time period.

        Returns
        -------
        tuple
            A tuple containing two elements: the lower bound (0) and the upper bound for the total amount of material j that can be purchased under discount contracts in period t. 
            The upper bound is retrieved from 'model.AmountPurchasedUB_Discount[j, t]'.
        """
        return (0, model.AmountPurchasedUB_Discount[j,t])
    model.AmountPurchasedTotal_Discount = Var(model.Streams, model.TimePeriods,
        bounds=get_Discount_Total_bounds, doc='Total amount of material j bought under discount contract at time period t')

    # Amount purchased below min amount for discount under discount contract
    # spd1(j, t) in GAMS
    def get_Discount_BelowMin_bounds(model, j, t):
        """
        Determines the lower and upper bounds for the amount of material j purchased below the minimum quantity for discounts under discount contracts during time period t.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
            Index of time period.

        Returns
        -------
        tuple
            A tuple (0, upper_bound), where '0' is the lower bound, and 'upper_bound' is 'model.AmountPurchasedBelowMinUB_Discount[j, t]'. 
            'model.AmountPurchasedBelowMinUB_Discount[j, t] indicates the maximal purchasable amount below the discount threshold for material j in period't.
        """
        return (0, model.AmountPurchasedBelowMinUB_Discount[j,t])
    model.AmountPurchasedBelowMin_Discount = Var(model.Streams,
        model.TimePeriods, bounds=get_Discount_BelowMin_bounds, doc='Amount purchased below min amount for discount under discount contract')

    # spd2(j, t) in GAMS
    # Amount purchased above min amount for discount under discount contract
    def get_Discount_AboveMin_bounds(model, j, t):
        """
        Determines the lower and upper bounds for the amount of material j purchased above the minimum quantity eligible for discounts under discount contracts during time period t.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
            Index of time period.

        Returns
        -------
        tuple
            A tuple (0, upper_bound), where '0' is the lower bound and 'upper_bound' is sourced from 'model.AmountPurchasedBelowMinUB_Discount[j, t]'. 
            'model.AmountPurchasedBelowMinUB_Discount[j, t]' indicates the maximum amount that can be purchased above the discount threshold for material j in period t.
        """
        return (0, model.AmountPurchasedBelowMinUB_Discount[j,t])
    model.AmountPurchasedAboveMin_Discount = Var(model.Streams,
        model.TimePeriods, bounds=get_Discount_AboveMin_bounds, doc='Amount purchased above min amount for discount under discount contract')

    # Amount purchased under bulk contract
    # spb(j, t) in GAMS
    def get_bulk_bounds(model, j, t):
        """
        Sets the lower and upper bounds for the amount of material j that can be purchased under a bulk contract during time period t.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
            Index of time period.

        Returns
        -------
        tuple
            A tuple (0, upper_bound), where '0' is the lower bound and 'upper_bound' is sourced from 'model.AmountPurchasedUB_Bulk[j,t]'.
            'model.AmountPurchasedUB_Bulk[j,t]' indicates the maximal allowable bulk purchase amount for material 'j' in period 't'.
        """
        return (0, model.AmountPurchasedUB_Bulk[j,t])
    model.AmountPurchased_Bulk = Var(model.Streams, model.TimePeriods,
        bounds=get_bulk_bounds, doc='Amount purchased under bulk contract')

    # spl(j, t) in GAMS
    # Amount purchased under Fixed Duration contract
    def get_FD_bounds(model, j, t):
        """
        Sets the lower and upper bounds for the quantity of material j that can be acquired under a fixed duration contract during the time period t.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
            Index of time period.

        Returns
        -------
        tuple
            A tuple (0, upper_bound), where '0' is the lower bound and upper_bound is taken from 'model.AmountPurchasedUB_FD[j, t]'.
            'model.AmountPurchasedUB_FD[j, t]' indicates the maximum permissible purchase quantity for material j under a fixed duration contract in time period t.
        """
        return (0, model.AmountPurchasedUB_FD[j,t])
    model.AmountPurchased_FD = Var(model.Streams, model.TimePeriods,
        bounds=get_FD_bounds, doc='Amount purchased under Fixed Duration contract')


    # costs

    # costpl(j, t) in GAMS
    # cost of variable length contract
    def get_CostUBs_FD(model, j, t):
        """
        Build the lower and upper bounds for the cost of purchasing material j under a fixed duration contract during time period t.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
           Index of time period.

        Returns
        -------
        tuple
            A tuple (0, upper_bound), where '0' is the lower bound, the minimal cost scenario, and 'upper_bound' is retrieved from 'model.CostUB_FD[j, t]', 
            'model.CostUB_FD[j, t]' represents the highest permissible cost for purchasing material j under a fixed duration contract in time period t.
        """
        return (0, model.CostUB_FD[j,t])
    model.Cost_FD = Var(model.Streams, model.TimePeriods, bounds=get_CostUBs_FD, doc='Cost of variable length contract')

    # costpf(j, t) in GAMS
    # cost of fixed duration contract
    def get_CostUBs_FP(model, j, t):
        """
        Sets the lower and upper bounds for the cost of purchasing material j under a fixed price contract during time period t.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
            Index of time period.

        Returns
        -------
        tuple
            A tuple (0, upper_bound), where '0' is the lower bound and upper_bound is sourced from 'model.CostUB_FP[j, t]'.
            'model.CostUB_FP[j, t]' denotes the highest permissible cost for purchasing material 'j' under a fixed price contract in time period 't'
        """
        return (0, model.CostUB_FP[j,t])
    model.Cost_FP = Var(model.Streams, model.TimePeriods, bounds=get_CostUBs_FP, doc='Cost of fixed price contract')

    # costpd(j, t) in GAMS
    # cost of discount contract
    def get_CostUBs_Discount(model, j, t):
        """
        Set the lower and upper bounds for the cost of acquiring material j under discount contracts during time period t.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
            Index of time period.

        Returns
        -------
        tuple
            A tuple (0, upper bound), where 0' represents the lower bound, indicating the lowest possible cost condition, and upper bound is the 'model.CostUB_Discount[j, t]'. 
            'model.CostUB_Discount[j, t]' represents the maximum allowed cost for purchasing material j under a discount contract in the given time period t.
        """
        return (0, model.CostUB_Discount[j,t])
    model.Cost_Discount = Var(model.Streams, model.TimePeriods,
        bounds=get_CostUBs_Discount, doc='Cost of discount contract')

    # costpb(j, t) in GAMS
    # cost of bulk contract
    def get_CostUBs_Bulk(model, j, t):
        """
        Set the lower and upper bounds for the cost incurred from purchasing material j under bulk contracts during time period t.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
            Index of time period.

        Returns
        -------
        tuple
            A tuple with 0 as the lower bound and 'model.CostUB_Bulk[j, t]' as the upper bound.
            'model.CostUB_Bulk[j, t]' represents the maximum cost for purchasing material j under a bulk contract in time period t.
        """
        return (0, model.CostUB_Bulk[j,t])
    model.Cost_Bulk = Var(model.Streams, model.TimePeriods, bounds=get_CostUBs_Bulk, doc='Cost of bulk contract')


    # binary variables

    model.BuyFPContract = RangeSet(0,1) # buy fixed price contract
    model.BuyDiscountContract = Set(initialize=('BelowMin', 'AboveMin', 'NotSelected'), doc='Buy discount contract')
    model.BuyBulkContract = Set(initialize=('BelowMin', 'AboveMin', 'NotSelected'), doc='Buy bulk contract')
    model.BuyFDContract = Set(initialize=('1Month', '2Month', '3Month', 'NotSelected'), doc='Buy fixed duration contract')


    ################
    # CONSTRAINTS
    ################

    # Objective: maximize profit
    def profit_rule(model):
        """
        Objective function: maximize profit of the medium-term purchasing contracts problem.
        The profit  is given by sales revenues, operating costs, purchasing costs, inventory costs, and shortfall penalties

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.

        Returns
        -------
        Pyomo.Objective
            Objective function: maximize profit of the medium-term purchasing contracts problem.
        """
        salesIncome = sum(model.Prices[j,t] * model.FlowRate[j,t]
            for j in model.Products for t in model.TimePeriods)
        purchaseCost = sum(model.Cost_FD[j,t]
                for j in model.RawMaterials for t in model.TimePeriods) + \
            sum(model.Cost_Discount[j,t]
                for j in model.RawMaterials for t in model.TimePeriods) + \
            sum(model.Cost_Bulk[j,t]
                for j in model.RawMaterials for t in model.TimePeriods) + \
            sum(model.Cost_FP[j,t]
                for j in model.RawMaterials for t in model.TimePeriods)
        productionCost = sum(model.OperatingCosts[i,t] * sum(model.FlowRate[j,t]
            for j in model.Streams if model.MainProducts[j,i])
            for i in model.Processes for t in model.TimePeriods)
        shortfallCost = sum(model.Shortfall[j,t] * model.ShortfallPenalty[j, t]
            for j in model.Products for t in model.TimePeriods)
        inventoryCost = sum(model.InventoryCost[j,t] * model.InventoryLevel[j,t]
            for j in model.Products for t in model.TimePeriods)
        return salesIncome - purchaseCost - productionCost - inventoryCost - shortfallCost
    model.profit = Objective(rule=profit_rule, sense=maximize, doc='Maximize profit')

    # flow of raw materials is the total amount purchased (accross all contracts)
    def raw_material_flow_rule(model, j, t):
        """
        Ensures the total flow of raw material j in time period t equals the sum of amounts purchased under all contract types.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Constraint
            An equality constraint ensuring the material flow balance for each raw material j in each time period t.
        """
        return model.FlowRate[j,t] == model.AmountPurchased_FD[j,t] + \
            model.AmountPurchased_FP[j,t] + model.AmountPurchased_Bulk[j,t] + \
            model.AmountPurchasedTotal_Discount[j,t]
    model.raw_material_flow = Constraint(model.RawMaterials, model.TimePeriods,
        rule=raw_material_flow_rule, doc='Material flow balance for each raw material j in each time period t')

    def discount_amount_total_rule(model, j, t):
        """
        Balances the total amount of material j purchased under discount contracts in time period t with the sum of amounts purchased below and above the minimum discount threshold.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Constraint
            An equality constraint that ensures the total discounted purchase amount of material j in time period t is the sum of amounts bought below and above the discount threshold.
        """
        return model.AmountPurchasedTotal_Discount[j,t] == \
            model.AmountPurchasedBelowMin_Discount[j,t] + \
            model.AmountPurchasedAboveMin_Discount[j,t]
    model.discount_amount_total_rule = Constraint(model.RawMaterials, model.TimePeriods,
        rule=discount_amount_total_rule, doc='Total discounted purchase amount of material j in time period t is the sum of amounts bought below and above the discount threshold')

    # mass balance equations for each node
    # these are specific to the process network in this example.
    def mass_balance_rule1(model, t):
        """
        Represents the mass balance equation for the first node in the process network.
        
        Stream 1 is the inlet stream, and streams 2 and 3 are the outlet streams.
        The mass balance equation states that the total flow rate into the node (stream 1) is equal to the sum of the flow rates out of the node (streams 2 and 3).

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Constraint
            A constraint that enforces the mass balance equation for the first node in the process network.
        """
        return model.FlowRate[1, t] == model.FlowRate[2, t] + model.FlowRate[3, t]
    model.mass_balance1 = Constraint(model.TimePeriods, rule=mass_balance_rule1, doc='Mass balance equation for the first node in the process network')

    def mass_balance_rule2(model, t):
        """
        Represents the mass balance equation for the second node in the process network.

        Stream 4 and 8 are the inlet streams, and stream 5 is the outlet stream.
        The mass balance equation states that the total flow rate into the node (streams 4 and 8) is equal to the flow rate out of the node (stream 5).

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Constraint
            A constraint that enforces the mass balance equation for the second node in the process network.
        """
        return model.FlowRate[5, t] == model.FlowRate[4, t] + model.FlowRate[8,t]
    model.mass_balance2 = Constraint(model.TimePeriods, rule=mass_balance_rule2, doc='Mass balance equation for the second node in the process network')

    def mass_balance_rule3(model, t):
        """
        Represents the mass balance equation for the third node in the process network.

        Stream 6 is the inlet stream, and stream 7 is the outlet stream.
        The mass balance equation states that the total flow rate into the node (stream 6) is equal to the flow rate out of the node (stream 7).

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Constraint
            A constraint that enforces the mass balance equation for the third node in the process network.
        """
        return model.FlowRate[6, t] == model.FlowRate[7, t]
    model.mass_balance3 = Constraint(model.TimePeriods, rule=mass_balance_rule3, doc='Mass balance equation for the third node in the process network')

    def mass_balance_rule4(model, t):
        """
        Represents the mass balance equation for Process 2 in the process network.

        Stream 3 and Stream 5 are the inlet of Process 2.
        The mass flowrate of Stream 3 is 10 times the mass flowrate of Stream 5.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Constraint
            A constraint that enforces the mass balance equation for Process 2 in the process network.
        """
        return model.FlowRate[3, t] == 10*model.FlowRate[5, t]
    model.mass_balance4 = Constraint(model.TimePeriods, rule=mass_balance_rule4, doc='Mass balance equation for Process 2 in the process network')

    # process input/output constraints
    # these are also totally specific to the process network
    def process_balance_rule1(model, t):
        """
        Represents the input/output balance equation for Process 1 in the process network.

        Process 1 has input Streams 2 and output Stream 9.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Constraint
            A constraint that enforces the input/output balance equation for Process 1 in the process network.
        """
        return model.FlowRate[9, t] == model.ProcessConstants[1] * model.FlowRate[2, t]
    model.process_balance1 = Constraint(model.TimePeriods, rule=process_balance_rule1, doc='Input/output balance equation for Process 1 in the process network')

    def process_balance_rule2(model, t):
        """
        Represents the input/output balance equation for Process 2 in the process network.

        Process 2 has input Streams 5 and 3 and output Stream 10.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Constraint
            A constraint that enforces the input/output balance equation for Process 2 in the process network.
        """
        return model.FlowRate[10, t] == model.ProcessConstants[2] * \
            (model.FlowRate[5, t] + model.FlowRate[3, t])
    model.process_balance2 = Constraint(model.TimePeriods, rule=process_balance_rule2, doc='Input/output balance equation for Process 2 in the process network')

    def process_balance_rule3(model, t):
        """
        Represents the input/output balance equation for Process 3 in the process network.

        Process 3 has input Stream 7 and outputs Streams 11 and 8.
        RandomConst_Line264 is a hardcoded constant and determines the portion of Stream 7 that goes to Stream 8.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Constraint
            A constraint that enforces the input/output balance equation for Process 3 in the process network.
        """
        return model.FlowRate[8, t] == RandomConst_Line264 * \
            model.ProcessConstants[3] * model.FlowRate[7, t]
    model.process_balance3 = Constraint(model.TimePeriods, rule=process_balance_rule3, doc='Input/output balance equation 1 for Process 3 in the process network')

    def process_balance_rule4(model, t):
        """
        Represents the input/output balance equation for Process 3 in the process network.

        Process 3 has input Stream 7 and outputs Streams 11 and 8.
        RandomConst_Line265 is a hardcoded constant and determines the portion of Stream 7 that goes to Stream 11.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Constraint
            A constraint that enforces the input/output balance equation for Process 3 in the process network.
        """
        return model.FlowRate[11, t] == RandomConst_Line265 * \
            model.ProcessConstants[3] * model.FlowRate[7, t]
    model.process_balance4 = Constraint(model.TimePeriods, rule=process_balance_rule4, doc='Input/output balance equation 2 for Process 3 in the process network')

    # process capacity contraints
    # these are hardcoded based on the three processes and the process flow structure
    def process_capacity_rule1(model, t):
        """
        Set the capacity constraint for Process 1 in the process network.

        Process 1 has a capacity constraint on Stream 9, which is the output stream of Process 1.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Constraint
            A constraint that enforces the capacity constraint for Process 1 in the process network.
        """
        return model.FlowRate[9, t] <= model.Capacity[1]
    model.process_capacity1 = Constraint(model.TimePeriods, rule=process_capacity_rule1, doc='Capacity constraint for Process 1 in the process network')

    def process_capacity_rule2(model, t):
        """
        Set the capacity constraint for Process 2 in the process network.

        Process 2 has a capacity constraint on Stream 10, which is the output stream of Process 2.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Constraint
            A constraint that enforces the capacity constraint for Process 2 in the process network.
        """
        return model.FlowRate[10, t] <= model.Capacity[2]
    model.process_capacity2 = Constraint(model.TimePeriods, rule=process_capacity_rule2, doc='Capacity constraint for Process 2 in the process network')

    def process_capacity_rule3(model, t):
        """
        Set the capacity constraint for Process 3 in the process network.

        Process 3 has capacity constraints on Streams 11 and 8, which are the output streams of Process 3.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Constraint
            A constraint that enforces the capacity constraint for Process 3 in the process network.
        """
        return model.FlowRate[11, t] + model.FlowRate[8, t] <= model.Capacity[3]
    model.process_capacity3 = Constraint(model.TimePeriods, rule=process_capacity_rule3, doc='Capacity constraint for Process 3 in the process network')

    # Inventory balance of final products
    # again, these are hardcoded.

    def inventory_balance1(model, t):
        """
        Maintains inventory balance for the material associated with stream 12 at the first inventory node across time periods.

        This constraint ensures that the inventory level of the material at the beginning of each time period t, combined with the incoming flow from stream 9, equals the sum of the outflow to the next process (or demand) represented by stream 12 and the inventory level at the end of the time period. For the initial time period, the previous inventory is assumed to be zero. 
        This balance is vital for tracking inventory levels accurately, allowing the model to make informed decisions about production, storage, and sales to maximize overall profit.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Constraint
            A constraint enforcing the balance of inventory levels for the material flowing through stream 12 at the first inventory node, taking into account the material inflows and outflows as well as changes in inventory from the previous to the current time period.
        """
        prev = 0 if t == min(model.TimePeriods) else model.InventoryLevel[12, t-1]
        return prev + model.FlowRate[9, t] == model.FlowRate[12, t] + model.InventoryLevel[12,t]
    model.inventory_balance1 = Constraint(model.TimePeriods, rule=inventory_balance1, doc='Inventory balance for material associated with stream 12 at the first inventory node')

    def inventory_balance_rule2(model, t):
        """
        Ensures inventory balance for the material associated with stream 13 at the second inventory node for the first time period.

        This constraint is applied only to the first time period (t=1) and ensures that the sum of incoming flows from streams 10 and 11 equals the sum of the outflow to the next process represented by stream 13 and the inventory level at the end of the period. 
        For periods beyond the first, this constraint is skipped, as the balance for these periods may be governed by other conditions or constraints within the model. This selective application is crucial for accurately modeling the startup phase of the inventory system, where initial conditions significantly impact subsequent operations.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Constraint or Constraint.Skip
            A constraint enforcing the inventory balance for the material flowing through stream 13 at the second inventory node during the first time period. 
            For all other periods, the function returns `Constraint.Skip`, indicating no constraint is applied.
        """
        if t != 1:
            return Constraint.Skip
        return model.FlowRate[10, t] + model.FlowRate[11, t] == \
            model.InventoryLevel[13,t] + model.FlowRate[13, t]
    model.inventory_balance2 = Constraint(model.TimePeriods, rule=inventory_balance_rule2, doc='Inventory balance for material associated with stream 13 at the second inventory node')

    def inventory_balance_rule3(model, t):
        """
        Maintains the inventory balance for material associated with stream 13 at the second inventory node for all time periods after the first.

        This constraint is crucial for modeling the dynamic behavior of inventory levels over time, ensuring that the sum of the previous period's inventory level and the current period's inflows from streams 10 and 11 equals the current period's outflow (through stream 13) and ending inventory level. 
        It reflects the principle of inventory continuity, accounting for inflows, outflows, and storage from one period to the next. 
        The constraint is skipped for the first period (t=1) to accommodate initial conditions or startup behaviors specific to the model's context.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Constraint or Constraint.Skip
            A constraint enforcing the inventory balance for the material flowing through stream 13 at the second inventory node from the second period onwards. 
            For the first period, the function returns `Constraint.Skip`, indicating the constraint does not apply.
        """
        if t <= 1:
            return Constraint.Skip
        return model.InventoryLevel[13, t-1] + model.FlowRate[10, t] + \
            model.FlowRate[11,t] == model.InventoryLevel[13, t] + model.FlowRate[13, t]
    model.inventory_balance3 = Constraint(model.TimePeriods, rule=inventory_balance_rule3, doc='Inventory balance for material associated with stream 13 at the second inventory node')

    # Max capacities of inventories
    def inventory_capacity_rule(model, j, t):
        """
        Sets the maximum inventory capacity for each material j at each time period t.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Constraint
            A constraint that sets the maximum permissible inventory level for material j in time period t, ensuring that the inventory does not exceed the predefined upper bound 'InventoryLevelUB[j, t]'. 
            This maintains the model's alignment with practical storage limitations.
        """
        return model.InventoryLevel[j,t] <= model.InventoryLevelUB[j,t]
    model.inventory_capacity_rule = Constraint(model.Products, model.TimePeriods, rule=inventory_capacity_rule, doc='Maximum inventory capacity for each material j at each time period t')

    # Shortfall calculation
    def shortfall_rule(model, j, t):
        """
        Calculates the shortfall for each product 'j' in each time period 't'.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Constraint
            A constraint defining the shortfall for product j during time period t as the difference between the supply or demand upper bound and the actual flow rate. 
            This calculation is pivotal for evaluating performance and identifying bottlenecks or excess capacities within the supply chain.
        """
        return model.Shortfall[j, t] == model.SupplyAndDemandUBs[j, t] - model.FlowRate[j,t]
    model.shortfall = Constraint(model.Products, model.TimePeriods, rule=shortfall_rule, doc='Shortfall calculation for each product j in each time period t')

    # maximum shortfall allowed
    def shortfall_max_rule(model, j, t):
        """
        Imposes an upper limit on the shortfall allowed for each product j in each time period t.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Constraint
            A constraint that limits the shortfall for product j during time period t to a maximum value specified by 'ShortfallUB[j, t]'. 
            This constraint is instrumental in aligning the model's solutions with real-world operational constraints and strategic objectives.
        """
        return model.Shortfall[j, t] <= model.ShortfallUB[j, t]
    model.shortfall_max = Constraint(model.Products, model.TimePeriods, rule=shortfall_max_rule, doc='Maximum shortfall allowed for each product j in each time period t')

    # maxiumum capacities of suppliers
    def supplier_capacity_rule(model, j, t):
        """
        Enforces the upper limits on the supply capacity for each raw material j provided by suppliers in each time period t.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Constraint
            A constraint that limits the flow rate of raw material j from suppliers in time period t to not exceed the predefined upper bound 'SupplyAndDemandUBs[j, t]'. 
            This constraint is crucial for ensuring the feasibility of the supply chain model and its alignment with practical supply capabilities.
        """
        return model.FlowRate[j, t] <= model.SupplyAndDemandUBs[j, t]
    model.supplier_capacity = Constraint(model.RawMaterials, model.TimePeriods, rule=supplier_capacity_rule, doc='Maximum supply capacity for each raw material j in each time period t')

    # demand upper bound
    def demand_UB_rule(model, j, t):
        """
        Ensures that the supply of each product j does not exceed its maximum demand in each time period t.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Constraint
            A constraint limiting the flow rate of product j to not exceed the predefined maximum demand 'SupplyAndDemandUBs[j, t]' in time period t, ensuring production is demand-driven.
        """
        return model.FlowRate[j, t] <= model.SupplyAndDemandUBs[j,t]
    model.demand_UB = Constraint(model.Products, model.TimePeriods, rule=demand_UB_rule, doc='Maximum demand allowed for each product j in each time period t')

    # demand lower bound
    def demand_LB_rule(model, j, t):
        """
        Ensures that the supply of each product j meets at least the minimum demand in each time period t.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Constraint
            A constraint ensuring that the flow rate of product j meets or exceeds the minimum demand 'DemandLB[j, t]' in time period t, supporting effective market engagement.
        """
        return model.FlowRate[j, t] >= model.DemandLB[j,t]
    model.demand_LB = Constraint(model.Products, model.TimePeriods, rule=demand_LB_rule, doc='Minimum demand required for each product j in each time period t')


    # FIXED PRICE CONTRACT

    # Disjunction for Fixed Price contract buying options
    def FP_contract_disjunct_rule(disjunct, j, t, buy):
        """
        Defines disjunctive constraints for procurement decisions under a Fixed Price (FP) contract for material j in time period t.

        A decision must be made whether to engage in purchasing under the contract terms or not for each material in each time period. 
        This function encapsulates the disjunctive nature of this decision: if the decision is to buy ('buy' parameter is True), 
        the amount purchased under the FP contract is limited by a predefined maximum ('MAX_AMOUNT_FP'); 
        otherwise, no purchase is made under the FP contract for that material and period. 
        This disjunctive approach allows for modeling complex decision-making processes in procurement strategies.

        Parameters
        ----------
        disjunct : Pyomo.Disjunct
            A Pyomo Disjunct object representing a part of the disjunction. It encapsulates the constraints that are valid under a specific scenario ('buy' or not buy).
        j : int
            Index of materials.
        t : int
            Index of time period.
        buy : str
            A decision parameter indicating whether to purchase ('buy' is True) under the FP contract or not ('buy' is False) for material j in time period t

        Notes
        -----
        The 'buy' parameter is treated as a binary variable in the context of the model, where True indicates a decision to engage in purchasing under the FP contract, and False indicates otherwise.
        """
        model = disjunct.model()
        if buy:
            disjunct.c = Constraint(expr=model.AmountPurchased_FP[j,t] <= MAX_AMOUNT_FP)
        else:
            disjunct.c = Constraint(expr=model.AmountPurchased_FP[j,t] == 0)
    model.FP_contract_disjunct = Disjunct(model.RawMaterials, model.TimePeriods,
        model.BuyFPContract, rule=FP_contract_disjunct_rule, doc='Disjunctive constraints for Fixed Price contract buying options')

    # Fixed price disjunction
    def FP_contract_rule(model, j, t):
        """
        Creates a choice between buying or not buying materials under a Fixed Price contract for each material 'j' and time 't'.

        This function sets up a disjunction, which is like a crossroads for the model: for each material and time period, it can choose one of the paths defined in the 'FP_contract_disjunct_rule'.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Disjunction
            A disjunction that represents the decision-making point for FP contract purchases, contributing to the model's overall procurement strategy.
        """
        return [model.FP_contract_disjunct[j,t,buy] for buy in model.BuyFPContract]
    model.FP_disjunction = Disjunction(model.RawMaterials, model.TimePeriods,
        rule=FP_contract_rule, doc='Disjunction for Fixed Price contract buying options')

    # cost constraint for fixed price contract (independent constraint)
    def FP_contract_cost_rule(model, j, t):
        """_summary_

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Constraint
            A constraint equating the FP contract cost for material j in period t to the product of the purchased amount and its price, ensuring accurate financial accounting in the model.
        """
        return model.Cost_FP[j,t] == model.AmountPurchased_FP[j,t] * \
            model.Prices[j,t]
    model.FP_contract_cost = Constraint(model.RawMaterials, model.TimePeriods,
        rule=FP_contract_cost_rule, doc='Cost constraint for Fixed Price contract')


    # DISCOUNT CONTRACT

    # Disjunction for Discount contract
    def discount_contract_disjunct_rule(disjunct, j, t, buy):
        """
        Sets rules for purchasing materials j in time t under discount contracts based on the buying decision 'buy'. 

        For discount contracts, the decision involves purchasing below or above a minimum amount for a discount, or not selecting the contract at all. 
        This rule reflects these choices by adjusting purchasing amounts and enforcing corresponding constraints.

        Parameters
        ----------
        disjunct : Pyomo.Disjunct
            A Pyomo Disjunct object representing a part of the disjunction. It encapsulates the constraints that are valid under a specific scenario ('BelowMin', 'AboveMin', or 'NotSelected').
            Index of materials.
        t : int
            Index of time period.
        buy : str
             Decision on purchasing strategy: 'BelowMin', 'AboveMin', or 'NotSelected'.
        
        """
        model = disjunct.model()
        if buy == 'BelowMin':
            disjunct.belowMin = Constraint(
                expr=model.AmountPurchasedBelowMin_Discount[j,t] <= \
                model.MinAmount_Discount[j,t])
            disjunct.aboveMin = Constraint(
                expr=model.AmountPurchasedAboveMin_Discount[j,t] == 0)
        elif buy == 'AboveMin':
            disjunct.belowMin = Constraint(
                expr=model.AmountPurchasedBelowMin_Discount[j,t] == \
                model.MinAmount_Discount[j,t])
            disjunct.aboveMin = Constraint(
                expr=model.AmountPurchasedAboveMin_Discount[j,t] >= 0)
        elif buy == 'NotSelected':
            disjunct.belowMin = Constraint(
                expr=model.AmountPurchasedBelowMin_Discount[j,t] == 0)
            disjunct.aboveMin = Constraint(
                expr=model.AmountPurchasedAboveMin_Discount[j,t] == 0)
        else:
            raise RuntimeError("Unrecognized choice for discount contract: %s" % buy)
    model.discount_contract_disjunct = Disjunct(model.RawMaterials, model.TimePeriods,
        model.BuyDiscountContract, rule=discount_contract_disjunct_rule, doc='Disjunctive constraints for Discount contract buying options')

    # Discount contract disjunction
    def discount_contract_rule(model, j, t):
        """
        Determines the disjunction for purchasing under discount contracts for each material and time period, based on available decisions.

        This function sets up the model to choose among different discount purchasing strategies, enhancing the flexibility in procurement planning.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Disjunction
            The disjunction representing the choice among discount contract purchasing strategies.
        """
        return [model.discount_contract_disjunct[j,t,buy] \
            for buy in model.BuyDiscountContract]
    model.discount_contract = Disjunction(model.RawMaterials, model.TimePeriods,
        rule=discount_contract_rule, doc='Disjunction for Discount contract buying options')

    # cost constraint for discount contract (independent constraint)
    def discount_cost_rule(model, j, t):
        """
        Calculates the cost of purchasing material 'j' in time 't' under a discount contract, accounting for different price levels.

        This constraint ensures the model correctly accounts for the total cost of purchases under discount contracts, which may involve different prices based on quantity thresholds.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Constraint
            The constraint that calculates the total cost of purchases under discount contracts.
        """
        return model.Cost_Discount[j,t] == model.RegPrice_Discount[j,t] * \
            model.AmountPurchasedBelowMin_Discount[j,t] + \
            model.DiscountPrice_Discount[j,t] * model.AmountPurchasedAboveMin_Discount[j,t]
    model.discount_cost = Constraint(model.RawMaterials, model.TimePeriods,
        rule=discount_cost_rule, doc='Cost constraint for Discount contract')


    # BULK CONTRACT

    # Bulk contract buying options disjunct
    def bulk_contract_disjunct_rule(disjunct, j, t, buy):
        """
        Defines conditions for bulk purchases of material j at time t based on the decision 'buy'.

        This rule determines how much of a material is bought under a bulk contract and at what price, based on whether purchases are below or above a specified minimum amount, or if the bulk option is not selected. 
        It enforces different constraints for the amount and cost of materials under these scenarios.

        Parameters
        ----------
        disjunct :  Pyomo.Disjunct
            A Pyomo Disjunct object representing a part of the disjunction. It encapsulates the constraints that are valid under a specific scenario ('BelowMin', 'AboveMin', or 'NotSelected').
        j : int
            Index of materials.
        t : int
            Index of time period.
        buy : str
            The decision on how to engage with the bulk contract: 'BelowMin', 'AboveMin', or 'NotSelected'.

        """
        model = disjunct.model()
        if buy == 'BelowMin':
            disjunct.amount = Constraint(
                expr=model.AmountPurchased_Bulk[j,t] <= model.MinAmount_Bulk[j,t])
            disjunct.price = Constraint(
                expr=model.Cost_Bulk[j,t] == model.RegPrice_Bulk[j,t] * \
                model.AmountPurchased_Bulk[j,t])
        elif buy == 'AboveMin':
            disjunct.amount = Constraint(
                expr=model.AmountPurchased_Bulk[j,t] >= model.MinAmount_Bulk[j,t])
            disjunct.price = Constraint(
                expr=model.Cost_Bulk[j,t] == model.DiscountPrice_Bulk[j,t] * \
                model.AmountPurchased_Bulk[j,t])
        elif buy == 'NotSelected':
            disjunct.amount = Constraint(expr=model.AmountPurchased_Bulk[j,t] == 0)
            disjunct.price = Constraint(expr=model.Cost_Bulk[j,t] == 0)
        else:
            raise RuntimeError("Unrecognized choice for bulk contract: %s" % buy)
    model.bulk_contract_disjunct = Disjunct(model.RawMaterials, model.TimePeriods,
        model.BuyBulkContract, rule=bulk_contract_disjunct_rule, doc='Disjunctive constraints for Bulk contract buying options')

    # Bulk contract disjunction
    def bulk_contract_rule(model, j, t):
        """
        Establishes a decision-making framework for bulk purchases, allowing the model to choose among predefined scenarios.

        This function sets up a flexible structure for deciding on bulk purchases. 
        Each material and time period can be evaluated independently, allowing the model to adapt to various conditions and optimize procurement strategies under bulk contracts.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Disjunction
            A set of disjunctive conditions that the model can choose from when making bulk purchasing decisions.
        """
        return [model.bulk_contract_disjunct[j,t,buy] for buy in model.BuyBulkContract]
    model.bulk_contract = Disjunction(model.RawMaterials, model.TimePeriods,
        rule=bulk_contract_rule, doc='Disjunction for Bulk contract buying options')


    # FIXED DURATION CONTRACT

    def FD_1mo_contract(disjunct, j, t):
        """
        Defines the constraints for engaging in a 1-month fixed duration contract for material j at time t. 
        This includes a minimum purchase amount and the cost calculation based on contract-specific prices.

        Parameters
        ----------
        disjunct :  Pyomo.Disjunct
            A component representing the 1-month contract scenario.
        j : int
            Index of materials.
        t : int
            Index of time period.
        """
        model = disjunct.model()
        disjunct.amount1 = Constraint(expr=model.AmountPurchased_FD[j,t] >= \
            MIN_AMOUNT_FD_1MONTH)
        disjunct.price1 = Constraint(expr=model.Cost_FD[j,t] == \
            model.Prices_Length[j,1,t] * model.AmountPurchased_FD[j,t])
    model.FD_1mo_contract = Disjunct(
       model.RawMaterials, model.TimePeriods, rule=FD_1mo_contract, doc='1-month fixed duration contract')

    def FD_2mo_contract(disjunct, j, t):
        """
        Establishes conditions for a 2-month fixed duration contract. 
        This involves a minimum purchase requirement for two consecutive periods and corresponding cost calculations.

        Parameters
        ----------
        disjunct :  Pyomo.Disjunct
            The 2-month contract scenario component
        j : int
            Index of materials.
        t : int
            Index of time period.
        """
        model = disjunct.model()
        disjunct.amount1 = Constraint(expr=model.AmountPurchased_FD[j,t] >= \
            model.MinAmount_Length[j,2])
        disjunct.price1 = Constraint(expr=model.Cost_FD[j,t] == \
            model.Prices_Length[j,2,t] * model.AmountPurchased_FD[j,t])
       # only enforce these if we aren't in the last time period
        if t < model.TimePeriods[-1]:
            disjunct.amount2 = Constraint(expr=model.AmountPurchased_FD[j, t+1] >= \
                model.MinAmount_Length[j,2])
            disjunct.price2 = Constraint(expr=model.Cost_FD[j,t+1] == \
                model.Prices_Length[j,2,t] * model.AmountPurchased_FD[j, t+1])
    model.FD_2mo_contract = Disjunct(
       model.RawMaterials, model.TimePeriods, rule=FD_2mo_contract, doc='2-month fixed duration contract')

    def FD_3mo_contract(disjunct, j, t):
        """
        Sets up a 3-month fixed duration contract scenario with minimum purchase requirements extending over three periods and the cost calculation.

        Parameters
        ----------
        disjunct :  Pyomo.Disjunct
            The 3-month contract scenario component.
        j : int
            Index of materials.
        t : int
            Index of time period.
        """
        model = disjunct.model()
        # NOTE: I think there is a mistake in the GAMS file in line 327.
        # they use the bulk minamount rather than the length one.
        #I am doing the same here for validation purposes.
        disjunct.amount1 = Constraint(expr=model.AmountPurchased_FD[j,t] >= \
            model.MinAmount_Bulk[j,3])
        disjunct.cost1 = Constraint(expr=model.Cost_FD[j,t] == \
            model.Prices_Length[j,3,t] * model.AmountPurchased_FD[j,t])
        # check we aren't in one of the last two time periods
        if t < model.TimePeriods[-1]:
            disjunct.amount2 = Constraint(expr=model.AmountPurchased_FD[j,t+1] >= \
                model.MinAmount_Length[j,3])
            disjunct.cost2 = Constraint(expr=model.Cost_FD[j,t+1] == \
                model.Prices_Length[j,3,t] * model.AmountPurchased_FD[j,t+1])
        if t < model.TimePeriods[-2]:
            disjunct.amount3 = Constraint(expr=model.AmountPurchased_FD[j,t+2] >= \
                model.MinAmount_Length[j,3])
            disjunct.cost3 = Constraint(expr=model.Cost_FD[j,t+2] == \
                model.Prices_Length[j,3,t] * model.AmountPurchased_FD[j,t+2])
    model.FD_3mo_contract = Disjunct(
        model.RawMaterials, model.TimePeriods, rule=FD_3mo_contract, doc='3-month fixed duration contract')

    def FD_no_contract(disjunct, j, t):
        """
        Represents the scenario where no fixed duration contract is selected for material j at time t. 
        Ensures no purchases or costs are accounted for under FD contracts.

        Parameters
        ----------
        disjunct : Pyomo.Disjunct
            The 'no contract' scenario component.
        j : int
            Index of materials.
        t : int
            Index of time period.
        """
        model = disjunct.model()
        disjunct.amount1 = Constraint(expr=model.AmountPurchased_FD[j,t] == 0)
        disjunct.cost1 = Constraint(expr=model.Cost_FD[j,t] == 0)
        if t < model.TimePeriods[-1]:
            disjunct.amount2 = Constraint(expr=model.AmountPurchased_FD[j,t+1] == 0)
            disjunct.cost2 = Constraint(expr=model.Cost_FD[j,t+1] == 0)
        if t < model.TimePeriods[-2]:
            disjunct.amount3 = Constraint(expr=model.AmountPurchased_FD[j,t+2] == 0)
            disjunct.cost3 = Constraint(expr=model.Cost_FD[j,t+2] == 0)
    model.FD_no_contract = Disjunct(
        model.RawMaterials, model.TimePeriods, rule=FD_no_contract, doc='No fixed duration contract')

    def FD_contract(model, j, t):
        """
        Consolidates the FD contract scenarios into a single decision framework, allowing the model to choose the most optimal contract length or to not select an FD contract for each material and time period.

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        j : int
            Index of materials.
        t : int
            Index of time period.

        Returns
        -------
        Pyomo.Disjunction
            The disjunctive decision structure for FD contracts.
        """
        return [ model.FD_1mo_contract[j,t], model.FD_2mo_contract[j,t],
                model.FD_3mo_contract[j,t], model.FD_no_contract[j,t], ]
    model.FD_contract = Disjunction(model.RawMaterials, model.TimePeriods,
       rule=FD_contract, doc='Fixed duration contract scenarios')

    return model


def build_concrete():
    """
    Build a concrete model applying the data of the medium-term purchasing contracts problem on build_model().

    Returns
    -------
    Pyomo.ConcreteModel
        A concrete model for the medium-term purchasing contracts problem.
    """
    return build_model().create_instance(join(this_file_dir(), 'med_term_purchasing.dat'))


if __name__ == "__main__":
    m = build_concrete()
    TransformationFactory('gdp.bigm').apply_to(m)
    SolverFactory('gams').solve(m, solver='baron', tee=True, add_options=['option optcr=1e-6;'])
    m.profit.display()
