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
        _type_
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
        tuple
            _description_
        """
        return (0, model.CostUB_FD[j,t])
    model.Cost_FD = Var(model.Streams, model.TimePeriods, bounds=get_CostUBs_FD, doc='Cost of variable length contract')

    # costpf(j, t) in GAMS
    # cost of fixed duration contract
    def get_CostUBs_FP(model, j, t):
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
        tuple
            _description_
        """
        return (0, model.CostUB_FP[j,t])
    model.Cost_FP = Var(model.Streams, model.TimePeriods, bounds=get_CostUBs_FP, doc='Cost of fixed price contract')

    # costpd(j, t) in GAMS
    # cost of discount contract
    def get_CostUBs_Discount(model, j, t):
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
        tuple
            _description_
        """
        return (0, model.CostUB_Discount[j,t])
    model.Cost_Discount = Var(model.Streams, model.TimePeriods,
        bounds=get_CostUBs_Discount, doc='Cost of discount contract')

    # costpb(j, t) in GAMS
    # cost of bulk contract
    def get_CostUBs_Bulk(model, j, t):
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
        tuple
            _description_
        """
        return (0, model.CostUB_Bulk[j,t])
    model.Cost_Bulk = Var(model.Streams, model.TimePeriods, bounds=get_CostUBs_Bulk, doc='Cost of bulk contract')


    # binary variables

    model.BuyFPContract = RangeSet(0,1)
    model.BuyDiscountContract = Set(initialize=('BelowMin', 'AboveMin', 'NotSelected'))
    model.BuyBulkContract = Set(initialize=('BelowMin', 'AboveMin', 'NotSelected'))
    model.BuyFDContract = Set(initialize=('1Month', '2Month', '3Month', 'NotSelected'))


    ################
    # CONSTRAINTS
    ################

    # Objective: maximize profit
    def profit_rule(model):
        """
        Objective function: maximize profit

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
        _type_
            _description_
        """
        return model.FlowRate[j,t] == model.AmountPurchased_FD[j,t] + \
            model.AmountPurchased_FP[j,t] + model.AmountPurchased_Bulk[j,t] + \
            model.AmountPurchasedTotal_Discount[j,t]
    model.raw_material_flow = Constraint(model.RawMaterials, model.TimePeriods,
        rule=raw_material_flow_rule)

    def discount_amount_total_rule(model, j, t):
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
        _type_
            _description_
        """
        return model.AmountPurchasedTotal_Discount[j,t] == \
            model.AmountPurchasedBelowMin_Discount[j,t] + \
            model.AmountPurchasedAboveMin_Discount[j,t]
    model.discount_amount_total_rule = Constraint(model.RawMaterials, model.TimePeriods,
        rule=discount_amount_total_rule)

    # mass balance equations for each node
    # these are specific to the process network in this example.
    def mass_balance_rule1(model, t):
        """_summary_

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        _type_
            _description_
        """
        return model.FlowRate[1, t] == model.FlowRate[2, t] + model.FlowRate[3, t]
    model.mass_balance1 = Constraint(model.TimePeriods, rule=mass_balance_rule1)

    def mass_balance_rule2(model, t):
        """_summary_

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        _type_
            _description_
        """
        return model.FlowRate[5, t] == model.FlowRate[4, t] + model.FlowRate[8,t]
    model.mass_balance2 = Constraint(model.TimePeriods, rule=mass_balance_rule2)

    def mass_balance_rule3(model, t):
        return model.FlowRate[6, t] == model.FlowRate[7, t]
    model.mass_balance3 = Constraint(model.TimePeriods, rule=mass_balance_rule3)

    def mass_balance_rule4(model, t):
        """_summary_

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        _type_
            _description_
        """
        return model.FlowRate[3, t] == 10*model.FlowRate[5, t]
    model.mass_balance4 = Constraint(model.TimePeriods, rule=mass_balance_rule4)

    # process input/output constraints
    # these are also totally specific to the process network
    def process_balance_rule1(model, t):
        """_summary_

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        _type_
            _description_
        """
        return model.FlowRate[9, t] == model.ProcessConstants[1] * model.FlowRate[2, t]
    model.process_balance1 = Constraint(model.TimePeriods, rule=process_balance_rule1)

    def process_balance_rule2(model, t):
        """_summary_

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        _type_
            _description_
        """
        return model.FlowRate[10, t] == model.ProcessConstants[2] * \
            (model.FlowRate[5, t] + model.FlowRate[3, t])
    model.process_balance2 = Constraint(model.TimePeriods, rule=process_balance_rule2)

    def process_balance_rule3(model, t):
        """_summary_

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        _type_
            _description_
        """
        return model.FlowRate[8, t] == RandomConst_Line264 * \
            model.ProcessConstants[3] * model.FlowRate[7, t]
    model.process_balance3 = Constraint(model.TimePeriods, rule=process_balance_rule3)

    def process_balance_rule4(model, t):
        """_summary_

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        _type_
            _description_
        """
        return model.FlowRate[11, t] == RandomConst_Line265 * \
            model.ProcessConstants[3] * model.FlowRate[7, t]
    model.process_balance4 = Constraint(model.TimePeriods, rule=process_balance_rule4)

    # process capacity contraints
    # these are hardcoded based on the three processes and the process flow structure
    def process_capacity_rule1(model, t):
        """_summary_

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        _type_
            _description_
        """
        return model.FlowRate[9, t] <= model.Capacity[1]
    model.process_capacity1 = Constraint(model.TimePeriods, rule=process_capacity_rule1)

    def process_capacity_rule2(model, t):
        """_summary_

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        _type_
            _description_
        """
        return model.FlowRate[10, t] <= model.Capacity[2]
    model.process_capacity2 = Constraint(model.TimePeriods, rule=process_capacity_rule2)

    def process_capacity_rule3(model, t):
        """_summary_

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        _type_
            _description_
        """
        return model.FlowRate[11, t] + model.FlowRate[8, t] <= model.Capacity[3]
    model.process_capacity3 = Constraint(model.TimePeriods, rule=process_capacity_rule3)

    # Inventory balance of final products
    # again, these are hardcoded.

    def inventory_balance1(model, t):
        """_summary_

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        _type_
            _description_
        """
        prev = 0 if t == min(model.TimePeriods) else model.InventoryLevel[12, t-1]
        return prev + model.FlowRate[9, t] == model.FlowRate[12, t] + model.InventoryLevel[12,t]
    model.inventory_balance1 = Constraint(model.TimePeriods, rule=inventory_balance1)

    def inventory_balance_rule2(model, t):
        """_summary_

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        _type_
            _description_
        """
        if t != 1:
            return Constraint.Skip
        return model.FlowRate[10, t] + model.FlowRate[11, t] == \
            model.InventoryLevel[13,t] + model.FlowRate[13, t]
    model.inventory_balance2 = Constraint(model.TimePeriods, rule=inventory_balance_rule2)

    def inventory_balance_rule3(model, t):
        """_summary_

        Parameters
        ----------
        model : Pyomo.AbstractModel
            Pyomo abstract model for medium-term purchasing contracts problem.
        t : int
            Index of time period.

        Returns
        -------
        _type_
            _description_
        """
        if t <= 1:
            return Constraint.Skip
        return model.InventoryLevel[13, t-1] + model.FlowRate[10, t] + \
            model.FlowRate[11,t] == model.InventoryLevel[13, t] + model.FlowRate[13, t]
    model.inventory_balance3 = Constraint(model.TimePeriods, rule=inventory_balance_rule3)

    # Max capacities of inventories
    def inventory_capacity_rule(model, j, t):
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
        _type_
            _description_
        """
        return model.InventoryLevel[j,t] <= model.InventoryLevelUB[j,t]
    model.inventory_capacity_rule = Constraint(model.Products, model.TimePeriods, rule=inventory_capacity_rule)

    # Shortfall calculation
    def shortfall_rule(model, j, t):
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
        _type_
            _description_
        """
        return model.Shortfall[j, t] == model.SupplyAndDemandUBs[j, t] - model.FlowRate[j,t]
    model.shortfall = Constraint(model.Products, model.TimePeriods, rule=shortfall_rule)

    # maximum shortfall allowed
    def shortfall_max_rule(model, j, t):
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
        _type_
            _description_
        """
        return model.Shortfall[j, t] <= model.ShortfallUB[j, t]
    model.shortfall_max = Constraint(model.Products, model.TimePeriods, rule=shortfall_max_rule)

    # maxiumum capacities of suppliers
    def supplier_capacity_rule(model, j, t):
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
        _type_
            _description_
        """
        return model.FlowRate[j, t] <= model.SupplyAndDemandUBs[j, t]
    model.supplier_capacity = Constraint(model.RawMaterials, model.TimePeriods, rule=supplier_capacity_rule)

    # demand upper bound
    def demand_UB_rule(model, j, t):
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
        _type_
            _description_
        """
        return model.FlowRate[j, t] <= model.SupplyAndDemandUBs[j,t]
    model.demand_UB = Constraint(model.Products, model.TimePeriods, rule=demand_UB_rule)
    # demand lower bound
    def demand_LB_rule(model, j, t):
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
        _type_
            _description_
        """
        return model.FlowRate[j, t] >= model.DemandLB[j,t]
    model.demand_LB = Constraint(model.Products, model.TimePeriods, rule=demand_LB_rule)


    # FIXED PRICE CONTRACT

    # Disjunction for Fixed Price contract buying options
    def FP_contract_disjunct_rule(disjunct, j, t, buy):
        """_summary_

        Parameters
        ----------
        disjunct : Pyomo.Disjunct
            _description_
        j : int
            Index of materials.
        t : int
            Index of time period.
        buy : _type_
            _description_
        """
        model = disjunct.model()
        if buy:
            disjunct.c = Constraint(expr=model.AmountPurchased_FP[j,t] <= MAX_AMOUNT_FP)
        else:
            disjunct.c = Constraint(expr=model.AmountPurchased_FP[j,t] == 0)
    model.FP_contract_disjunct = Disjunct(model.RawMaterials, model.TimePeriods,
        model.BuyFPContract, rule=FP_contract_disjunct_rule)

    # Fixed price disjunction
    def FP_contract_rule(model, j, t):
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
        _type_
            _description_
        """
        return [model.FP_contract_disjunct[j,t,buy] for buy in model.BuyFPContract]
    model.FP_disjunction = Disjunction(model.RawMaterials, model.TimePeriods,
        rule=FP_contract_rule)

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
        _type_
            _description_
        """
        return model.Cost_FP[j,t] == model.AmountPurchased_FP[j,t] * \
            model.Prices[j,t]
    model.FP_contract_cost = Constraint(model.RawMaterials, model.TimePeriods,
        rule=FP_contract_cost_rule)


    # DISCOUNT CONTRACT

    # Disjunction for Discount contract
    def discount_contract_disjunct_rule(disjunct, j, t, buy):
        """_summary_

        Parameters
        ----------
        disjunct : Pyomo.Disjunct
            _description_
        j : int
            Index of materials.
        t : int
            Index of time period.
        buy : str
            Decision parameter that indicates whether to buy under the fixed price contract or not.

        Raises
        ------
        RuntimeError
            _description_
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
        model.BuyDiscountContract, rule=discount_contract_disjunct_rule)

    # Discount contract disjunction
    def discount_contract_rule(model, j, t):
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
        _type_
            _description_
        """
        return [model.discount_contract_disjunct[j,t,buy] \
            for buy in model.BuyDiscountContract]
    model.discount_contract = Disjunction(model.RawMaterials, model.TimePeriods,
        rule=discount_contract_rule)

    # cost constraint for discount contract (independent constraint)
    def discount_cost_rule(model, j, t):
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
        _type_
            _description_
        """
        return model.Cost_Discount[j,t] == model.RegPrice_Discount[j,t] * \
            model.AmountPurchasedBelowMin_Discount[j,t] + \
            model.DiscountPrice_Discount[j,t] * model.AmountPurchasedAboveMin_Discount[j,t]
    model.discount_cost = Constraint(model.RawMaterials, model.TimePeriods,
        rule=discount_cost_rule)


    # BULK CONTRACT

    # Bulk contract buying options disjunct
    def bulk_contract_disjunct_rule(disjunct, j, t, buy):
        """_summary_

        Parameters
        ----------
        disjunct :  Pyomo.Disjunct
            -description_
        j : int
            Index of materials.
        t : int
            Index of time period.
        buy : str
            Decision parameter that indicates whether to buy under the fixed price contract or not.

        Raises
        ------
        RuntimeError
            _description_
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
        model.BuyBulkContract, rule=bulk_contract_disjunct_rule)

    # Bulk contract disjunction
    def bulk_contract_rule(model, j, t):
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
        _type_
            _description_
        """
        return [model.bulk_contract_disjunct[j,t,buy] for buy in model.BuyBulkContract]
    model.bulk_contract = Disjunction(model.RawMaterials, model.TimePeriods,
        rule=bulk_contract_rule)


    # FIXED DURATION CONTRACT

    def FD_1mo_contract(disjunct, j, t):
        """_summary_

        Parameters
        ----------
        disjunct :  Index
            _description_
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
       model.RawMaterials, model.TimePeriods, rule=FD_1mo_contract)

    def FD_2mo_contract(disjunct, j, t):
        """_summary_

        Parameters
        ----------
        disjunct :  Index
            _description_
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
       model.RawMaterials, model.TimePeriods, rule=FD_2mo_contract)

    def FD_3mo_contract(disjunct, j, t):
        """_summary_

        Parameters
        ----------
        disjunct :  Index
            _description_
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
        model.RawMaterials, model.TimePeriods, rule=FD_3mo_contract)

    def FD_no_contract(disjunct, j, t):
        """_summary_

        Parameters
        ----------
        disjunct :  Index
            _description_
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
        model.RawMaterials, model.TimePeriods, rule=FD_no_contract)

    def FD_contract(model, j, t):
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
        _type_
            _description_
        """
        return [ model.FD_1mo_contract[j,t], model.FD_2mo_contract[j,t],
                model.FD_3mo_contract[j,t], model.FD_no_contract[j,t], ]
    model.FD_contract = Disjunction(model.RawMaterials, model.TimePeriods,
       rule=FD_contract)

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
