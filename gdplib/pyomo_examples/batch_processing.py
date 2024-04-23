#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from os.path import join

from pyomo.common.fileutils import this_file_dir
from pyomo.environ import *
from pyomo.gdp import *

# Problem from http://www.minlp.org/library/problem/index.php?i=172&lib=GDP
# We are minimizing the cost of a design of a plant with parallel processing units and storage tanks
# in between. We decide the number and volume of units, and the volume and location of the storage
# tanks. The problem is convexified and has a nonlinear objective and global constraints

# NOTE: When I refer to 'gams' in the comments, that is Batch101006_BM.gms for now. It's confusing
# because the _opt file is different (It has hard-coded bigM parameters so that each constraint 
# has the "optimal" bigM).


def build_model():
    """
    Constructs and initializes a Pyomo model for the batch processing problem. 

    The model is designed to minimize the total cost associated with the design and operation of a plant consisting of multiple
    parallel processing units with intermediate storage tanks. 
    It involves determining the optimal number and sizes of processing units, batch sizes for different products at various stages, and sizes and
    placements of storage tanks to ensure operational efficiency while meeting production requirements within a specified time horizon.

    Parameters
    ----------
    None

    Returns
    -------
    Pyomo.ConcreteModel
        An instance of the Pyomo ConcreteModel class representing the batch processing optimization model, 
        ready to be solved with an appropriate solver.

    References
    ----------
    [1] Ravemark, E. Optimization models for design and operation of chemical batch processes. Ph.D. Thesis, ETH Zurich, 1995. https://doi.org/10.3929/ethz-a-001591449
    [2] Vecchietti, A., & Grossmann, I. E. (1999). LOGMIP: a disjunctive 0–1 non-linear optimizer for process system models. Computers & chemical engineering, 23(4-5), 555-565. https://doi.org/10.1016/S0098-1354(97)87539-4
    """
    model = AbstractModel()

    # TODO: it looks like they set a bigM for each j. Which I need to look up how to do...
    model.BigM = Suffix(direction=Suffix.LOCAL)
    model.BigM[None] = 1000


    # Constants from GAMS
    StorageTankSizeFactor = 10
    StorageTankSizeFactorByProd = 3
    MinFlow = -log(10000)
    VolumeLB = log(300)
    VolumeUB = log(3500)
    StorageTankSizeLB = log(100)
    StorageTankSizeUB = log(15000)
    UnitsInPhaseUB = log(6)
    UnitsOutOfPhaseUB = log(6)
    # TODO: YOU ARE HERE. YOU HAVEN'T ACTUALLY MADE THESE THE BOUNDS YET, NOR HAVE YOU FIGURED OUT WHOSE
    # BOUNDS THEY ARE. AND THERE ARE MORE IN GAMS.

    # Sets

    model.PRODUCTS = Set(doc='Set of Products')
    model.STAGES = Set(doc='Set of Stages', ordered=True)
    model.PARALLELUNITS = Set(doc='Set of Parallel Units', ordered=True)

    # TODO: this seems like an over-complicated way to accomplish this task...
    def filter_out_last(model, j):
        """
        Filters out the last stage from the set of stages to avoid considering it in certain constraints 
        or disjunctions where the next stage would be required but doesn't exist.

        Parameters
        ----------
        model : Pyomo.ConcreteModel
            The Pyomo model for the batch processing optimization problem.
        j : int
            The index representing the stage in the processing sequence. Stages are ordered and include various
            processing steps required for product completion.

        Returns
        -------
        bool
            Returns True if the stage is not the last one in the set, False otherwise.
        """
        return j != model.STAGES.last()
    model.STAGESExceptLast = Set(initialize=model.STAGES, filter=filter_out_last)


    # TODO: these aren't in the formulation??
    # model.STORAGE_TANKS = Set()

    # Parameters

    model.HorizonTime = Param(doc='Horizon Time')
    model.Alpha1 = Param(doc='Cost Parameter of the units')
    model.Alpha2 = Param(doc='Cost Parameter of the intermediate storage tanks')
    model.Beta1 = Param(doc='Exponent Parameter of the units')
    model.Beta2 = Param(doc='Exponent Parameter of the intermediate storage tanks')

    model.ProductionAmount = Param(model.PRODUCTS)
    model.ProductSizeFactor = Param(model.PRODUCTS, model.STAGES)
    model.ProcessingTime = Param(model.PRODUCTS, model.STAGES)

    # These are hard-coded in the GAMS file, hence the defaults
    model.StorageTankSizeFactor = Param(model.STAGES, default=StorageTankSizeFactor)
    model.StorageTankSizeFactorByProd = Param(model.PRODUCTS, model.STAGES,
                                              default=StorageTankSizeFactorByProd)

    # TODO: bonmin wasn't happy and I think it might have something to do with this?
    # or maybe issues with convexity or a lack thereof... I don't know yet.
    # I made PRODUCTS ordered so I could do this... Is that bad? And it does index
    # from 1, right?
    def get_log_coeffs(model, k):
        """
        Calculates the logarithmic coefficients used in the model, typically for transforming linear 
        relationships into logarithmic form for optimization purposes.

        Parameters
        ----------
        model : Pyomo.ConcreteModel
            The Pyomo model for the batch processing optimization problem.
        k : int
            The index representing a parallel unit.

        Returns
        -------
        float
            The logarithm of the position of the parallel unit within its set, used as a coefficient in the model.
        """
        return log(model.PARALLELUNITS.ord(k))

    model.LogCoeffs = Param(model.PARALLELUNITS, initialize=get_log_coeffs, doc='Logarithmic Coefficients')

    # bounds
    model.volumeLB = Param(model.STAGES, default=VolumeLB, doc='Lower Bound of Volume of the Units')
    model.volumeUB = Param(model.STAGES, default=VolumeUB, doc='Upper Bound of Volume of the Units')
    model.storageTankSizeLB = Param(model.STAGES, default=StorageTankSizeLB, doc='Lower Bound of Storage Tank Size')
    model.storageTankSizeUB = Param(model.STAGES, default=StorageTankSizeUB, doc='Upper Bound of Storage Tank Size')
    model.unitsInPhaseUB = Param(model.STAGES, default=UnitsInPhaseUB, doc='Upper Bound of Units in Phase')
    model.unitsOutOfPhaseUB = Param(model.STAGES, default=UnitsOutOfPhaseUB, doc='Upper Bound of Units Out of Phase')


    # Variables

    # TODO: right now these match the formulation. There are more in GAMS...

    # unit size of stage j
    # model.volume = Var(model.STAGES)
    # # TODO: GAMS has a batch size indexed just by products that isn't in the formulation... I'm going
    # # to try to avoid it for the moment...
    # # batch size of product i at stage j
    # model.batchSize = Var(model.PRODUCTS, model.STAGES)
    # # TODO: this is different in GAMS... They index by stages too?
    # # cycle time of product i divided by batch size of product i
    # model.cycleTime = Var(model.PRODUCTS)
    # # number of units in parallel out-of-phase (or in phase) at stage j
    # model.unitsOutOfPhase = Var(model.STAGES)
    # model.unitsInPhase = Var(model.STAGES)
    # # TODO: what are we going to do as a boundary condition here? For that last stage?
    # # size of intermediate storage tank between stage j and j+1
    # model.storageTankSize = Var(model.STAGES)

    # variables for convexified problem
    # TODO: I am beginning to think these are my only variables actually.
    # GAMS never un-logs them, I don't think. And I think the GAMs ones
    # must be the log ones.
    def get_volume_bounds(model, j):
        """
        Defines the bounds for the volume of processing units at each stage.

        Parameters
        ----------
        model : Pyomo.ConcreteModel
            The Pyomo model for the batch processing optimization problem.
        j : int
            Index for the processing stages in the plant. Stages are ordered and include various processing steps.

        Returns
        -------
        tuple
            A tuple containing the lower and upper bounds for the volume of processing units at stage j..
        """
        return (model.volumeLB[j], model.volumeUB[j])
    model.volume_log = Var(model.STAGES, bounds=get_volume_bounds, doc='Logarithmic Volume of the Units')
    model.batchSize_log = Var(model.PRODUCTS, model.STAGES, doc='Logarithmic Batch Size of the Products')
    model.cycleTime_log = Var(model.PRODUCTS, doc='Logarithmic Cycle Time of the Products')

    def get_unitsOutOfPhase_bounds(model, j):
        """
        Defines the bounds for the logarithmic representation of the number of units out of phase at each stage.

        Parameters
        ----------
        model : Pyomo.ConcreteModel
            The Pyomo model for the batch processing optimization problem.
        j : int
            Index for the processing stages in the plant. Stages are ordered and include various processing steps.

        Returns
        -------
        tuple
            A tuple containing the lower and upper bounds for the logarithmic representation of the number of units out of phase at stage j.
        """
        return (0, model.unitsOutOfPhaseUB[j])
    model.unitsOutOfPhase_log = Var(model.STAGES, bounds=get_unitsOutOfPhase_bounds, doc='Logarithmic Units Out of Phase')

    def get_unitsInPhase_bounds(model, j):
        """
        Defines the allowable bounds for the logarithmic number of processing units operating in phase at a given stage in the manufacturing process.

        Parameters
        ----------
        model : Pyomo.ConcreteModel
            The Pyomo model for the batch processing optimization problem.
        j : int
            Index for the processing stages in the plant. Stages are ordered and include various processing steps.

        Returns
        -------
        tuple
            A tuple containing the minimum and maximum bounds for the logarithmic number of units in phase at stage j, ensuring model constraints are met.
        """
        return (0, model.unitsInPhaseUB[j])
    model.unitsInPhase_log = Var(model.STAGES, bounds=get_unitsInPhase_bounds, doc='Logarithmic Units In Phase')

    def get_storageTankSize_bounds(model, j):
        """
        Determines the lower and upper bounds for the logarithmic representation of the storage tank size between stages j and j+1.

        Parameters
        ----------
        model : Pyomo.ConcreteModel
            The Pyomo model for the batch processing optimization problem.
        j : int
            Index for the processing stages in the plant. Stages are ordered and include various processing steps.

        Returns
        -------
        tuple
            A tuple containing the lower and upper bounds for the storage tank size at the specified stage.
        """
        return (model.storageTankSizeLB[j], model.storageTankSizeUB[j])
    # TODO: these bounds make it infeasible...
    model.storageTankSize_log = Var(model.STAGES, bounds=get_storageTankSize_bounds, doc='Logarithmic Storage Tank Size')

    # binary variables for deciding number of parallel units in and out of phase
    model.outOfPhase = Var(model.STAGES, model.PARALLELUNITS, within=Binary, doc='Out of Phase Units')
    model.inPhase = Var(model.STAGES, model.PARALLELUNITS, within=Binary, doc='In Phase Units')

    # Objective

    def get_cost_rule(model):
        """
        Defines the objective function for the model, representing the total cost of the plant design.

        Parameters
        ----------
        model : Pyomo.ConcreteModel
            The Pyomo model for the batch processing optimization problem.

        Returns
        -------
        Pyomo.Expression
            A Pyomo expression representing the total cost of the plant design.

        Notes
        -----
        The cost is a function of the volume of processing units and the size of storage tanks, each scaled by respective cost 
        parameters and exponentiated to reflect non-linear cost relationships.
        """
        return model.Alpha1 * sum(exp(model.unitsInPhase_log[j] + model.unitsOutOfPhase_log[j] + \
                                              model.Beta1 * model.volume_log[j]) for j in model.STAGES) +\
            model.Alpha2 * sum(exp(model.Beta2 * model.storageTankSize_log[j]) for j in model.STAGESExceptLast)
    model.min_cost = Objective(rule=get_cost_rule, doc='Minimize the Total Cost of the Plant Design')

    # Constraints
    def processing_capacity_rule(model, j, i):
        """
        Ensures that the volume of each processing unit at stage j is sufficient to accommodate the batch size of product i, 
        taking into account the size factor of the product and the number of units in phase at that stage.

        Parameters
        ----------
        model : Pyomo.ConcreteModel
            The Pyomo model for the batch processing optimization problem.
        j : int
            Index for the processing stages in the plant. Stages are ordered and include various processing steps.
        i : int
            The index representing a specific product. Products have unique processing requirements, including
            batch sizes and processing times, that vary by stage.

        Returns
        -------
        Pyomo.Expression
            A Pyomo expression that defines the processing capacity constraint for product `i` at stage `j`.
        """
        return model.volume_log[j] >= log(model.ProductSizeFactor[i, j]) + model.batchSize_log[i, j] - \
            model.unitsInPhase_log[j]
    model.processing_capacity = Constraint(model.STAGES, model.PRODUCTS, rule=processing_capacity_rule, doc='Processing Capacity')

    def processing_time_rule(model, j, i):
        """
        Ensures that the cycle time for product i at stage j, adjusted for the number of out-of-phase units, meets the required processing time.

        Parameters
        ----------
        model : Pyomo.ConcreteModel
            The Pyomo model for the batch processing optimization problem.
        j : int
            Index for the processing stages in the plant. Stages are ordered and include various processing steps.
        i : int
            Product index, representing different products being processed in the plant, each with its own set of
            processing times across various stages.

        Returns
        -------
        Pyomo.Expression
            A Pyomo expression defining the constraint that the cycle time for processing product `i` at stage `j`
            must not exceed the maximum allowed, considering the batch size and the units out of phase at this stage.
        """
        return model.cycleTime_log[i] >= log(model.ProcessingTime[i, j]) - model.batchSize_log[i, j] - \
            model.unitsOutOfPhase_log[j]
    model.processing_time = Constraint(model.STAGES, model.PRODUCTS, rule=processing_time_rule, doc='Processing Time')

    def finish_in_time_rule(model):
        """
        Ensures that the total production time across all products does not exceed the defined time horizon for the process.

        Parameters
        ----------
        model : Pyomo.ConcreteModel
            The Pyomo model for the batch processing optimization problem.

        Returns
        -------
        Pyomo.Expression
            A Pyomo constraint expression ensuring the total production time does not exceed the time horizon for the plant.
        """
        return model.HorizonTime >= sum(model.ProductionAmount[i]*exp(model.cycleTime_log[i]) \
                                        for i in model.PRODUCTS)
    model.finish_in_time = Constraint(rule=finish_in_time_rule, doc='Finish in Time')


    # Disjunctions

    def storage_tank_selection_disjunct_rule(disjunct, selectStorageTank, j):
        """
        Defines the conditions under which a storage tank will be included or excluded between stages j and j+1. 
        This rule is applied to a disjunct, which is part of a disjunction representing this binary decision.

        Parameters
        ----------
        disjunct :  Pyomo.Disjunct
            A Pyomo Disjunct object representing a specific case within the disjunction.
        selectStorageTank : int
            A binary indicator (0 or 1) where 1 means a storage tank is included and 0 means it is not.
        j : int
            Index for the processing stages in the plant. Stages are ordered and include various processing steps.

        Returns
        -------
        None
            This function defines constraints within the disjunct based on the decision to include (selectStorageTank=1) or exclude (selectStorageTank=0) a storage tank. 
            Constraints ensure the storage tank's volume can accommodate the batch sizes at stage j and j+1 if included, or ensure batch size continuity if excluded.
        """
        model = disjunct.model()
        def volume_stage_j_rule(disjunct, i):
            """ 
            Ensures the storage tank size between stages j and j+1 is sufficient to accommodate the batch size of product i at stage j, considering the storage tank size factor.

            Parameters
            ----------
            disjunct : Pyomo.Disjunct
                The disjunct within which this constraint is defined.
            i : int
                Product index, representing different products being processed in the plant, each with its own set of
                processing times across various stages.

            Returns
            -------
            Pyomo.Constraint
                A constraint ensuring the storage tank size is sufficient for the batch size at stage j.
            """
            return model.storageTankSize_log[j] >= log(model.StorageTankSizeFactor[j]) + \
                model.batchSize_log[i, j]
        
        def volume_stage_jPlus1_rule(disjunct, i):
            """
            Ensures the storage tank size between stages j and j+1 is sufficient to accommodate the batch size of product i at stage j+1, considering the storage tank size factor.

            Parameters
            ----------
            disjunct : Pyomo.Disjunct
                The disjunct within which this constraint is defined.
            i : int
                Product index, representing different products being processed in the plant, each with its own set of
                processing times across various stages.

            Returns
            -------
            Pyomo.Constraint
                A constraint ensuring the storage tank size is sufficient for the batch size at stage j+1.
            """
            return model.storageTankSize_log[j] >= log(model.StorageTankSizeFactor[j]) + \
                model.batchSize_log[i, j+1]
        
        def batch_size_rule(disjunct, i):
            """
            Ensures the difference in batch sizes between stages j and j+1 for product i is within the acceptable limits defined by the storage tank size factor by product.

            Parameters
            ----------
            disjunct : Pyomo.Disjunct
               The disjunct within which this constraint is defined.
            i : int
                Product index, representing different products being processed in the plant, each with its own set of
                processing times across various stages.

            Returns
            -------
            Pyomo.Constraint
                A constraint enforcing acceptable batch size differences between stages j and j+1.
            """
            return inequality(-log(model.StorageTankSizeFactorByProd[i,j]),
                              model.batchSize_log[i,j] - model.batchSize_log[i, j+1],
                              log(model.StorageTankSizeFactorByProd[i,j]))
        
        def no_batch_rule(disjunct, i):
            """
            Enforces batch size continuity between stages j and j+1 for product i, applicable when no storage tank is selected.

            Parameters
            ----------
            disjunct : Pyomo.Disjunct
                The disjunct within which this constraint is defined.
            i : int
                Product index, representing different products being processed in the plant, each with its own set of
                processing times across various stages.

            Returns
            -------
            Pyomo.Constraint
                A constraint ensuring batch size continuity between stages j and j+1
            """
            return model.batchSize_log[i,j] - model.batchSize_log[i,j+1] == 0

        if selectStorageTank:
            disjunct.volume_stage_j = Constraint(model.PRODUCTS, rule=volume_stage_j_rule)
            disjunct.volume_stage_jPlus1 = Constraint(model.PRODUCTS,
                                                      rule=volume_stage_jPlus1_rule)
            disjunct.batch_size = Constraint(model.PRODUCTS, rule=batch_size_rule)
        else:
            # The formulation says 0, but GAMS has this constant.
            # 04/04: Francisco says volume should be free:
            # disjunct.no_volume = Constraint(expr=model.storageTankSize_log[j] == MinFlow)
            disjunct.no_batch = Constraint(model.PRODUCTS, rule=no_batch_rule)
    model.storage_tank_selection_disjunct = Disjunct([0,1], model.STAGESExceptLast,
                                           rule=storage_tank_selection_disjunct_rule, doc='Storage Tank Selection Disjunct')

    def select_storage_tanks_rule(model, j):
        """
        Defines a disjunction for the model to choose between including or not including a storage tank between stages j and j+1.

        Parameters
        ----------
        model : Pyomo.ConcreteModel
            The Pyomo model for the batch processing optimization problem.
        j : int
            Index for the processing stages in the plant. Stages are ordered and include various processing steps.

        Returns
        -------
        list
            A list of disjuncts representing the choices for including or not including a storage tank between stages j and j+1.
        """
        return [model.storage_tank_selection_disjunct[selectTank, j] for selectTank in [0,1]]
    model.select_storage_tanks = Disjunction(model.STAGESExceptLast, rule=select_storage_tanks_rule, doc='Select Storage Tanks')

    # though this is a disjunction in the GAMs model, it is more efficiently formulated this way:
    # TODO: what on earth is k? Number of Parallel units.
    def units_out_of_phase_rule(model, j):
        """
        Defines the constraints for the logarithmic representation of the number of units k out of phase in stage j.

        Parameters
        ----------
        model : Pyomo.ConcreteModel
            The Pyomo model for the batch processing optimization problem.
        j : int
            Index for the processing stages in the plant. Stages are ordered and include various processing steps.

        Returns
        -------
        None
            Adds a constraint to the Pyomo model representing the logarithmic sum of out-of-phase units at stage j. 
            This constraint is not returned but directly added to the model.

        Notes
        -----
        These are not directly related to disjunctions but more to the logical modeling of unit operations.
        """
        return model.unitsOutOfPhase_log[j] == sum(model.LogCoeffs[k] * model.outOfPhase[j,k] \
                                                   for k in model.PARALLELUNITS)
    model.units_out_of_phase = Constraint(model.STAGES, rule=units_out_of_phase_rule, doc='Units Out of Phase')

    def units_in_phase_rule(model, j):
        """_summary_
        Defines the constraints for the logarithmic representation of the number of units k in-phase in stage j.

        Parameters
        ----------
        model : Pyomo.ConcreteModel
            The Pyomo model for the batch processing optimization problem.
        j : int
            Index for the processing stages in the plant. Stages are ordered and include various processing steps.

        Returns
        -------
        None
            Incorporates a constraint into the Pyomo model that corresponds to the logarithmic sum of in-phase units at stage j. 
            The constraint is directly applied to the model without an explicit return value.

        Notes
        -----
        These are not directly related to disjunctions but more to the logical modeling of unit operations.
        """
        return model.unitsInPhase_log[j] == sum(model.LogCoeffs[k] * model.inPhase[j,k] \
                                                for k in model.PARALLELUNITS)
    model.units_in_phase = Constraint(model.STAGES, rule=units_in_phase_rule, doc='Units In Phase')

    def units_out_of_phase_xor_rule(model, j):
        """
        Enforces an exclusive OR (XOR) constraint ensuring that exactly one configuration for the number of units out of phase is selected at stage j.

        Parameters
        ----------
        model : Pyomo.ConcreteModel
            The Pyomo model for the batch processing optimization problem.
        j : int
            Index for the processing stages in the plant. Stages are ordered and include various processing steps.

        Returns
        -------
        Pyomo.Constraint
            A Pyomo constraint expression calculating the logarithmic representation of the number of units out of phase at stage j
        """
        return sum(model.outOfPhase[j,k] for k in model.PARALLELUNITS) == 1
    model.units_out_of_phase_xor = Constraint(model.STAGES, rule=units_out_of_phase_xor_rule, doc='Exclusive OR for Units Out of Phase')

    def units_in_phase_xor_rule(model, j):
        """
        Enforces an exclusive OR (XOR) constraint ensuring that exactly one configuration for the number of units in phase is selected at stage j.

        Parameters
        ----------
        model : Pyomo.ConcreteModel
            The Pyomo model for the batch processing optimization problem.
        j : int
            Index for the processing stages in the plant. Stages are ordered and include various processing steps.

        Returns
        -------
        Pyomo.Constraint
            A Pyomo constraint expression enforcing the XOR condition for units out of phase at stage j.
        """
        return sum(model.inPhase[j,k] for k in model.PARALLELUNITS) == 1
    model.units_in_phase_xor = Constraint(model.STAGES, rule=units_in_phase_xor_rule, doc='Exclusive OR for Units In Phase')

    return model.create_instance(join(this_file_dir(), 'batch_processing.dat'))


if __name__ == "__main__":
    m = build_model()
    TransformationFactory('gdp.bigm').apply_to(m)
    SolverFactory('gams').solve(m, solver='baron', tee=True, add_options=['option optcr=1e-6;'])
    m.min_cost.display()
