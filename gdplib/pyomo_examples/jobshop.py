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

#
# Jobshop example from http://www.gams.com/modlib/libhtml/logmip4.htm
#
# This model solves a jobshop scheduling, which has a set of jobs
# which must be processed in sequence of stages but not all jobs
# require all stages. A zero wait transfer policy is assumed between
# stages. To obtain a feasible solution it is necessary to eliminate
# all clashes between jobs. It requires that no two jobs be performed
# at any stage at any time. The objective is to minimize the makespan,
# the time to complete all jobs.
#
# References:
#
# Raman & Grossmann, Modelling and computational techniques for logic based integer programming, Computers and Chemical Engineering 18, 7, p.563-578, 1994.
#
# Aldo Vecchietti, LogMIP User's Manual, http://www.logmip.ceride.gov.ar/, 2007
#


def build_model():
    """
    Build the jobshop scheduling model, which has a set of jobs which must be processed in sequence of stages but not all jobs require all stages.
    A zero wait transfer policy is assumed between stages.
    To obtain a feasible solution it is necessary to eliminate all clashes between jobs.
    It requires that no two jobs be performed at any stage at any time. The objective is to minimize the makespan, the time to complete all jobs.

    References:
        Raman & Grossmann, Modelling and computational techniques for logic based integer programming, Computers and Chemical Engineering 18, 7, p.563-578, 1994.
        Aldo Vecchietti, LogMIP User's Manual, http://www.logmip.ceride.gov.ar/, 2007
    Args: None
    Returns: AbstractModel: jobshop scheduling model, which has a set of jobs which must be processed in sequence of stages but not all jobs require all stages.
    """
    model = AbstractModel()

    model.JOBS = Set(ordered=True, doc='Set of jobs')
    model.STAGES = Set(ordered=True, doc='Set of stages')
    model.I_BEFORE_K = RangeSet(0, 1)

    # Task durations
    model.tau = Param(model.JOBS, model.STAGES, default=0)

    # Total Makespan (this will be the objective)
    model.ms = Var()

    # Start time of each job
    def t_bounds(model, I):
        """
        Calculate the time bounds for the start time of each job in a scheduling model.

        Args:
            model (Pyomo.Abstractmodel): job shop scheduling model, which has a set of jobs which must be processed in sequence of stages but not all jobs require all stages.
                A zero wait transfer policy is assumed between stages.
            I (str): job index

        Returns:
            tuple: (lower bound, upper bound) for the start time of each job in a scheduling model.
        """
        return (0, sum(value(model.tau[idx]) for idx in model.tau))

    model.t = Var(
        model.JOBS,
        within=NonNegativeReals,
        bounds=t_bounds,
        doc='Start time of each job',
    )

    # Auto-generate the L set (potential collisions between 2 jobs at any stage.
    def _L_filter(model, I, K, J):
        """
        Filter for the L set (potential collisions between 2 jobs at any stage.

        Args:
            model (Pyomo.Abstractmodel): jobshop scheduling model, which has a set of jobs which must be processed in sequence of stages but not all jobs require all stages.
                A zero wait transfer policy is assumed between stages.
            I (str): job index
            K (str): job index that is greater than I (After I)
            J (int): stage index

        Returns:
            expression: True if I < K and the parameters, model.tau[I,J] and model.tau[K,J]
        """
        return I < K and model.tau[I, J] and model.tau[K, J]

    model.L = Set(
        initialize=model.JOBS * model.JOBS * model.STAGES,
        dimen=3,
        filter=_L_filter,
        doc='Set of potential collisions between 2 jobs at any stage',
    )

    # Makespan is greater than the start time of every job + that job's
    # total duration
    def _feas(model, I):
            """This function creates a constraint that ensures the makespan is greater than the sum of the start time of every job and that job's total duration.

            Args:
                model (Pyomo.Abstractmodel): jobshop scheduling model, which has a set of jobs which must be processed in sequence of stages but not all jobs require all stages.
                A zero wait transfer policy is assumed between stages.
                I (str): job index

            Returns:
                expression: True if the makespan is greater than the sum of the start time of every job and that job's total duration.
            """
            return model.ms >= model.t[I] + sum(model.tau[I, M] for M in model.STAGES)

    model.Feas = Constraint(
        model.JOBS,
        rule=_feas,
        doc='Makespan is greater than the start time of every job + that job'
        's total duration',
    )

    # Disjunctions to prevent clashes at a stage: This creates a set of
    # disjunct pairs: one if job I occurs before job K and the other if job
    # K occurs before job I.
    def _NoClash(disjunct, I, K, J, IthenK):
        """
        Disjunctions to prevent clashes at a stage: This creates a set of disjunct pairs: one if job I occurs before job K and the other if job K occurs before job I.

        Args:
            model (Pyomo.Disjunction): The disjunction of the model.
            I (str): job index
            K (str): job index that is greater than I (After I)
            J (int): stage index
            IthenK (bool): True if I occurs before K, False if K occurs before I

        Returns:
            None, but creates a disjunction to prevent clashes at a stage.
        """
        model = disjunct.model()
        lhs = model.t[I] + sum([M < J and model.tau[I, M] or 0 for M in model.STAGES])
        rhs = model.t[K] + sum([M < J and model.tau[K, M] or 0 for M in model.STAGES])
        if IthenK:
            disjunct.c = Constraint(expr=lhs + model.tau[I, J] <= rhs)
        else:
            disjunct.c = Constraint(expr=rhs + model.tau[K, J] <= lhs)

    model.NoClash = Disjunct(
        model.L,
        model.I_BEFORE_K,
        rule=_NoClash,
        doc='Disjunctions to prevent clashes at a stage',
    )

    # Define the disjunctions: either job I occurs before K or K before I
    def _disj(model, I, K, J):
        """
        Define the disjunctions: either job I occurs before K or K before I

        Args:
            model (Pyomo.Abstractmodel): jobshop scheduling model, which has a set of jobs which must be processed in sequence of stages but not all jobs require all stages.
            I (str): job index
            K (str): job index that is greater than I (After I)
            J (int): stage index

        Returns:
            list: list of disjunctions to prevent clashes at a stage.
        """
        return [model.NoClash[I, K, J, IthenK] for IthenK in model.I_BEFORE_K]

    model.disj = Disjunction(
        model.L,
        rule=_disj,
        doc='Define the disjunctions: either job I occurs before K or K before I',
    )

    # minimize makespan
    model.makespan = Objective(
        expr=model.ms, doc='Objective Function: Minimize the makespan'
    )
    return model


def build_small_concrete():
    """
    Build a small jobshop scheduling model for testing purposes.
    The AbstractModel is instantiated with the data in the file jobshop-small.dat turning it into a ConcreteModel.

    Args:
        None, but the data file jobshop-small.dat must be in the same directory as this file.

    Returns:
        ConcreteModel: jobshop scheduling model, which has a set of jobs which must be processed in sequence of stages but not all jobs require all stages.
            A zero wait transfer policy is assumed between stages.
    """
    return build_model().create_instance(join(this_file_dir(), 'jobshop-small.dat'))


if __name__ == "__main__":
    m = build_small_concrete()
    TransformationFactory('gdp.bigm').apply_to(m)
    SolverFactory('gams').solve(
        m, solver='baron', tee=True, add_options=['option optcr=1e-6;']
    )
    m.makespan.display()
