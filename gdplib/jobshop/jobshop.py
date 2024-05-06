#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
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
# Raman & Grossmann, Modelling and computational techniques for logic based integer programming, Computers and Chemical Engineering 18, 7, p.563-578, 1994. DOI: 10.1016/0098-1354(93)E0010-7.
#
# Aldo Vecchietti, LogMIP User's Manual, http://www.logmip.ceride.gov.ar/, 2007
#


def build_model():
    """
    Build and return a jobshop scheduling model.

    This function constructs a Pyomo abstract model for jobshop scheduling, aiming to minimize the makespan.
    It includes sets of jobs and stages, with the assumption of a zero-wait policy between stages.
    The model enforces constraints to avoid job clashes at any stage and minimizes the total completion time.


    Parameters
    ----------
    None

    Returns
    -------
    model : Pyomo.AbstractModel
        The jobshop scheduling model, which has a set of jobs which must be processed in sequence of stages but not all jobs require all stages.

    References
    ----------
    Raman & Grossmann, Modelling and computational techniques for logic based integer programming, Computers and Chemical Engineering 18, 7, p.563-578, 1994.
    Aldo Vecchietti, LogMIP User's Manual, http://www.logmip.ceride.gov.ar/, 2007
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

        Parameters
        ----------
        model : Pyomo.Abstractmodel
            The job shop scheduling model, which has a set of jobs which must be processed in sequence of stages but not all jobs require all stages.
            A zero wait transfer policy is assumed between stages.
        I : str
            The index of the job index

        Returns
        -------
        tuple
            A tuple containing the lower and upper bounds for the start time of the job.
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
        Filter for the L set (potential collisions between 2 jobs at any stage).

        Parameters
        ----------
        model : Pyomo.Abstractmodel
            The jobshop scheduling model, which has a set of jobs which must be processed in sequence of stages but not all jobs require all stages.
            A zero wait transfer policy is assumed between stages.
        I : str
            job index
        K : str
            job index that is greater than I (After I)
        J : int
            stage index

        Returns
        -------
        bool
            Returns `True` if job `I` precedes job `K` and both jobs require processing at stage `J`, indicating a potential scheduling clash.
            'False' otherwise.
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
        """
        This function creates a constraint that ensures the makespan is greater than the sum of the start time of every job and that job's total duration.

        Parameters
        ----------
        model : Pyomo.Abstractmodel
            The jobshop scheduling model, which has a set of jobs which must be processed in sequence of stages but not all jobs require all stages.
            A zero wait transfer policy is assumed between stages.
        I : str
            job index

        Returns
        -------
        Pyomo.Constraint.Expression
            A constraint expression that ensures the makespan is greater than or equal to the sum of the start time and total duration for the job.
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

        Parameters
        ----------
        model : Pyomo.Disjunct
            The disjunction of the model.
        I : str
            job index
        K : str
            job index that is greater than I (After I)
        J : int
            stage index
        IthenK : bool
            A boolean flag indicating if job I is scheduled before job K (`True`) or vice versa (`False`).

        Returns
        -------
        None
            However, a constraint is added to the disjunction to prevent clashes at a stage.
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

        Parameters
        ----------
        model : Pyomo.Abstractmodel
            jobshop scheduling model, which has a set of jobs which must be processed in sequence of stages but not all jobs require all stages.
        I : str
            job index
        K : str
            job index that is greater than I (After I)
        J : int
            stage index

        Returns
        -------
        list of Pyomo.Disjunct
            A list of disjunctions for the given jobs and stage, enforcing that one job must precede the other to avoid clashes.
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
    model.create_instance(join(this_file_dir(), 'jobshop-small.dat'))
    return model


if __name__ == "__main__":
    m = build_model()
    TransformationFactory('gdp.bigm').apply_to(m)
    SolverFactory('gams').solve(
        m, solver='baron', tee=True, add_options=['option optcr=1e-6;']
    )
    m.makespan.display()
