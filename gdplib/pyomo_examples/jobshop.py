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

# Changes I made for the measurement
from pyomo.contrib import gdpopt
from datetime import datetime
import os
import json
import time
import sys

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
# Raman & Grossmann, Computers and Chemical Engineering 18, 7, p.563-578, 1994.
#
# Aldo Vecchietti, LogMIP User's Manual, http://www.logmip.ceride.gov.ar/, 2007
#

def build_model():
    model = AbstractModel()

    model.JOBS = Set(ordered=True)
    model.STAGES = Set(ordered=True)
    model.I_BEFORE_K = RangeSet(0,1)

    # Task durations
    model.tau = Param(model.JOBS, model.STAGES, default=0)

    # Total Makespan (this will be the objective)
    model.ms = Var()
    # Start time of each job
    def t_bounds(model, I):
        return (0, sum(value(model.tau[idx]) for idx in model.tau))
    model.t = Var( model.JOBS, within=NonNegativeReals, bounds=t_bounds )

    # Auto-generate the L set (potential collisions between 2 jobs at any stage.
    def _L_filter(model, I, K, J):
        return I < K and model.tau[I,J] and model.tau[K,J]
    model.L = Set( initialize=model.JOBS * model.JOBS * model.STAGES,
                   dimen=3, filter=_L_filter)

    # Makespan is greater than the start time of every job + that job's
    # total duration
    def _feas(model, I):
        return model.ms >= model.t[I] + sum(model.tau[I,M] for M in model.STAGES)
    model.Feas = Constraint(model.JOBS, rule=_feas)

    # Disjunctions to prevent clashes at a stage: This creates a set of
    # disjunct pairs: one if job I occurs before job K and the other if job
    # K occurs before job I.
    def _NoClash(disjunct, I, K, J, IthenK):
        model = disjunct.model()
        lhs = model.t[I] + sum([M<J and model.tau[I,M] or 0 for M in model.STAGES])
        rhs = model.t[K] + sum([M<J and model.tau[K,M] or 0 for M in model.STAGES])
        if IthenK:
            disjunct.c = Constraint(expr=lhs+model.tau[I,J]<=rhs)
        else:
            disjunct.c = Constraint(expr=rhs+model.tau[K,J]<=lhs)
    model.NoClash = Disjunct(model.L, model.I_BEFORE_K, rule=_NoClash)

    # Define the disjunctions: either job I occurs before K or K before I
    def _disj(model, I, K, J):
        return [model.NoClash[I,K,J,IthenK] for IthenK in model.I_BEFORE_K]
    model.disj = Disjunction(model.L, rule=_disj)

    # minimize makespan
    model.makespan = Objective(expr=model.ms)
    return model


def build_small_concrete():
    return build_model().create_instance(join(this_file_dir(), 'jobshop-small.dat'))

# New codes that I have added
def log_results(log_file, transformation, solver, solver_options, model, results, elapsed_time):
    with open(log_file, 'a') as f:  # 'a' opens the file for appending
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Attempt to extract the objective value
        try:
            objective = model.makespan.expr()
        except Exception as e:
            objective = str(e)

        # Solver status (e.g., ok, warning, error)
        solver_status = str(results.solver.status)

        # Termination condition (e.g., optimal, infeasible)
        termination_condition = str(results.solver.termination_condition)

        # Check if the solution is optimal
        solution_status = "Unknown"
        if results.solver.termination_condition == pyomo.opt.TerminationCondition.optimal:
            solution_status = "Optimal"
        elif results.solver.termination_condition == pyomo.opt.TerminationCondition.infeasible:
            solution_status = "Infeasible"
        # Add more conditions as needed

        log_entry = {
            'time': current_time,
            'transformation': transformation,
            'solver': solver,
            'solver_options': str(solver_options),
            'objective': str(objective),
            'solver_status': solver_status,
            'termination_condition': termination_condition,
            'solution_status': solution_status,
            'elapsed_time': elapsed_time  # Include elapsed time
        }
        f.write(json.dumps(log_entry) + '\n')

def run_and_log_gdp_reformulation(log_file, transformation, solver_name, solver_options=None):
    start_time = time.time()  # Start the timer
    
    m = build_small_concrete()
    TransformationFactory(transformation).apply_to(m)
    solver = SolverFactory(solver_name)

    if solver_name == 'gams' and solver_options is not None:
        # For GAMS, options are passed in a list to the `add_options` parameter of the `solve` method
        gams_options = [f'option {key}={value};' for key, value in solver_options.items()]
        results = solver.solve(m, tee=True, add_options=gams_options)
    else:
        if solver_options is not None:
            for option, value in solver_options.items():
                solver.options[option] = value
        results = solver.solve(m, tee=True)

    elapsed_time = time.time() - start_time  # Calculate elapsed time
    log_results(log_file, transformation, solver_name, solver_options, m, results, elapsed_time)

def run_and_log_gdpopt(log_file, strategies, NLP_solvers, MIP_solver, timelimit=900):
    for strategy in strategies:
        for nlp_solver in NLP_solvers:
            start_time = time.time()  # Start the timer for each run
            
            m = build_small_concrete()
            solver = SolverFactory('gdpopt')
            results = solver.solve(
                m,
                tee=True,
                strategy=strategy,
                nlp_solver='gams',
                nlp_solver_args={'solver': nlp_solver},
                mip_solver=MIP_solver,
                time_limit=timelimit,
            )
            
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            log_results(log_file, "None (Using GDPopt)", strategy, {'nlp_solver': nlp_solver, 'mip_solver': MIP_solver}, m, results, elapsed_time)

if __name__ == "__main__":
    # m = build_small_concrete()
    # TransformationFactory('gdp.bigm').apply_to(m)
    # SolverFactory('gams').solve(m, solver='scip', tee=True, add_options=['option optcr=1e-6;'])
    # m.makespan.display()
    
    # Changes I made on the measurement
    log_file = 'model_runs.log'
    transformations = ['gdp.bigm', 'gdp.hull']
    solvers = [('gams', {'solver': 'baron', 'optcr': 1e-6}), ('gams', {'solver': 'cplex', 'optcr': 1e-6}), ('gams', {'solver': 'scip', 'optcr': 1e-6})]
    strategies = ['LOA', 'GLOA']
    NLP_solvers = ['ipopth', 'knitro', 'conopt', 'baron']
    MIP_solver = 'gurobi'

    # Clear the log file at the start of the script
    with open(log_file, 'w') as f:
        pass # Opening in 'w' mode and closing the file clears it

    # Run and log GDP reformulation with direct solving
    for transformation in transformations:
        for solver, solver_options in solvers:
            run_and_log_gdp_reformulation(log_file, transformation, solver, solver_options)
    
    # Run and log GDPopt strategies without clearing the log file
    run_and_log_gdpopt(log_file, strategies, NLP_solvers, MIP_solver)

    # Read and print the contents of the log file
    with open(log_file, 'r') as f:
        print(f.read())

    # Specify the path to your log file
    log_file_path = "/local/scratch/a/lee4382/repository/gdplib/gdplib/pyomo_examples/jobshop_run_logs.json"

    # Read the contents of the original log file
    with open(log_file, 'r') as original_log_file:
        log_contents = original_log_file.read()

    # Write the contents to the new location
    with open(log_file_path, "w") as new_log_file:
        new_log_file.write(log_contents)

    print(f"Log entries have been saved to {log_file_path}")