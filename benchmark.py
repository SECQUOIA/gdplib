import os
import json
import time
import sys
from datetime import datetime
from importlib import import_module
from pyomo.environ import *
import sys
from contextlib import redirect_stdout


def benchmark(
    model, strategy, timelimit, result_dir, subsolver="gams", solver_gams="baron"
):
    """Benchmark the model using the given strategy and subsolver.

    The result files include the solver output and the JSON representation of the results.

    Parameters
    ----------
    model : PyomoModel
        the model to be solved
    strategy : string
        the strategy used to solve the model
    timelimit : int
        the time limit for the solver
    result_dir : string
        the directory to store the benchmark results
    subsolver : string
        the subsolver used to solve the model
    solver_gams : string
        the solver used to solve the model in GAMS

    Returns
    -------
    None
    """
    # We clone the model to avoid the solver starting from the optimal solution from previous solve.
    model = model.clone()

    # Direct the solver output to a file
    if strategy in ["gdp.bigm", "gdp.hull"]:
        with open(
            result_dir + "/" + strategy + "_" + subsolver + "_" + solver_gams + ".log",
            "w",
        ) as f:
            with redirect_stdout(f):
                transformation_start_time = time.time()
                TransformationFactory(strategy).apply_to(model)
                transformation_end_time = time.time()
                results = SolverFactory(subsolver).solve(
                    model,
                    tee=True,
                    solver=solver_gams,
                    add_options=[
                        "option reslim=3600;option threads=1;option optcr=1e-2;"
                    ],
                    # keepfiles=True,
                    # tmpdir=os.path.join(result_dir, strategy, "nlp"),
                    # symbolic_solver_labels=True,
                )
                results.solver.strategy = strategy
                results.solver.transformation_time = (
                    transformation_end_time - transformation_start_time
                )
                print(results)
    elif strategy in [
        "gdpopt.enumerate",
        "gdpopt.loa",
        "gdpopt.gloa",
        "gdpopt.lbb",
        "gdpopt.ric",
    ]:
        with open(
            result_dir + "/" + strategy + "_" + subsolver + "_" + solver_gams + ".log",
            "w",
        ) as f:
            with redirect_stdout(f):
                results = SolverFactory(strategy).solve(
                    model,
                    tee=True,
                    # bound_tolerance=1e-2,  # default is 1e-6
                    nlp_solver=subsolver,
                    nlp_solver_args=dict(
                        solver=solver_gams,
                        add_options=[
                            "option threads=1;",
                            '$onecho > baron.opt',
                            'FirstLoc 1',
                            # # 'nlpsol 9', #9: IPOPT, 6: GAMS NLP solver, default -1: Automatic NLP solver selection and switching strategy
                            # 'optcr 1.e-2',
                            # # 'optca 1.e-2',
                            # #'maxiter 0', # force BARON to terminate after root node preprocessing
                            # # 'maxiter 1', # termination after the solution of the root node
                            # # 'maxiter 1e3',
                            # #'numloc -1', # local searches in preprocessing from randomly generated starting points until global optimality is proved or MaxTime seconds have elapsed.
                            # #'reslim 760',
                            '$offecho',
                            'GAMS_MODEL.optfile=1',
                            "option optcr=1e-2;",
                        ],
                        # keepfiles=True,
                        # tmpdir=os.path.join(result_dir, strategy, "nlp"),
                        # symbolic_solver_labels=True,
                        # logfile=result_dir + "/" + strategy + "_" + "nlp" + ".log",
                        tee=True,
                    ),
                    mip_solver=subsolver,
                    mip_solver_args=dict(add_options=["option threads=1"], tee=True),
                    minlp_solver=subsolver,
                    minlp_solver_args=dict(
                        solver=solver_gams,
                        add_options=[
                            "option threads=1;",
                            # '$onecho > baron.opt',
                            # # 'FirstLoc 1',
                            # # 'nlpsol 9', #9: IPOPT, 6: GAMS NLP solver, default -1: Automatic NLP solver selection and switching strategy
                            # 'optcr 1.e-2',
                            # # 'optca 1.e-2',
                            # #'maxiter 0', # force BARON to terminate after root node preprocessing
                            # # 'maxiter 1', # termination after the solution of the root node
                            # # 'maxiter 1e3',
                            # #'numloc -1', # local searches in preprocessing from randomly generated starting points until global optimality is proved or MaxTime seconds have elapsed.
                            # #'reslim 760',
                            # '$offecho',
                            # 'GAMS_MODEL.optfile=1',
                            "option optcr=1e-6;",
                        ],
                        tee=True,
                        # keepfiles=True,
                        # tmpdir=os.path.join(result_dir, strategy, "minlp"),
                        # symbolic_solver_labels=True,
                    ),
                    local_minlp_solver=subsolver,
                    local_minlp_solver_args=dict(
                        solver=solver_gams,
                        add_options=[
                            "option threads=1;",
                            '$onecho > baron.opt',
                            'FirstLoc 1',
                            # # 'nlpsol 9', #9: IPOPT, 6: GAMS NLP solver, default -1: Automatic NLP solver selection and switching strategy
                            # 'optcr 1.e-2',
                            # # 'optca 1.e-2',
                            # #'maxiter 0', # force BARON to terminate after root node preprocessing
                            # # 'maxiter 1', # termination after the solution of the root node
                            # # 'maxiter 1e3',
                            # #'numloc -1', # local searches in preprocessing from randomly generated starting points until global optimality is proved or MaxTime seconds have elapsed.
                            # #'reslim 760',
                            '$offecho',
                            'GAMS_MODEL.optfile=1',
                            "option optcr=1e-2;",
                        ],
                        tee=True,
                        # keepfiles=True,
                        # tmpdir=os.path.join(result_dir, strategy, "local_minlp"),
                        # symbolic_solver_labels=True,
                    ),
                    time_limit=timelimit,
                )
                # results.solver.strategy = strategy
                print(results)

    with open(
        result_dir + "/" + strategy + "_" + subsolver + "_" + solver_gams + ".json", "w"
    ) as f:
        json.dump(results.json_repn(), f)
    return None


if __name__ == "__main__":
    instance_list = [
        # "batch_processing",
        # "biofuel",  # enumeration got stuck
        # "cstr",
        # "disease_model",
        "ex1_linan_2023",
        # "gdp_col",
        # "hda",
        # "jobshop",
        # "kaibel",
        # "med_term_purchasing",
        # "methanol",
        # "mod_hens",
        # "modprodnet",
        # "positioning",
        # "small_batch",
        # "spectralog",
        # "stranded_gas",
        # "syngas"
    ]
    strategy_list = [
        # "gdp.bigm",
        # "gdp.hull",
        "gdpopt.enumerate",
        # "gdpopt.loa",
        # "gdpopt.gloa",
        # "gdpopt.ric",
        # "gdpopt.lbb",
    ]
    solver_gams_list = [
        'baron',
        # 'scip'
    ]
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timelimit = 3600

    for instance in instance_list:
        result_dir = "gdplib/" + instance + "/benchmark_result/" + current_time
        os.makedirs(result_dir, exist_ok=True)

        print("Benchmarking instance: ", instance)
        model = import_module("gdplib." + instance).build_model()

        for strategy in strategy_list:
            for solver_gams in solver_gams_list:
                if os.path.exists(
                    result_dir
                    + "/"
                    + strategy
                    + "_"
                    + "gams"
                    + "_"
                    + solver_gams
                    + ".json"
                ):
                    continue
                try:
                    benchmark(
                        model, strategy, timelimit, result_dir, "gams", solver_gams
                    )
                except:
                    pass
