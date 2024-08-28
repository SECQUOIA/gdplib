import os
import json
import time
import sys
from datetime import datetime
from importlib import import_module
from pyomo.environ import *
import sys
from contextlib import redirect_stdout


def benchmark(model, strategy, timelimit, result_dir, subsolver="scip"):
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

    Returns
    -------
    None
    """
    # We clone the model to avoid the solver starting from the optimal solution from previous solve.
    model = model.clone()

    # Direct the solver output to a file
    if strategy in ["gdp.bigm", "gdp.hull"]:
        with open(result_dir + "/" + strategy + "_" + subsolver + ".log", "w") as f:
            with redirect_stdout(f):
                transformation_start_time = time.time()
                TransformationFactory(strategy).apply_to(model)
                transformation_end_time = time.time()
                results = SolverFactory(subsolver).solve(
                    model,
                    tee=True,
                    add_options=["option reslim=3600;option threads=1;"],
                )
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
        with open(result_dir + "/" + strategy + "_" + subsolver + ".log", "w") as f:
            with redirect_stdout(f):
                results = SolverFactory(strategy).solve(
                    model,
                    tee=True,
                    nlp_solver=subsolver,
                    nlp_solver_args=dict(add_options=["option threads=1;"]),
                    mip_solver=subsolver,
                    mip_solver_args=dict(add_options=["option threads=1;"]),
                    minlp_solver=subsolver,
                    minlp_solver_args=dict(add_options=["option threads=1;"]),
                    local_minlp_solver=subsolver,
                    local_minlp_solver_args=dict(add_options=["option threads=1;"]),
                    time_limit=timelimit,
                )
                print(results)

    with open(result_dir + "/" + strategy + "_" + subsolver + ".json", "w") as f:
        json.dump(results.json_repn(), f)
    return None


if __name__ == "__main__":
    instance_list = [
        # "batch_processing",
        # "biofuel", # enumeration got stuck
        # "cstr",
        "disease_model",
        "ex1_linan_2023",
        "gdp_col",
        "hda",
        "jobshop",
        # "kaibel",
        "med_term_purchasing",
        "methanol",
        "mod_hens",
        "modprodnet",
        "positioning",
        "small_batch",
        "spectralog",
        "stranded_gas",
        "syngas",
    ]
    strategy_list = [
        "gdp.bigm",
        "gdp.hull",
        "gdpopt.enumerate",
        "gdpopt.loa",
        "gdpopt.gloa",
        "gdpopt.ric",
        "gdpopt.lbb",
    ]
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timelimit = 3600

    for instance in instance_list:
        result_dir = "gdplib/" + instance + "/benchmark_result/"
        os.makedirs(result_dir, exist_ok=True)

        print("Benchmarking instance: ", instance)
        model = import_module("gdplib." + instance).build_model()

        for strategy in strategy_list:
            if os.path.exists(result_dir + "/" + strategy + "_" + "gams" + ".json"):
                continue
            try:
                benchmark(model, strategy, timelimit, result_dir, "gams")
            except:
                pass
