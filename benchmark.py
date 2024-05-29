import os
import json
import time
import sys
from datetime import datetime
from importlib import import_module
from pyomo.environ import *


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
    model = model.clone()
    stdout = sys.stdout
    if strategy in ["gdp.bigm", "gdp.hull"]:
        transformation_start_time = time.time()
        TransformationFactory(strategy).apply_to(model)
        transformation_end_time = time.time()
        with open(
            result_dir + "/" + strategy + "_" + subsolver + ".log", "w"
        ) as sys.stdout:
            results = SolverFactory(subsolver).solve(
                model, tee=True, timelimit=timelimit
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
        with open(
            result_dir + "/" + strategy + "_" + subsolver + ".log", "w"
        ) as sys.stdout:
            results = SolverFactory(strategy).solve(
                model,
                tee=True,
                nlp_solver=subsolver,
                mip_solver=subsolver,
                minlp_solver=subsolver,
                local_minlp_solver=subsolver,
                time_limit=timelimit,
            )
            print(results)

    sys.stdout = stdout
    with open(result_dir + "/" + strategy + "_" + subsolver + ".json", "w") as f:
        json.dump(results.json_repn(), f)
    return None


if __name__ == "__main__":
    instance_list = [
        # "batch_processing",
        # "biofuel",
        # "disease_model",
        # "gdp_col",
        # "hda",
        "jobshop",
        # "kaibel",
        # "positioning",
        # "spectralog",
        # "med_term_purchasing",
        # "methanol",
        # "mod_hens",
        # "modprodnet",
        # "stranded_gas",
        # "syngas",
    ]
    strategy_list = [
        "gdp.bigm",
        "gdp.hull",
        "gdpopt.enumerate",
        "gdpopt.loa",
        "gdpopt.gloa",
        "gdpopt.ric",
    ]
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timelimit = 600

    for instance in instance_list:
        print("Benchmarking instance: " + instance)
        result_dir = "gdplib/" + instance + "/benchmark_result/"
        os.makedirs(result_dir, exist_ok=True)

        model = import_module("gdplib." + instance).build_model()

        for strategy in strategy_list:
            benchmark(model, strategy, timelimit, result_dir)
