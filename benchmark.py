from pyomo.environ import *
import os
import json
import time
import sys
from datetime import datetime
from gdplib.jobshop.jobshop import build_model
from importlib import import_module
from pyomo.util.model_size import build_model_size_report
import pandas as pd

subsolver = "scip"


def benchmark(model, strategy, timelimit, result_dir):
    """Benchmark the model using the given strategy and subsolver.

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
            results = SolverFactory("scip").solve(model, tee=True, timelimit=timelimit)
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
                nlp_solver="scip",
                mip_solver="scip",
                minlp_solver="scip",
                local_minlp_solver="scip",
                time_limit=timelimit,
            )
            print(results)

    sys.stdout = stdout
    with open(result_dir + "/" + strategy + "_" + subsolver + ".json", "w") as f:
        json.dump(results.json_repn(), f)


if __name__ == "__main__":
    instance_list = [
        # "batch_processing",
        # "biofuel",
        # "disease_model",
        # "gdp_col",
        # "hda",
        "jobshop",
        # "kaibel",
        # "logical",
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
        result_dir = "gdplib/" + instance + "/benchmark_result/" + current_time
        os.makedirs(result_dir, exist_ok=True)

        model = import_module("gdplib." + instance).build_model()
        report = build_model_size_report(model)
        report_df = pd.DataFrame(report.overall, index=[0]).T
        report_df.index.name = "Component"
        report_df.columns = ["Number"]
        # Generate the model size report (Markdown)
        report_df.to_markdown("gdplib/" + instance + "/" + "model_size_report.md")

        # Generate the model size report (JSON)
        # TODO: check if we need the json file.
        with open("gdplib/" + instance + "/" + "model_size_report.json", "w") as f:
            json.dump(report, f)

        for strategy in strategy_list:
            benchmark(model, strategy, timelimit, result_dir)
