import os
import json
import math
import time
import traceback
from datetime import datetime
from importlib import import_module
from pyomo.environ import *
from contextlib import redirect_stdout

TRANSFORMATION_STRATEGIES = ["gdp.bigm", "gdp.hull"]
GDPOPT_STRATEGIES = [
    "gdpopt.enumerate",
    "gdpopt.loa",
    "gdpopt.gloa",
    "gdpopt.lbb",
    "gdpopt.ric",
]


def _gams_solve_options(timelimit, optcr="1e-2"):
    return [f"option reslim={timelimit};option threads=1;option optcr={optcr};"]


def _gams_common_options(timelimit):
    return ["option threads=1;", f"option reslim={timelimit};"]


def _gams_firstloc_options(timelimit):
    return [
        "option threads=1;",
        '$onecho > baron.opt',
        'FirstLoc 1',
        '$offecho',
        'GAMS_MODEL.optfile=1',
        "option optcr=1e-2;",
        f"option reslim={timelimit};",
    ]


def _result_file(result_dir, strategy, subsolver, solver_gams, suffix):
    solver_label = f"_{solver_gams}" if subsolver == "gams" else ""
    return os.path.join(result_dir, f"{strategy}_{subsolver}{solver_label}.{suffix}")


def _transformation_solve_kwargs(timelimit, subsolver, solver_gams):
    kwargs = {"tee": True}
    if subsolver == "gams":
        kwargs["solver"] = solver_gams
        kwargs["add_options"] = _gams_solve_options(timelimit)
    else:
        kwargs["timelimit"] = timelimit
    return kwargs


def _gdpopt_solve_kwargs(timelimit, subsolver, solver_gams):
    kwargs = {
        "tee": True,
        "nlp_solver": subsolver,
        "mip_solver": subsolver,
        "minlp_solver": subsolver,
        "local_minlp_solver": subsolver,
        "time_limit": timelimit,
    }
    if subsolver == "gams":
        kwargs.update(
            {
                "nlp_solver_args": dict(
                    solver=solver_gams,
                    add_options=_gams_firstloc_options(timelimit),
                    tee=True,
                ),
                "mip_solver_args": dict(
                    solver=solver_gams,
                    add_options=_gams_common_options(timelimit),
                    tee=True,
                ),
                "minlp_solver_args": dict(
                    solver=solver_gams,
                    add_options=_gams_common_options(timelimit)
                    + ["option optcr=1e-6;"],
                    tee=True,
                ),
                "local_minlp_solver_args": dict(
                    solver=solver_gams,
                    add_options=_gams_firstloc_options(timelimit),
                    tee=True,
                ),
            }
        )
    return kwargs


def _gdpopt_subsolvers(subsolver, solver_gams):
    if subsolver != "gams":
        return {
            "nlp": subsolver,
            "mip": subsolver,
            "minlp": subsolver,
            "local_minlp": subsolver,
        }
    return {
        "nlp": {"interface": subsolver, "solver": solver_gams},
        "mip": {"interface": subsolver, "solver": solver_gams},
        "minlp": {"interface": subsolver, "solver": solver_gams},
        "local_minlp": {"interface": subsolver, "solver": solver_gams},
    }


def _benchmark_metadata(strategy, timelimit, subsolver, solver_gams):
    metadata = {
        "Strategy": strategy,
        "Time limit": timelimit,
        "Solver interface": subsolver,
    }
    if subsolver == "gams":
        metadata["GAMS solver"] = solver_gams
    if strategy in GDPOPT_STRATEGIES:
        metadata["Subsolvers"] = _gdpopt_subsolvers(subsolver, solver_gams)
    return metadata


def _json_safe_result(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe_result(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_result(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_result(item) for item in value]
    return value


def _write_results_json(results, path, strategy, timelimit, subsolver, solver_gams):
    payload = results.json_repn()
    payload["Benchmark"] = [
        _benchmark_metadata(strategy, timelimit, subsolver, solver_gams)
    ]
    payload = _json_safe_result(payload)
    with open(path, "w") as f:
        json.dump(payload, f, allow_nan=False)


def _write_failure_log(result_dir, instance, strategy, subsolver, solver_gams, err):
    path = os.path.join(result_dir, f"{strategy}_{subsolver}_{solver_gams}_failure.log")
    with open(path, "w") as f:
        f.write(f"Instance: {instance}\n")
        f.write(f"Strategy: {strategy}\n")
        f.write(f"Solver interface: {subsolver}\n")
        if subsolver == "gams":
            f.write(f"GAMS solver: {solver_gams}\n")
        f.write("\n")
        f.write("".join(traceback.format_exception(type(err), err, err.__traceback__)))
    return path


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
    if strategy in TRANSFORMATION_STRATEGIES:
        with open(
            _result_file(result_dir, strategy, subsolver, solver_gams, "log"), "w"
        ) as f:
            with redirect_stdout(f):
                transformation_start_time = time.time()
                TransformationFactory(strategy).apply_to(model)
                transformation_end_time = time.time()
                results = SolverFactory(subsolver).solve(
                    model,
                    **_transformation_solve_kwargs(timelimit, subsolver, solver_gams),
                    # keepfiles=True,
                    # tmpdir=os.path.join(result_dir, strategy, "nlp"),
                    # symbolic_solver_labels=True,
                )
                results.solver.strategy = strategy
                results.solver.transformation_time = (
                    transformation_end_time - transformation_start_time
                )
                print(results)
    elif strategy in GDPOPT_STRATEGIES:
        with open(
            _result_file(result_dir, strategy, subsolver, solver_gams, "log"), "w"
        ) as f:
            with redirect_stdout(f):
                results = SolverFactory(strategy).solve(
                    model, **_gdpopt_solve_kwargs(timelimit, subsolver, solver_gams)
                )
                print(results)
    else:
        raise ValueError(f"Unknown benchmark strategy: {strategy}")

    _write_results_json(
        results,
        _result_file(result_dir, strategy, subsolver, solver_gams, "json"),
        strategy,
        timelimit,
        subsolver,
        solver_gams,
    )
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
    failures = []

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
                except Exception as err:
                    failure_path = _write_failure_log(
                        result_dir, instance, strategy, "gams", solver_gams, err
                    )
                    failures.append((instance, strategy, solver_gams, failure_path))

    if failures:
        for instance, strategy, solver_gams, failure_path in failures:
            print(
                "Benchmark failed: "
                f"{instance} {strategy} gams/{solver_gams}; see {failure_path}"
            )
        raise SystemExit(f"{len(failures)} benchmark run(s) failed")
