import argparse
import csv
import json
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path

import pyomo
from pyomo.common.collections import ComponentSet
from pyomo.environ import SolverFactory, TransformationFactory

try:
    from gdplib._version import __version__ as GDPLIB_VERSION
except ImportError:
    GDPLIB_VERSION = "unknown"

TRANSFORMATION_STRATEGIES = ["gdp.bigm", "gdp.hull"]
GDPOPT_STRATEGIES = [
    "gdpopt.enumerate",
    "gdpopt.loa",
    "gdpopt.gloa",
    "gdpopt.lbb",
    "gdpopt.ric",
]
GDPOPT_CUSTOM_INIT_STRATEGIES = {"gdpopt.loa", "gdpopt.gloa", "gdpopt.ric"}
DEFAULT_STRATEGIES = TRANSFORMATION_STRATEGIES + GDPOPT_STRATEGIES
DEFAULT_LOCAL_FIRST_STRATEGIES = DEFAULT_STRATEGIES
PR58_BENCHMARK_INSTANCES = [
    "batch_processing",
    "biofuel",
    "cstr",
    "disease_model",
    "ex1_linan_2023",
    "gdp_col",
    "hda",
    "kaibel",
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
DEFAULT_TIMELIMIT = 3600
DEFAULT_SUBSOLVER = "gams"
DEFAULT_GAMS_SOLVER = "baron"
DEFAULT_SOLVER_PROFILE = "gams-local"
DEFAULT_METADATA_DIR = "benchmark_runs"
DEFAULT_GAMS_OPTCR = "1e-6"
SOLVER_PROFILES = {
    "gams-local": {
        "description": (
            "Local nonlinear GAMS profile: DICOPT for transformed/local MINLP "
            "roles, IPOPTH for NLP roles, and Gurobi for MIP roles."
        ),
        "subsolver": "gams",
        "gams_solvers": ["dicopt"],
        "gams_nlp_solver": "ipopth",
        "gams_mip_solver": "gurobi",
        "gams_minlp_solver": "dicopt",
        "gams_local_minlp_solver": "dicopt",
    },
    "gams-gurobi": {
        "description": (
            "GAMS/Gurobi profile for transformed/MIP/MINLP roles with IPOPTH "
            "for GDPopt NLP subproblems."
        ),
        "subsolver": "gams",
        "gams_solvers": ["gurobi"],
        "gams_nlp_solver": "ipopth",
        "gams_mip_solver": "gurobi",
        "gams_minlp_solver": "gurobi",
        "gams_local_minlp_solver": "gurobi",
    },
    "gams-baron": {
        "description": "Global GAMS/BARON profile matching the original PR #58 run.",
        "subsolver": "gams",
        "gams_solvers": ["baron"],
        "gams_nlp_solver": "baron",
        "gams_mip_solver": "baron",
        "gams_minlp_solver": "baron",
        "gams_local_minlp_solver": "baron",
    },
}
DEPRECATION_TRACKER_ISSUE = 55
PR58_MODEL_ISSUES = {
    "batch_processing": 61,
    "biofuel": 62,
    "cstr": 63,
    "disease_model": 64,
    "ex1_linan_2023": 65,
    "gdp_col": 66,
    "hda": 67,
    "kaibel": 68,
    "med_term_purchasing": 69,
    "methanol": 70,
    "mod_hens": 71,
    "modprodnet": 72,
    "positioning": 73,
    "small_batch": 74,
    "spectralog": 75,
    "stranded_gas": 76,
    "syngas": 77,
}


@dataclass(frozen=True)
class BenchmarkCase:
    instance: str
    strategy: str
    subsolver: str
    solver_gams: str
    timelimit: int = DEFAULT_TIMELIMIT
    solver_profile: str = DEFAULT_SOLVER_PROFILE
    gams_nlp_solver: str | None = None
    gams_mip_solver: str | None = None
    gams_minlp_solver: str | None = None
    gams_local_minlp_solver: str | None = None
    label: str | None = None


@dataclass(frozen=True)
class BenchmarkFailure:
    instance: str
    strategy: str
    subsolver: str
    solver_gams: str
    failure_log: str


class _WarningLogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        if record.levelno >= logging.WARNING:
            self.records.append(record)


class _CapturePyomoLogs:
    def __init__(self):
        self.handler = _WarningLogCapture()
        self.logger = logging.getLogger("pyomo")

    def __enter__(self):
        self.previous_handlers = list(self.logger.handlers)
        self.previous_level = self.logger.level
        self.previous_propagate = self.logger.propagate
        self.logger.handlers = [self.handler]
        self.logger.setLevel(logging.WARNING)
        self.logger.propagate = False
        return self.handler.records

    def __exit__(self, exc_type, exc, tb):
        self.logger.handlers = self.previous_handlers
        self.logger.setLevel(self.previous_level)
        self.logger.propagate = self.previous_propagate


def _gams_solve_options(timelimit, optcr=DEFAULT_GAMS_OPTCR):
    return [f"option reslim={timelimit};option threads=1;option optcr={optcr};"]


def _gams_common_options(timelimit, optcr=DEFAULT_GAMS_OPTCR):
    return [
        "option threads=1;",
        f"option reslim={timelimit};",
        f"option optcr={optcr};",
    ]


def _gams_firstloc_options(timelimit, optcr=DEFAULT_GAMS_OPTCR):
    return [
        "option threads=1;",
        "$onecho > baron.opt",
        "FirstLoc 1",
        "$offecho",
        "GAMS_MODEL.optfile=1",
        f"option optcr={optcr};",
        f"option reslim={timelimit};",
    ]


def _gams_decomposition_solver_options(gams_nlp_solver=None, gams_mip_solver=None):
    options = []
    if gams_nlp_solver:
        options.append(f"option nlp={gams_nlp_solver};")
    if gams_mip_solver:
        options.append(f"option mip={gams_mip_solver};")
    return options


def _safe_label(label):
    if not label:
        return ""
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in str(label).strip()
    )


def _result_file(result_dir, strategy, subsolver, solver_gams, suffix, label=None):
    solver_label = f"_{solver_gams}" if subsolver == "gams" else ""
    case_label = f"_{_safe_label(label)}" if label else ""
    return os.path.join(
        result_dir, f"{strategy}_{subsolver}{solver_label}{case_label}.{suffix}"
    )


def _gams_direct_solve_options(
    timelimit, solver_gams, gams_nlp_solver=None, gams_mip_solver=None
):
    options = _gams_solve_options(timelimit)
    if solver_gams.lower() in {"dicopt", "sbb"}:
        options += _gams_decomposition_solver_options(gams_nlp_solver, gams_mip_solver)
    return options


def _transformation_solve_kwargs(
    timelimit, subsolver, solver_gams, gams_nlp_solver=None, gams_mip_solver=None
):
    kwargs = {"tee": True}
    if subsolver == "gams":
        kwargs["solver"] = solver_gams
        kwargs["add_options"] = _gams_direct_solve_options(
            timelimit, solver_gams, gams_nlp_solver, gams_mip_solver
        )
    else:
        kwargs["timelimit"] = timelimit
    return kwargs


def _gams_role_solver(solver_gams, solver_name):
    return solver_name or solver_gams


def _gams_role_options(
    timelimit, solver_name, role, gams_nlp_solver=None, gams_mip_solver=None
):
    solver_name = (solver_name or "").lower()
    if solver_name == "baron" and role in {"nlp", "local_minlp"}:
        return _gams_firstloc_options(timelimit)
    options = _gams_common_options(timelimit)
    if role in {"minlp", "local_minlp"} and solver_name in {"dicopt", "sbb"}:
        options += _gams_decomposition_solver_options(gams_nlp_solver, gams_mip_solver)
    return options


def _gdpopt_solve_kwargs(
    timelimit,
    subsolver,
    solver_gams,
    gams_nlp_solver=None,
    gams_mip_solver=None,
    gams_minlp_solver=None,
    gams_local_minlp_solver=None,
):
    kwargs = {
        "tee": True,
        "nlp_solver": subsolver,
        "mip_solver": subsolver,
        "minlp_solver": subsolver,
        "local_minlp_solver": subsolver,
        "time_limit": timelimit,
    }
    if subsolver == "gams":
        nlp_solver = _gams_role_solver(solver_gams, gams_nlp_solver)
        mip_solver = _gams_role_solver(solver_gams, gams_mip_solver)
        minlp_solver = _gams_role_solver(solver_gams, gams_minlp_solver)
        local_minlp_solver = _gams_role_solver(solver_gams, gams_local_minlp_solver)
        kwargs.update(
            {
                "nlp_solver_args": dict(
                    solver=nlp_solver,
                    add_options=_gams_role_options(timelimit, nlp_solver, "nlp"),
                    tee=True,
                ),
                "mip_solver_args": dict(
                    solver=mip_solver,
                    add_options=_gams_role_options(timelimit, mip_solver, "mip"),
                    tee=True,
                ),
                "minlp_solver_args": dict(
                    solver=minlp_solver,
                    add_options=_gams_role_options(
                        timelimit,
                        minlp_solver,
                        "minlp",
                        gams_nlp_solver,
                        gams_mip_solver,
                    ),
                    tee=True,
                ),
                "local_minlp_solver_args": dict(
                    solver=local_minlp_solver,
                    add_options=_gams_role_options(
                        timelimit,
                        local_minlp_solver,
                        "local_minlp",
                        gams_nlp_solver,
                        gams_mip_solver,
                    ),
                    tee=True,
                ),
            }
        )
    return kwargs


def _gdpopt_model_initialization_kwargs(model, strategy):
    if strategy not in GDPOPT_CUSTOM_INIT_STRATEGIES:
        return {}

    initial_disjuncts = getattr(model, "gdpopt_initial_disjuncts", None)
    if not initial_disjuncts:
        return {}

    return {
        "init_algorithm": "custom_disjuncts",
        "custom_init_disjuncts": [
            ComponentSet(disjunct_set) for disjunct_set in initial_disjuncts
        ],
    }


def _gdpopt_subsolvers(
    subsolver,
    solver_gams,
    gams_nlp_solver=None,
    gams_mip_solver=None,
    gams_minlp_solver=None,
    gams_local_minlp_solver=None,
):
    if subsolver != "gams":
        return {
            "nlp": subsolver,
            "mip": subsolver,
            "minlp": subsolver,
            "local_minlp": subsolver,
        }
    nlp_solver = _gams_role_solver(solver_gams, gams_nlp_solver)
    mip_solver = _gams_role_solver(solver_gams, gams_mip_solver)
    minlp_solver = _gams_role_solver(solver_gams, gams_minlp_solver)
    local_minlp_solver = _gams_role_solver(solver_gams, gams_local_minlp_solver)
    return {
        "nlp": {"interface": subsolver, "solver": nlp_solver},
        "mip": {"interface": subsolver, "solver": mip_solver},
        "minlp": {"interface": subsolver, "solver": minlp_solver},
        "local_minlp": {"interface": subsolver, "solver": local_minlp_solver},
    }


def _benchmark_metadata(
    strategy,
    timelimit,
    subsolver,
    solver_gams,
    solver_profile=None,
    gams_nlp_solver=None,
    gams_mip_solver=None,
    gams_minlp_solver=None,
    gams_local_minlp_solver=None,
    label=None,
):
    metadata = {
        "Strategy": strategy,
        "Time limit": timelimit,
        "Solver interface": subsolver,
    }
    if label:
        metadata["Case label"] = label
    if solver_profile:
        metadata["Solver profile"] = solver_profile
    if subsolver == "gams":
        metadata["GAMS solver"] = solver_gams
        metadata["GAMS optcr"] = DEFAULT_GAMS_OPTCR
    if strategy in GDPOPT_STRATEGIES:
        metadata["Subsolvers"] = _gdpopt_subsolvers(
            subsolver,
            solver_gams,
            gams_nlp_solver,
            gams_mip_solver,
            gams_minlp_solver,
            gams_local_minlp_solver,
        )
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


def _write_results_json(
    results,
    path,
    strategy,
    timelimit,
    subsolver,
    solver_gams,
    solver_profile=None,
    gams_nlp_solver=None,
    gams_mip_solver=None,
    gams_minlp_solver=None,
    gams_local_minlp_solver=None,
    label=None,
):
    payload = results.json_repn()
    payload["Benchmark"] = [
        _benchmark_metadata(
            strategy,
            timelimit,
            subsolver,
            solver_gams,
            solver_profile,
            gams_nlp_solver,
            gams_mip_solver,
            gams_minlp_solver,
            gams_local_minlp_solver,
            label,
        )
    ]
    payload = _json_safe_result(payload)
    with open(path, "w") as f:
        json.dump(payload, f, allow_nan=False)


def _write_failure_log(
    result_dir, instance, strategy, subsolver, solver_gams, err, label=None
):
    case_label = f"_{_safe_label(label)}" if label else ""
    path = os.path.join(
        result_dir, f"{strategy}_{subsolver}_{solver_gams}{case_label}_failure.log"
    )
    with open(path, "w") as f:
        f.write(f"Instance: {instance}\n")
        f.write(f"Strategy: {strategy}\n")
        f.write(f"Solver interface: {subsolver}\n")
        if subsolver == "gams":
            f.write(f"GAMS solver: {solver_gams}\n")
        if label:
            f.write(f"Case label: {label}\n")
        f.write("\n")
        f.write("".join(traceback.format_exception(type(err), err, err.__traceback__)))
    return path


def benchmark(
    model,
    strategy,
    timelimit,
    result_dir,
    subsolver="gams",
    solver_gams="baron",
    solver_profile=None,
    gams_nlp_solver=None,
    gams_mip_solver=None,
    gams_minlp_solver=None,
    gams_local_minlp_solver=None,
    label=None,
):
    """Benchmark the model using the given strategy and subsolver.

    The result files include solver output and a JSON representation of the results.

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
    os.makedirs(result_dir, exist_ok=True)
    # We clone the model to avoid the solver starting from a previous solve.
    model = model.clone()

    if strategy in TRANSFORMATION_STRATEGIES:
        with open(
            _result_file(result_dir, strategy, subsolver, solver_gams, "log", label),
            "w",
        ) as f:
            with redirect_stdout(f):
                transformation_start_time = time.time()
                TransformationFactory(strategy).apply_to(model)
                transformation_end_time = time.time()
                results = SolverFactory(subsolver).solve(
                    model,
                    **_transformation_solve_kwargs(
                        timelimit,
                        subsolver,
                        solver_gams,
                        gams_nlp_solver,
                        gams_mip_solver,
                    ),
                )
                results.solver.strategy = strategy
                results.solver.transformation_time = (
                    transformation_end_time - transformation_start_time
                )
                print(results)
    elif strategy in GDPOPT_STRATEGIES:
        with open(
            _result_file(result_dir, strategy, subsolver, solver_gams, "log", label),
            "w",
        ) as f:
            with redirect_stdout(f):
                solve_kwargs = _gdpopt_solve_kwargs(
                    timelimit,
                    subsolver,
                    solver_gams,
                    gams_nlp_solver,
                    gams_mip_solver,
                    gams_minlp_solver,
                    gams_local_minlp_solver,
                )
                solve_kwargs.update(
                    _gdpopt_model_initialization_kwargs(model, strategy)
                )
                results = SolverFactory(strategy).solve(model, **solve_kwargs)
                print(results)
    else:
        raise ValueError(f"Unknown benchmark strategy: {strategy}")

    _write_results_json(
        results,
        _result_file(result_dir, strategy, subsolver, solver_gams, "json", label),
        strategy,
        timelimit,
        subsolver,
        solver_gams,
        solver_profile,
        gams_nlp_solver,
        gams_mip_solver,
        gams_minlp_solver,
        gams_local_minlp_solver,
        label,
    )
    return None


def _profile_value(solver_profile, key):
    return SOLVER_PROFILES[solver_profile][key]


def _case_from_values(
    instance,
    strategy,
    timelimit,
    solver_profile,
    subsolver=None,
    solver_gams=None,
    gams_nlp_solver=None,
    gams_mip_solver=None,
    gams_minlp_solver=None,
    gams_local_minlp_solver=None,
    label=None,
):
    if not instance:
        raise ValueError("Benchmark case is missing required field: instance")
    if not strategy:
        raise ValueError("Benchmark case is missing required field: strategy")
    if solver_profile not in SOLVER_PROFILES:
        raise ValueError(f"Unknown solver profile: {solver_profile}")
    if strategy not in DEFAULT_STRATEGIES:
        raise ValueError(f"Unknown benchmark strategy: {strategy}")
    subsolver = subsolver or _profile_value(solver_profile, "subsolver")
    solver_gams = solver_gams or _profile_value(solver_profile, "gams_solvers")[0]
    return BenchmarkCase(
        instance=instance,
        strategy=strategy,
        subsolver=subsolver,
        solver_gams=solver_gams,
        timelimit=int(timelimit),
        solver_profile=solver_profile,
        gams_nlp_solver=(
            gams_nlp_solver or _profile_value(solver_profile, "gams_nlp_solver")
        ),
        gams_mip_solver=(
            gams_mip_solver or _profile_value(solver_profile, "gams_mip_solver")
        ),
        gams_minlp_solver=(
            gams_minlp_solver or _profile_value(solver_profile, "gams_minlp_solver")
        ),
        gams_local_minlp_solver=(
            gams_local_minlp_solver
            or _profile_value(solver_profile, "gams_local_minlp_solver")
        ),
        label=label,
    )


def benchmark_cases(
    instances,
    strategies,
    subsolver,
    solver_gams_list,
    timelimit=DEFAULT_TIMELIMIT,
    solver_profile=DEFAULT_SOLVER_PROFILE,
    gams_nlp_solver=None,
    gams_mip_solver=None,
    gams_minlp_solver=None,
    gams_local_minlp_solver=None,
):
    return [
        _case_from_values(
            instance=instance,
            strategy=strategy,
            timelimit=timelimit,
            solver_profile=solver_profile,
            subsolver=subsolver,
            solver_gams=solver_gams,
            gams_nlp_solver=gams_nlp_solver,
            gams_mip_solver=gams_mip_solver,
            gams_minlp_solver=gams_minlp_solver,
            gams_local_minlp_solver=gams_local_minlp_solver,
        )
        for instance in instances
        for strategy in strategies
        for solver_gams in solver_gams_list
    ]


def _string_or_none(value):
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _row_value(row, *names):
    for name in names:
        if name in row:
            value = _string_or_none(row[name])
            if value is not None:
                return value
    return None


def _case_from_row(row, defaults):
    solver_profile = _row_value(row, "solver_profile") or defaults.solver_profile
    use_global_defaults = solver_profile == defaults.solver_profile
    timelimit = _row_value(row, "timelimit", "time_limit", "time_limit_s")
    return _case_from_values(
        instance=_row_value(row, "instance", "model"),
        strategy=_row_value(row, "strategy", "method"),
        timelimit=timelimit or defaults.timelimit,
        solver_profile=solver_profile,
        subsolver=_row_value(row, "subsolver", "solver_interface")
        or (defaults.subsolver if use_global_defaults else None),
        solver_gams=_row_value(row, "gams_solver", "solver_gams")
        or (defaults.gams_solvers[0] if use_global_defaults else None),
        gams_nlp_solver=_row_value(row, "gams_nlp_solver")
        or (defaults.gams_nlp_solver if use_global_defaults else None),
        gams_mip_solver=_row_value(row, "gams_mip_solver")
        or (defaults.gams_mip_solver if use_global_defaults else None),
        gams_minlp_solver=_row_value(row, "gams_minlp_solver")
        or (defaults.gams_minlp_solver if use_global_defaults else None),
        gams_local_minlp_solver=_row_value(row, "gams_local_minlp_solver")
        or (defaults.gams_local_minlp_solver if use_global_defaults else None),
        label=_row_value(row, "label", "case_label"),
    )


def read_benchmark_cases_file(path, defaults):
    path = Path(path)
    if path.suffix.lower() == ".json":
        with open(path, "r") as f:
            payload = json.load(f)
        rows = payload["cases"] if isinstance(payload, dict) else payload
    else:
        with open(path, "r", newline="") as f:
            rows = list(csv.DictReader(f))
    cases = [_case_from_row(row, defaults) for row in rows]
    if not cases:
        raise ValueError(f"No benchmark cases found in {path}")
    return cases


def selected_cases(args):
    if getattr(args, "cases_file", None):
        return read_benchmark_cases_file(args.cases_file, args)
    return benchmark_cases(
        args.instances,
        args.strategies,
        args.subsolver,
        args.gams_solvers,
        args.timelimit,
        args.solver_profile,
        args.gams_nlp_solver,
        args.gams_mip_solver,
        args.gams_minlp_solver,
        args.gams_local_minlp_solver,
    )


def _case_instances(cases):
    return list(dict.fromkeys(case.instance for case in cases))


def _run_id():
    return "gdplib_benchmark_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _result_dir(instance, run_id):
    return Path("gdplib") / instance / "benchmark_result" / run_id


def _metadata_dir(base_dir, run_id):
    return Path(base_dir) / run_id


def _result_json_path(case, run_id):
    return Path(
        _result_file(
            _result_dir(case.instance, run_id),
            case.strategy,
            case.subsolver,
            case.solver_gams,
            "json",
            case.label,
        )
    )


def _check_solver_available(name):
    try:
        return SolverFactory(name).available(False), None
    except Exception as err:
        return False, err


def _check_transformation_available(name):
    try:
        return TransformationFactory(name) is not None, None
    except Exception as err:
        return False, err


def _resolve_solver_profile(args):
    if not hasattr(args, "solver_profile"):
        return args
    profile = SOLVER_PROFILES[args.solver_profile]
    if args.subsolver is None:
        args.subsolver = profile["subsolver"]
    if args.gams_solvers is None:
        args.gams_solvers = list(profile["gams_solvers"])
    for role in ("nlp", "mip", "minlp", "local_minlp"):
        attr = f"gams_{role}_solver"
        if getattr(args, attr) is None:
            setattr(args, attr, profile[attr])
    return args


def _gams_role_solver_labels(args):
    return {
        "nlp": args.gams_nlp_solver,
        "mip": args.gams_mip_solver,
        "minlp": args.gams_minlp_solver,
        "local_minlp": args.gams_local_minlp_solver,
    }


def run_preflight(args, stream=None):  # noqa: C901
    stream = stream or sys.stdout
    cases = selected_cases(args)
    instances = _case_instances(cases)
    strategies = sorted(set(case.strategy for case in cases))
    subsolvers = sorted(set(case.subsolver for case in cases))
    gams_cases = [case for case in cases if case.subsolver == "gams"]
    ok = True

    print("Benchmark plan", file=stream)
    if getattr(args, "cases_file", None):
        print(f"  cases file: {args.cases_file}", file=stream)
    print(f"  instances: {len(instances)}", file=stream)
    print(f"  strategies: {len(strategies)}", file=stream)
    print(f"  cases: {len(cases)}", file=stream)
    timelimits = sorted(set(case.timelimit for case in cases))
    timelimit_label = (
        str(timelimits[0]) if len(timelimits) == 1 else ", ".join(map(str, timelimits))
    )
    print(f"  timelimit: {timelimit_label}", file=stream)
    if hasattr(args, "solver_profile"):
        profiles = sorted(set(case.solver_profile for case in cases))
        print(f"  solver profiles: {', '.join(profiles)}", file=stream)
    print(f"  subsolvers: {', '.join(subsolvers)}", file=stream)
    if gams_cases:
        gams_solvers = sorted(set(case.solver_gams for case in gams_cases))
        print(f"  GAMS solvers: {', '.join(gams_solvers)}", file=stream)
        if any(case.strategy in GDPOPT_STRATEGIES for case in gams_cases):
            role_labels = {
                "nlp": sorted(set(case.gams_nlp_solver for case in gams_cases)),
                "mip": sorted(set(case.gams_mip_solver for case in gams_cases)),
                "minlp": sorted(set(case.gams_minlp_solver for case in gams_cases)),
                "local_minlp": sorted(
                    set(case.gams_local_minlp_solver for case in gams_cases)
                ),
            }
            print(
                "  GDPopt GAMS role solvers: "
                + ", ".join(
                    f"{role}={'/'.join(filter(None, solvers))}"
                    for role, solvers in role_labels.items()
                ),
                file=stream,
            )

    if args.check_solvers:
        print("\nSolver and strategy checks", file=stream)
        for subsolver in subsolvers:
            available, err = _check_solver_available(subsolver)
            if available:
                print(f"  OK solver interface: {subsolver}", file=stream)
            else:
                ok = False
                detail = f" ({err})" if err is not None else ""
                print(f"  FAIL solver interface: {subsolver}{detail}", file=stream)
        if gams_cases:
            gams_executable = shutil.which("gams")
            if gams_executable:
                print(f"  OK gams executable: {gams_executable}", file=stream)
            else:
                ok = False
                print("  FAIL gams executable: not found on PATH", file=stream)
            print(
                "  NOTE GAMS solver names are passed through to GAMS at solve time",
                file=stream,
            )
        for strategy in strategies:
            if strategy in TRANSFORMATION_STRATEGIES:
                available, err = _check_transformation_available(strategy)
                label = "transformation"
            else:
                available, err = _check_solver_available(strategy)
                label = "GDPopt strategy"
            if available:
                print(f"  OK {label}: {strategy}", file=stream)
            else:
                ok = False
                detail = f" ({err})" if err is not None else ""
                print(f"  FAIL {label}: {strategy}{detail}", file=stream)

    if args.build_models:
        print("\nModel construction checks", file=stream)
        for instance in instances:
            try:
                module = import_module("gdplib." + instance)
                build_model = getattr(module, "build_model")
                build_model()
            except Exception as err:
                ok = False
                print(
                    f"  FAIL gdplib.{instance}.build_model(): "
                    f"{type(err).__name__}: {err}",
                    file=stream,
                )
            else:
                print(f"  OK gdplib.{instance}.build_model()", file=stream)

    if ok:
        print("\nPreflight passed", file=stream)
    else:
        print("\nPreflight failed", file=stream)
    return ok


def _command_output(command):
    try:
        completed = subprocess.run(command, capture_output=True, check=False, text=True)
    except OSError as err:
        return f"{type(err).__name__}: {err}"
    output = completed.stdout.strip()
    if completed.stderr.strip():
        output = output + ("\n" if output else "") + completed.stderr.strip()
    return output


def collect_environment():
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pyomo": pyomo.__version__,
        "gdplib": GDPLIB_VERSION,
        "gams_executable": shutil.which("gams"),
        "gams_available": SolverFactory("gams").available(False),
        "git_head": _command_output(["git", "rev-parse", "HEAD"]),
        "git_status": _command_output(["git", "status", "--short"]),
    }


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _strict_json_load(path):
    def reject_constant(value):
        raise ValueError(f"non-strict JSON constant: {value}")

    with open(path, "r") as f:
        return json.load(f, parse_constant=reject_constant)


def _normalize_warning_message(message):
    return " ".join(str(message).split())


def _is_deprecation_warning(category, message):
    text = f"{category} {message}".lower()
    return (
        "deprecated" in text
        or "deprecation" in text
        or "deprecate" in text
        or "implicitly casting" in text
        or "indicator_var' value" in text
        or 'indicator_var" value' in text
    )


def _warning_row(context, source, category, message):
    normalized = _normalize_warning_message(message)
    instance = context["instance"]
    is_deprecation = _is_deprecation_warning(category, normalized)
    return {
        "instance": instance,
        "model_issue": PR58_MODEL_ISSUES.get(instance),
        "phase": context["phase"],
        "strategy": context.get("strategy"),
        "subsolver": context.get("subsolver"),
        "solver_gams": context.get("solver_gams"),
        "source": source,
        "category": category,
        "is_deprecation_candidate": is_deprecation,
        "deprecation_tracker_issue": (
            DEPRECATION_TRACKER_ISSUE if is_deprecation else None
        ),
        "message": normalized,
    }


def _capture_warning_rows(context, func):
    with _CapturePyomoLogs() as log_records:
        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always")
            error = None
            try:
                func()
            except Exception as err:
                error = err

    rows = []
    for record in log_records:
        rows.append(
            _warning_row(
                context,
                f"{record.pathname}:{record.lineno}",
                record.name,
                record.getMessage(),
            )
        )
    for record in warning_records:
        rows.append(
            _warning_row(
                context,
                f"{record.filename}:{record.lineno}",
                record.category.__name__,
                record.message,
            )
        )
    return rows, error


def summarize_warning_rows(rows):
    summary = {}
    for row in rows:
        key = (
            row["instance"],
            row["model_issue"],
            row["phase"],
            row["strategy"],
            row["subsolver"],
            row["solver_gams"],
            row["category"],
            row["is_deprecation_candidate"],
            row["deprecation_tracker_issue"],
            row["message"],
        )
        if key not in summary:
            summary[key] = {
                "instance": row["instance"],
                "model_issue": row["model_issue"],
                "phase": row["phase"],
                "strategy": row["strategy"],
                "subsolver": row["subsolver"],
                "solver_gams": row["solver_gams"],
                "category": row["category"],
                "is_deprecation_candidate": row["is_deprecation_candidate"],
                "deprecation_tracker_issue": row["deprecation_tracker_issue"],
                "count": 0,
                "sources": set(),
                "message": row["message"],
            }
        summary[key]["count"] += 1
        summary[key]["sources"].add(row["source"])
    rows = list(summary.values())
    for row in rows:
        row["sources"] = "; ".join(sorted(row["sources"]))
    return sorted(
        rows,
        key=lambda row: (
            row["instance"],
            row["phase"],
            row["strategy"] or "",
            not row["is_deprecation_candidate"],
            row["category"],
            row["message"],
        ),
    )


def _write_warning_markdown(path, summary_rows, deprecations_only=False):
    filtered_rows = [
        row
        for row in summary_rows
        if not deprecations_only or row["is_deprecation_candidate"]
    ]
    title = (
        "Deprecation Warning Candidates"
        if deprecations_only
        else "Benchmark Warning Summary"
    )
    with open(path, "w") as f:
        f.write(f"# {title}\n\n")
        if not filtered_rows:
            f.write("No warnings captured.\n")
            return
        f.write("| Instance | Model issue | Phase | Strategy | Category | Count | ")
        f.write("Message |\n")
        f.write("|---|---:|---|---|---|---:|---|\n")
        for row in filtered_rows:
            message = row["message"].replace("|", "\\|")
            strategy = row["strategy"] or ""
            model_issue = row["model_issue"] or ""
            f.write(
                f"| {row['instance']} | {model_issue} | {row['phase']} | "
                f"{strategy} | {row['category']} | {row['count']} | "
                f"{message} |\n"
            )


def write_warning_reports(metadata_path, warning_rows, errors):
    metadata_path.mkdir(parents=True, exist_ok=True)
    event_fields = [
        "instance",
        "model_issue",
        "phase",
        "strategy",
        "subsolver",
        "solver_gams",
        "source",
        "category",
        "is_deprecation_candidate",
        "deprecation_tracker_issue",
        "message",
    ]
    _write_csv(metadata_path / "warning_events.csv", warning_rows, event_fields)
    with open(metadata_path / "warning_events.json", "w") as f:
        json.dump(warning_rows, f, indent=2, allow_nan=False)

    summary_rows = summarize_warning_rows(warning_rows)
    summary_fields = [
        "instance",
        "model_issue",
        "phase",
        "strategy",
        "subsolver",
        "solver_gams",
        "category",
        "is_deprecation_candidate",
        "deprecation_tracker_issue",
        "count",
        "sources",
        "message",
    ]
    _write_csv(metadata_path / "warning_summary.csv", summary_rows, summary_fields)
    with open(metadata_path / "warning_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2, allow_nan=False)
    _write_warning_markdown(metadata_path / "warning_summary.md", summary_rows)
    _write_warning_markdown(
        metadata_path / "issue_55_deprecation_candidates.md",
        summary_rows,
        deprecations_only=True,
    )

    error_rows = [
        {
            "instance": error["instance"],
            "phase": error["phase"],
            "strategy": error.get("strategy"),
            "subsolver": error.get("subsolver"),
            "solver_gams": error.get("solver_gams"),
            "error_type": type(error["error"]).__name__,
            "message": str(error["error"]),
            "failure_log": error.get("failure_log"),
        }
        for error in errors
    ]
    _write_csv(
        metadata_path / "warning_capture_errors.csv",
        error_rows,
        [
            "instance",
            "phase",
            "strategy",
            "subsolver",
            "solver_gams",
            "error_type",
            "message",
            "failure_log",
        ],
    )
    with open(metadata_path / "warning_capture_errors.json", "w") as f:
        json.dump(error_rows, f, indent=2, allow_nan=False)
    return summary_rows, error_rows


def collect_result_rows(run_id, instances):
    rows = []
    for instance in instances:
        result_dir = _result_dir(instance, run_id)
        for path in sorted(result_dir.glob("*.json")):
            data = _strict_json_load(path)
            problem = (data.get("Problem") or [{}])[0]
            solver = (data.get("Solver") or [{}])[0]
            metadata = (data.get("Benchmark") or [{}])[0]
            sense = problem.get("Sense")
            lower_bound = problem.get("Lower bound")
            upper_bound = problem.get("Upper bound")
            objective = None
            if solver.get("Termination condition") != "infeasible":
                if sense == "minimize":
                    objective = upper_bound
                elif sense == "maximize":
                    objective = lower_bound
            rows.append(
                {
                    "instance": instance,
                    "strategy": metadata.get("Strategy"),
                    "case_label": metadata.get("Case label"),
                    "solver_profile": metadata.get("Solver profile"),
                    "solver_interface": metadata.get("Solver interface"),
                    "gams_solver": metadata.get("GAMS solver"),
                    "time_limit_s": metadata.get("Time limit"),
                    "problem_name": problem.get("Name"),
                    "sense": sense,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "objective_value": objective,
                    "solver_name": solver.get("Name"),
                    "termination_condition": solver.get("Termination condition"),
                    "status": solver.get("Status"),
                    "user_time_s": solver.get("User time"),
                    "json_path": str(path),
                    "log_path": str(path.with_suffix(".log")),
                }
            )
    return rows


def _write_run_metadata(metadata_path, run_id, args, failures, cases=None):
    metadata_path.mkdir(parents=True, exist_ok=True)
    environment = collect_environment()
    cases = cases or selected_cases(args)
    instances = _case_instances(cases)
    strategies = sorted(set(case.strategy for case in cases))
    with open(metadata_path / "environment.json", "w") as f:
        json.dump(environment, f, indent=2, allow_nan=False)
    with open(metadata_path / "run_configuration.json", "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "cases_file": getattr(args, "cases_file", None),
                "instances": instances,
                "strategies": strategies,
                "timelimit": args.timelimit,
                "solver_profile": getattr(args, "solver_profile", None),
                "subsolver": args.subsolver,
                "gams_solvers": args.gams_solvers,
                "gams_nlp_solver": getattr(args, "gams_nlp_solver", None),
                "gams_mip_solver": getattr(args, "gams_mip_solver", None),
                "gams_minlp_solver": getattr(args, "gams_minlp_solver", None),
                "gams_local_minlp_solver": getattr(
                    args, "gams_local_minlp_solver", None
                ),
                "skip_existing": getattr(args, "skip_existing", None),
                "cases": [asdict(case) for case in cases],
            },
            f,
            indent=2,
            allow_nan=False,
        )
    failure_rows = [asdict(failure) for failure in failures]
    _write_csv(
        metadata_path / "failures.csv",
        failure_rows,
        ["instance", "strategy", "subsolver", "solver_gams", "failure_log"],
    )
    with open(metadata_path / "failures.json", "w") as f:
        json.dump(failure_rows, f, indent=2, allow_nan=False)


def _write_result_reports(metadata_path, run_id, instances):
    rows = collect_result_rows(run_id, instances)
    fieldnames = [
        "instance",
        "strategy",
        "case_label",
        "solver_profile",
        "solver_interface",
        "gams_solver",
        "time_limit_s",
        "problem_name",
        "sense",
        "lower_bound",
        "upper_bound",
        "objective_value",
        "solver_name",
        "termination_condition",
        "status",
        "user_time_s",
        "json_path",
        "log_path",
    ]
    _write_csv(metadata_path / "results_flat.csv", rows, fieldnames)
    with open(metadata_path / "results_flat.json", "w") as f:
        json.dump(rows, f, indent=2, allow_nan=False)
    return rows


def _generate_summary(run_id, instances):
    folders = [str(_result_dir(instance, run_id)) for instance in instances]
    if not any(any(Path(folder).glob("*.json")) for folder in folders):
        return False
    from gdplib.benchmark_summary import generate_benchmark_summary

    generate_benchmark_summary(folders)
    return True


def run_benchmark_campaign(args, stream=None):  # noqa: C901
    stream = stream or sys.stdout
    run_id = args.run_id or _run_id()
    metadata_path = _metadata_dir(args.metadata_dir, run_id)
    cases = selected_cases(args)
    instances = _case_instances(cases)

    if not args.skip_preflight:
        if not run_preflight(args, stream=stream):
            return 2
        if args.dry_run:
            print("\nDry run complete; no benchmark solves were launched", file=stream)
            return 0
    elif args.dry_run:
        print(f"Dry run complete; {len(cases)} cases selected", file=stream)
        return 0

    metadata_path.mkdir(parents=True, exist_ok=True)
    failures = []
    cases_by_instance = {
        instance: [case for case in cases if case.instance == instance]
        for instance in instances
    }

    for instance in instances:
        result_dir = _result_dir(instance, run_id)
        result_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nBenchmarking instance: {instance}", file=stream, flush=True)
        try:
            model = import_module("gdplib." + instance).build_model()
        except Exception as err:
            failure_path = _write_failure_log(
                result_dir,
                instance,
                "build_model",
                cases_by_instance[instance][0].subsolver,
                cases_by_instance[instance][0].solver_gams,
                err,
            )
            failures.append(
                BenchmarkFailure(
                    instance,
                    "build_model",
                    cases_by_instance[instance][0].subsolver,
                    cases_by_instance[instance][0].solver_gams,
                    failure_path,
                )
            )
            if args.fail_fast:
                break
            continue

        for case in cases_by_instance[instance]:
            if args.skip_existing and _result_json_path(case, run_id).exists():
                print(
                    f"Skipping existing result: {case.instance} {case.strategy} "
                    f"{case.subsolver}/{case.solver_gams}",
                    file=stream,
                    flush=True,
                )
                continue
            print(
                f"Running: {case.instance} {case.strategy} "
                f"{case.subsolver}/{case.solver_gams} "
                f"limit={case.timelimit}s profile={case.solver_profile}",
                file=stream,
                flush=True,
            )
            try:
                benchmark(
                    model,
                    case.strategy,
                    case.timelimit,
                    result_dir,
                    case.subsolver,
                    case.solver_gams,
                    case.solver_profile,
                    case.gams_nlp_solver,
                    case.gams_mip_solver,
                    case.gams_minlp_solver,
                    case.gams_local_minlp_solver,
                    case.label,
                )
            except Exception as err:
                failure_path = _write_failure_log(
                    result_dir,
                    case.instance,
                    case.strategy,
                    case.subsolver,
                    case.solver_gams,
                    err,
                    case.label,
                )
                failures.append(
                    BenchmarkFailure(
                        case.instance,
                        case.strategy,
                        case.subsolver,
                        case.solver_gams,
                        failure_path,
                    )
                )
                if args.fail_fast:
                    break
        if args.fail_fast and failures:
            break

    _write_run_metadata(metadata_path, run_id, args, failures, cases)
    rows = _write_result_reports(metadata_path, run_id, instances)
    if args.summary:
        summary_generated = _generate_summary(run_id, instances)
    else:
        summary_generated = False
    print(f"\nRun id: {run_id}", file=stream)
    print(f"Result rows: {len(rows)}", file=stream)
    print(f"Failures: {len(failures)}", file=stream)
    print(f"Run metadata: {metadata_path}", file=stream)
    if summary_generated:
        print("Benchmark summary: benchmark_summary", file=stream)
    if failures:
        return 1
    return 0


def _write_warning_error(result_dir, context, error):
    failure_log = _write_failure_log(
        result_dir,
        context["instance"],
        context["phase"] if context["strategy"] is None else context["strategy"],
        context["subsolver"],
        context["solver_gams"],
        error,
    )
    return {
        "instance": context["instance"],
        "phase": context["phase"],
        "strategy": context["strategy"],
        "subsolver": context["subsolver"],
        "solver_gams": context["solver_gams"],
        "error": error,
        "failure_log": failure_log,
    }


def run_warning_capture(args, stream=None):  # noqa: C901
    stream = stream or sys.stdout
    run_id = args.run_id or _run_id()
    metadata_path = _metadata_dir(args.metadata_dir, run_id)
    cases = selected_cases(args)
    instances = _case_instances(cases)
    cases_by_instance = {
        instance: [case for case in cases if case.instance == instance]
        for instance in instances
    }
    warning_rows = []
    errors = []

    print("Warning capture plan", file=stream)
    print(f"  run id: {run_id}", file=stream)
    print(f"  mode: {args.mode}", file=stream)
    if getattr(args, "cases_file", None):
        print(f"  cases file: {args.cases_file}", file=stream)
    print(f"  instances: {len(instances)}", file=stream)
    if args.mode != "build":
        print(f"  cases: {len(cases)}", file=stream)
    if args.mode == "solve":
        timelimits = sorted(set(case.timelimit for case in cases))
        timelimit_label = (
            str(timelimits[0])
            if len(timelimits) == 1
            else ", ".join(map(str, timelimits))
        )
        print(f"  timelimit: {timelimit_label}", file=stream)

    for instance in instances:
        result_dir = _result_dir(instance, run_id)
        result_dir.mkdir(parents=True, exist_ok=True)
        module = import_module("gdplib." + instance)
        build_model = getattr(module, "build_model")
        model_holder = {}
        build_context = {
            "instance": instance,
            "phase": "build",
            "strategy": None,
            "subsolver": cases_by_instance[instance][0].subsolver,
            "solver_gams": cases_by_instance[instance][0].solver_gams,
        }

        print(f"\nCapturing warnings: {instance} build", file=stream, flush=True)

        def build():
            model_holder["model"] = build_model()

        rows, error = _capture_warning_rows(build_context, build)
        warning_rows.extend(rows)
        if error is not None:
            errors.append(_write_warning_error(result_dir, build_context, error))
            if args.fail_fast:
                break
            continue

        model = model_holder["model"]
        if args.mode == "build":
            continue

        if args.mode == "transform":
            transform_strategies = sorted(
                set(
                    case.strategy
                    for case in cases_by_instance[instance]
                    if case.strategy in TRANSFORMATION_STRATEGIES
                )
            )
            for strategy in transform_strategies:
                representative_case = cases_by_instance[instance][0]
                context = {
                    "instance": instance,
                    "phase": "transform",
                    "strategy": strategy,
                    "subsolver": representative_case.subsolver,
                    "solver_gams": representative_case.solver_gams,
                }
                print(
                    f"Capturing warnings: {instance} {strategy} transform",
                    file=stream,
                    flush=True,
                )

                def transform():
                    TransformationFactory(strategy).apply_to(model.clone())

                rows, error = _capture_warning_rows(context, transform)
                warning_rows.extend(rows)
                if error is not None:
                    errors.append(_write_warning_error(result_dir, context, error))
                    if args.fail_fast:
                        break
            if args.fail_fast and errors:
                break
            continue

        for case in cases_by_instance[instance]:
            context = {
                "instance": instance,
                "phase": "solve",
                "strategy": case.strategy,
                "subsolver": case.subsolver,
                "solver_gams": case.solver_gams,
            }
            print(
                f"Capturing warnings: {case.instance} {case.strategy} "
                f"{case.subsolver}/{case.solver_gams}",
                file=stream,
                flush=True,
            )

            def solve():
                benchmark(
                    model,
                    case.strategy,
                    case.timelimit,
                    result_dir,
                    case.subsolver,
                    case.solver_gams,
                    case.solver_profile,
                    case.gams_nlp_solver,
                    case.gams_mip_solver,
                    case.gams_minlp_solver,
                    case.gams_local_minlp_solver,
                    case.label,
                )

            rows, error = _capture_warning_rows(context, solve)
            warning_rows.extend(rows)
            if error is not None:
                errors.append(_write_warning_error(result_dir, context, error))
                if args.fail_fast:
                    break
        if args.fail_fast and errors:
            break

    _write_run_metadata(metadata_path, run_id, args, [], cases)
    summary_rows, error_rows = write_warning_reports(
        metadata_path, warning_rows, errors
    )
    deprecation_count = sum(
        row["count"] for row in summary_rows if row["is_deprecation_candidate"]
    )
    print(f"\nRun id: {run_id}", file=stream)
    print(f"Warning events: {len(warning_rows)}", file=stream)
    print(f"Unique warning rows: {len(summary_rows)}", file=stream)
    print(f"Deprecation candidate events: {deprecation_count}", file=stream)
    print(f"Capture errors: {len(error_rows)}", file=stream)
    print(f"Warning metadata: {metadata_path}", file=stream)
    if error_rows:
        return 1
    return 0


def _add_campaign_options(parser, default_timelimit=DEFAULT_TIMELIMIT):
    parser.add_argument(
        "--instances",
        nargs="+",
        default=PR58_BENCHMARK_INSTANCES,
        help="Model package names to benchmark.",
    )
    parser.add_argument(
        "--cases-file",
        default=None,
        help=(
            "CSV or JSON case file. Each row may set instance, strategy, "
            "timelimit, solver_profile, subsolver, gams_solver, GAMS role "
            "solvers, and label."
        ),
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=DEFAULT_STRATEGIES,
        default=DEFAULT_STRATEGIES,
        help="Benchmark strategies to run.",
    )
    parser.add_argument("--timelimit", type=int, default=default_timelimit)
    parser.add_argument(
        "--solver-profile",
        choices=sorted(SOLVER_PROFILES),
        default=DEFAULT_SOLVER_PROFILE,
        help="Named solver configuration for the benchmark campaign.",
    )
    parser.add_argument(
        "--subsolver",
        default=None,
        help="Override the profile's Pyomo solver interface.",
    )
    parser.add_argument(
        "--gams-solvers",
        nargs="+",
        default=None,
        help=(
            "Override the profile's GAMS solver names for direct transformed "
            "solves when --subsolver=gams."
        ),
    )
    parser.add_argument(
        "--gams-nlp-solver",
        default=None,
        help="Override the profile's GAMS solver for GDPopt NLP subproblems.",
    )
    parser.add_argument(
        "--gams-mip-solver",
        default=None,
        help="Override the profile's GAMS solver for GDPopt MIP masters.",
    )
    parser.add_argument(
        "--gams-minlp-solver",
        default=None,
        help="Override the profile's GAMS solver for GDPopt MINLP subproblems.",
    )
    parser.add_argument(
        "--gams-local-minlp-solver",
        default=None,
        help="Override the profile's GAMS solver for GDPopt local MINLP roles.",
    )
    parser.add_argument(
        "--build-models",
        dest="build_models",
        action="store_true",
        default=True,
        help="Construct every selected model during preflight.",
    )
    parser.add_argument(
        "--no-build-models",
        dest="build_models",
        action="store_false",
        help="Do not construct models during preflight.",
    )
    parser.add_argument(
        "--check-solvers",
        dest="check_solvers",
        action="store_true",
        default=True,
        help="Check solver interfaces and strategy plugins during preflight.",
    )
    parser.add_argument(
        "--no-check-solvers",
        dest="check_solvers",
        action="store_false",
        help="Do not check solver availability during preflight.",
    )


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="gdplib-benchmark",
        description="Run or preflight GDPlib benchmark campaigns.",
    )
    subparsers = parser.add_subparsers(dest="command")

    preflight = subparsers.add_parser(
        "preflight",
        help="Validate the selected benchmark campaign without solving models.",
    )
    _add_campaign_options(preflight)
    preflight.set_defaults(func=lambda args: 0 if run_preflight(args) else 1)

    run = subparsers.add_parser("run", help="Run the selected benchmark campaign.")
    _add_campaign_options(run)
    run.add_argument("--run-id", default=None)
    run.add_argument("--metadata-dir", default=DEFAULT_METADATA_DIR)
    run.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Skip cases whose JSON result already exists for the run id.",
    )
    run.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Rerun cases even when JSON results already exist.",
    )
    run.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Start solves without running preflight checks first.",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preflight and print the selected campaign without solving.",
    )
    run.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the campaign after the first build or solve failure.",
    )
    run.add_argument(
        "--summary",
        dest="summary",
        action="store_true",
        default=True,
        help="Generate benchmark_summary outputs after the run.",
    )
    run.add_argument(
        "--no-summary",
        dest="summary",
        action="store_false",
        help="Do not generate benchmark_summary outputs after the run.",
    )
    run.set_defaults(func=run_benchmark_campaign)

    warning_cmd = subparsers.add_parser(
        "warnings",
        help="Capture Pyomo/Python warnings for a fast benchmark smoke campaign.",
    )
    _add_campaign_options(warning_cmd, default_timelimit=1)
    warning_cmd.add_argument("--run-id", default=None)
    warning_cmd.add_argument("--metadata-dir", default=DEFAULT_METADATA_DIR)
    warning_cmd.add_argument(
        "--mode",
        choices=["build", "transform", "solve"],
        default="build",
        help=(
            "Capture warnings during model construction, transformation-only "
            "smokes, or short solver-backed solves."
        ),
    )
    warning_cmd.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop warning capture after the first build, transform, or solve error.",
    )
    warning_cmd.set_defaults(func=run_warning_capture)
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    _resolve_solver_profile(args)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
