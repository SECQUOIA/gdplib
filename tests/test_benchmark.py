import json
from pathlib import Path

from benchmark import (
    _benchmark_metadata,
    _gams_solve_options,
    _gdpopt_solve_kwargs,
    _json_safe_result,
    _transformation_solve_kwargs,
    _write_failure_log,
)


def test_gams_solve_options_use_requested_timelimit():
    assert _gams_solve_options(123) == [
        "option reslim=123;option threads=1;option optcr=1e-2;"
    ]


def test_non_gams_transformation_solver_uses_native_timelimit():
    assert _transformation_solve_kwargs(45, "scip", "baron") == {
        "tee": True,
        "timelimit": 45,
    }


def test_gdpopt_gams_solver_args_use_requested_timelimit_and_solver():
    kwargs = _gdpopt_solve_kwargs(78, "gams", "baron")

    assert kwargs["time_limit"] == 78
    assert kwargs["nlp_solver_args"]["solver"] == "baron"
    assert kwargs["mip_solver_args"]["solver"] == "baron"
    assert kwargs["minlp_solver_args"]["solver"] == "baron"
    assert kwargs["local_minlp_solver_args"]["solver"] == "baron"
    assert "option reslim=78;" in kwargs["nlp_solver_args"]["add_options"]
    assert "option reslim=78;" in kwargs["mip_solver_args"]["add_options"]
    assert "option reslim=78;" in kwargs["minlp_solver_args"]["add_options"]
    assert "option reslim=78;" in kwargs["local_minlp_solver_args"]["add_options"]


def test_json_safe_result_replaces_non_finite_values():
    payload = {
        "Problem": [{"Lower bound": float("-inf"), "Upper bound": float("nan")}],
        "Solver": [{"Timing": {"total": float("inf")}}],
    }

    safe_payload = _json_safe_result(payload)

    assert safe_payload["Problem"][0]["Lower bound"] is None
    assert safe_payload["Problem"][0]["Upper bound"] is None
    assert safe_payload["Solver"][0]["Timing"]["total"] is None
    json.dumps(safe_payload, allow_nan=False)


def test_benchmark_metadata_records_gdpopt_gams_subsolvers():
    metadata = _benchmark_metadata("gdpopt.loa", 3600, "gams", "baron")

    assert metadata["Strategy"] == "gdpopt.loa"
    assert metadata["GAMS solver"] == "baron"
    assert metadata["Subsolvers"]["nlp"] == {"interface": "gams", "solver": "baron"}
    assert metadata["Subsolvers"]["mip"] == {"interface": "gams", "solver": "baron"}
    assert metadata["Subsolvers"]["minlp"] == {"interface": "gams", "solver": "baron"}
    assert metadata["Subsolvers"]["local_minlp"] == {
        "interface": "gams",
        "solver": "baron",
    }


def test_write_failure_log_records_context(tmp_path):
    try:
        raise RuntimeError("solver failed")
    except RuntimeError as err:
        path = _write_failure_log(
            tmp_path, "jobshop", "gdpopt.loa", "gams", "baron", err
        )

    contents = Path(path).read_text()
    assert "Instance: jobshop" in contents
    assert "Strategy: gdpopt.loa" in contents
    assert "Solver interface: gams" in contents
    assert "GAMS solver: baron" in contents
    assert "RuntimeError: solver failed" in contents
