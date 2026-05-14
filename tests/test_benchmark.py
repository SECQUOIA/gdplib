import json
from pathlib import Path
from types import SimpleNamespace

import pyomo.environ as pyo
from pyomo.core.expr.visitor import polynomial_degree

from gdplib.benchmark import (
    DEFAULT_GAMS_OPTCR,
    DEFAULT_STRATEGIES,
    PR58_BENCHMARK_INSTANCES,
    _benchmark_metadata,
    _generate_summary,
    _gams_solve_options,
    _gdpopt_model_initialization_kwargs,
    _gdpopt_solve_kwargs,
    _json_safe_result,
    _resolve_solver_profile,
    _transformation_solve_kwargs,
    _write_failure_log,
    benchmark_cases,
    main,
    read_benchmark_cases_file,
    summarize_warning_rows,
)


def test_gams_solve_options_use_requested_timelimit():
    assert _gams_solve_options(123) == [
        "option reslim=123;option threads=1;option optcr=1e-6;"
    ]


def test_default_gams_optcr_is_tight_for_benchmark_comparisons():
    assert DEFAULT_GAMS_OPTCR == "1e-6"


def test_non_gams_transformation_solver_uses_native_timelimit():
    assert _transformation_solve_kwargs(45, "scip", "baron") == {
        "tee": True,
        "timelimit": 45,
    }


def test_gams_dicopt_transformation_records_decomposition_solvers():
    kwargs = _transformation_solve_kwargs(45, "gams", "dicopt", "ipopth", "gurobi")

    assert kwargs["solver"] == "dicopt"
    assert (
        "option reslim=45;option threads=1;option optcr=1e-6;" in kwargs["add_options"]
    )
    assert "option nlp=ipopth;" in kwargs["add_options"]
    assert "option mip=gurobi;" in kwargs["add_options"]


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
    assert "option optcr=1e-6;" in kwargs["nlp_solver_args"]["add_options"]
    assert "option optcr=1e-6;" in kwargs["mip_solver_args"]["add_options"]
    assert "option optcr=1e-6;" in kwargs["minlp_solver_args"]["add_options"]
    assert "option optcr=1e-6;" in kwargs["local_minlp_solver_args"]["add_options"]


def test_gdpopt_local_profile_uses_role_solvers():
    kwargs = _gdpopt_solve_kwargs(
        12,
        "gams",
        "dicopt",
        gams_nlp_solver="ipopth",
        gams_mip_solver="gurobi",
        gams_minlp_solver="dicopt",
        gams_local_minlp_solver="dicopt",
    )

    assert kwargs["time_limit"] == 12
    assert kwargs["nlp_solver_args"]["solver"] == "ipopth"
    assert kwargs["mip_solver_args"]["solver"] == "gurobi"
    assert kwargs["minlp_solver_args"]["solver"] == "dicopt"
    assert kwargs["local_minlp_solver_args"]["solver"] == "dicopt"
    assert "option reslim=12;" in kwargs["nlp_solver_args"]["add_options"]
    assert "option optcr=1e-6;" in kwargs["mip_solver_args"]["add_options"]
    assert "option nlp=ipopth;" in kwargs["minlp_solver_args"]["add_options"]
    assert "option mip=gurobi;" in kwargs["minlp_solver_args"]["add_options"]
    assert "option optcr=1e-6;" in kwargs["minlp_solver_args"]["add_options"]
    assert "option optcr=1e-6;" in kwargs["local_minlp_solver_args"]["add_options"]
    assert "$onecho > baron.opt" not in kwargs["nlp_solver_args"]["add_options"]


def test_gdpopt_model_initialization_kwargs_use_custom_disjuncts():
    import gdplib.positioning

    model = gdplib.positioning.build_model()

    kwargs = _gdpopt_model_initialization_kwargs(model)

    assert kwargs["init_algorithm"] == "custom_disjuncts"
    assert len(kwargs["custom_init_disjuncts"]) == 1
    assert len(kwargs["custom_init_disjuncts"][0]) == len(model.consumers)


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
    assert metadata["GAMS optcr"] == "1e-6"
    assert metadata["Subsolvers"]["nlp"] == {"interface": "gams", "solver": "baron"}
    assert metadata["Subsolvers"]["mip"] == {"interface": "gams", "solver": "baron"}
    assert metadata["Subsolvers"]["minlp"] == {"interface": "gams", "solver": "baron"}
    assert metadata["Subsolvers"]["local_minlp"] == {
        "interface": "gams",
        "solver": "baron",
    }


def test_solver_profile_resolution_sets_local_gams_defaults():
    class Args:
        solver_profile = "gams-local"
        subsolver = None
        gams_solvers = None
        gams_nlp_solver = None
        gams_mip_solver = None
        gams_minlp_solver = None
        gams_local_minlp_solver = None

    args = _resolve_solver_profile(Args())

    assert args.subsolver == "gams"
    assert args.gams_solvers == ["dicopt"]
    assert args.gams_nlp_solver == "ipopth"
    assert args.gams_mip_solver == "gurobi"
    assert args.gams_minlp_solver == "dicopt"
    assert args.gams_local_minlp_solver == "dicopt"


def test_cases_file_parses_per_case_solver_settings(tmp_path):
    cases_file = tmp_path / "cases.csv"
    cases_file.write_text(
        "instance,strategy,solver_profile,gams_solver,timelimit,label,"
        "gams_nlp_solver,gams_mip_solver,gams_minlp_solver,"
        "gams_local_minlp_solver\n"
        "batch_processing,gdpopt.loa,gams-local,dicopt,30,local_loa,"
        "ipopth,gurobi,dicopt,dicopt\n"
        "batch_processing,gdpopt.loa,gams-baron,baron,60,global_loa,"
        "baron,baron,baron,baron\n"
    )
    defaults = SimpleNamespace(
        timelimit=99,
        solver_profile="gams-local",
        subsolver="gams",
        gams_solvers=["dicopt"],
        gams_nlp_solver="ipopth",
        gams_mip_solver="gurobi",
        gams_minlp_solver="dicopt",
        gams_local_minlp_solver="dicopt",
    )

    cases = read_benchmark_cases_file(cases_file, defaults)

    assert len(cases) == 2
    assert cases[0].solver_profile == "gams-local"
    assert cases[0].timelimit == 30
    assert cases[0].solver_gams == "dicopt"
    assert cases[0].gams_nlp_solver == "ipopth"
    assert cases[0].gams_mip_solver == "gurobi"
    assert cases[0].label == "local_loa"
    assert cases[1].solver_profile == "gams-baron"
    assert cases[1].gams_nlp_solver == "baron"
    assert cases[1].label == "global_loa"


def test_committed_pr58_local_cases_cover_default_matrix():
    cases_file = Path(__file__).parents[1] / "benchmark_cases" / "pr58_local.csv"
    defaults = SimpleNamespace(
        timelimit=99,
        solver_profile="gams-local",
        subsolver="gams",
        gams_solvers=["dicopt"],
        gams_nlp_solver="ipopth",
        gams_mip_solver="gurobi",
        gams_minlp_solver="dicopt",
        gams_local_minlp_solver="dicopt",
    )

    cases = read_benchmark_cases_file(cases_file, defaults)

    assert len(cases) == len(PR58_BENCHMARK_INSTANCES) * len(DEFAULT_STRATEGIES)
    assert {case.instance for case in cases} == set(PR58_BENCHMARK_INSTANCES)
    assert {case.strategy for case in cases} == set(DEFAULT_STRATEGIES)
    assert {case.solver_profile for case in cases} == {"gams-local"}
    med_term_direct_mip_cases = {
        ("med_term_purchasing", "gdp.bigm"),
        ("med_term_purchasing", "gdp.hull"),
    }
    med_term_lbb_case = ("med_term_purchasing", "gdpopt.lbb")
    assert {
        case.solver_gams
        for case in cases
        if (case.instance, case.strategy) in med_term_direct_mip_cases
    } == {"gurobi"}
    assert {
        case.solver_gams
        for case in cases
        if (case.instance, case.strategy) not in med_term_direct_mip_cases
    } == {"dicopt"}
    assert {case.gams_nlp_solver for case in cases} == {"ipopth"}
    assert {case.gams_mip_solver for case in cases} == {"gurobi"}
    assert {
        case.gams_minlp_solver
        for case in cases
        if (case.instance, case.strategy) == med_term_lbb_case
    } == {"gurobi"}
    assert {
        case.gams_minlp_solver
        for case in cases
        if (case.instance, case.strategy) != med_term_lbb_case
    } == {"dicopt"}
    assert {case.gams_local_minlp_solver for case in cases} == {"dicopt"}


def test_med_term_purchasing_direct_reformulations_are_linear_mips():
    import gdplib.med_term_purchasing

    model = gdplib.med_term_purchasing.build_model()

    for transformation in ("gdp.bigm", "gdp.hull"):
        transformed = model.clone()
        pyo.TransformationFactory(transformation).apply_to(transformed)
        constraints = transformed.component_data_objects(pyo.Constraint, active=True)
        objectives = transformed.component_data_objects(pyo.Objective, active=True)
        degrees = [polynomial_degree(constraint.body) for constraint in constraints]
        degrees.extend(polynomial_degree(objective.expr) for objective in objectives)

        assert any(
            var.is_binary() for var in transformed.component_data_objects(pyo.Var)
        )
        assert all(degree in (0, 1) for degree in degrees)


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


def test_benchmark_cases_expand_campaign_matrix():
    cases = benchmark_cases(
        ["batch_processing", "cstr"], ["gdp.bigm", "gdp.hull"], "gams", ["baron"]
    )

    assert len(cases) == 4
    assert cases[0].instance == "batch_processing"
    assert cases[0].strategy == "gdp.bigm"
    assert cases[0].subsolver == "gams"
    assert cases[0].solver_gams == "baron"


def test_cli_preflight_can_plan_without_solver_or_model_checks(capsys):
    status = main(
        [
            "preflight",
            "--instances",
            "batch_processing",
            "--strategies",
            "gdp.bigm",
            "--no-build-models",
            "--no-check-solvers",
        ]
    )

    output = capsys.readouterr().out
    assert status == 0
    assert "Benchmark plan" in output
    assert "cases: 1" in output
    assert "solver profiles: gams-local" in output
    assert "Preflight passed" in output


def test_cli_preflight_uses_cases_file(tmp_path, capsys):
    cases_file = tmp_path / "cases.csv"
    cases_file.write_text(
        "instance,strategy,solver_profile,gams_solver,timelimit,label\n"
        "batch_processing,gdpopt.loa,gams-local,dicopt,30,local_loa\n"
        "cstr,gdp.bigm,gams-gurobi,gurobi,20,gurobi_bigm\n"
    )

    status = main(
        [
            "preflight",
            "--cases-file",
            str(cases_file),
            "--no-build-models",
            "--no-check-solvers",
        ]
    )

    output = capsys.readouterr().out
    assert status == 0
    assert f"cases file: {cases_file}" in output
    assert "instances: 2" in output
    assert "cases: 2" in output
    assert "solver profiles: gams-gurobi, gams-local" in output


def test_cli_run_dry_run_uses_preflight_without_solving(capsys):
    status = main(
        [
            "run",
            "--instances",
            "batch_processing",
            "--strategies",
            "gdp.bigm",
            "--no-build-models",
            "--no-check-solvers",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert status == 0
    assert "Benchmark plan" in output
    assert "Dry run complete" in output


def test_generate_summary_uses_packaged_summary_module(tmp_path, monkeypatch):
    result_dir = tmp_path / "gdplib" / "methanol" / "benchmark_result" / "run_1"
    result_dir.mkdir(parents=True)
    (result_dir / "gdpopt.gloa_gams_baron.json").write_text(
        json.dumps(
            {
                "Problem": [
                    {
                        "Name": "methanol-gloa",
                        "Lower bound": -1743.4292381783366,
                        "Upper bound": -1743.4292381783366,
                        "Sense": "minimize",
                    }
                ],
                "Solver": [
                    {
                        "Name": "GDPopt (22, 5, 13) - GLOA",
                        "Termination condition": "optimal",
                        "User time": 3.7971969350182917,
                    }
                ],
            }
        )
    )
    monkeypatch.chdir(tmp_path)

    assert _generate_summary("run_1", ["methanol"]) is True

    combined_data = tmp_path / "benchmark_summary" / "combined_data.csv"
    assert combined_data.exists()
    assert "GDPopt (22, 5, 13) - GLOA" in combined_data.read_text()


def test_generate_summary_returns_false_without_json_results(tmp_path, monkeypatch):
    result_dir = tmp_path / "gdplib" / "methanol" / "benchmark_result" / "empty_run"
    result_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    assert _generate_summary("empty_run", ["methanol"]) is False
    assert not (tmp_path / "benchmark_summary").exists()


def test_warning_summary_marks_indicator_casts_as_deprecations():
    rows = summarize_warning_rows(
        [
            {
                "instance": "gdp_col",
                "model_issue": 66,
                "phase": "build",
                "strategy": None,
                "subsolver": "gams",
                "solver_gams": "baron",
                "source": "pyomo.py:1",
                "category": "pyomo.core",
                "is_deprecation_candidate": True,
                "deprecation_tracker_issue": 55,
                "message": (
                    "WARNING: implicitly casting 'tray[2].indicator_var' "
                    "value 1 to bool"
                ),
            }
        ]
    )

    assert rows[0]["count"] == 1
    assert rows[0]["model_issue"] == 66
    assert rows[0]["deprecation_tracker_issue"] == 55


def test_cli_warning_capture_build_mode_writes_reports(tmp_path, capsys):
    status = main(
        [
            "warnings",
            "--instances",
            "batch_processing",
            "--run-id",
            "warning_test",
            "--metadata-dir",
            str(tmp_path),
        ]
    )

    output = capsys.readouterr().out
    report_dir = tmp_path / "warning_test"
    assert status == 0
    assert "Warning capture plan" in output
    assert (report_dir / "warning_events.csv").exists()
    assert (report_dir / "warning_summary.csv").exists()
    assert (report_dir / "issue_55_deprecation_candidates.md").exists()
