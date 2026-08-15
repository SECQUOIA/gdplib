import math
import os

import pytest
import pyomo.environ as pyo
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import TerminationCondition

from gdplib.batch_processing import build_model


def _gloa_gams_baron_kwargs(timelimit):
    """Self-contained GDPopt kwargs for the GAMS/BARON smoke test.

    Kept independent of gdplib.benchmark internals so benchmark refactors
    cannot break this model regression test.
    """

    def _solver_args():
        return dict(
            solver="baron",
            add_options=[f"option reslim={timelimit};", "option optcr=1e-6;"],
            tee=False,
        )

    return {
        "tee": False,
        "time_limit": timelimit,
        "nlp_solver": "gams",
        "nlp_solver_args": _solver_args(),
        "mip_solver": "gams",
        "mip_solver_args": _solver_args(),
        "minlp_solver": "gams",
        "minlp_solver_args": _solver_args(),
        "local_minlp_solver": "gams",
        "local_minlp_solver_args": _solver_args(),
    }


def test_batch_processing_cycle_time_log_has_finite_source_bounds():
    model = build_model()

    for i in model.PRODUCTS:
        expected_lb = max(
            math.log(pyo.value(model.ProcessingTime[i, j]))
            - pyo.value(model.batchSize_log[i, j].ub)
            - pyo.value(model.unitsOutOfPhaseUB[j])
            for j in model.STAGES
        )
        expected_ub = math.log(
            pyo.value(model.HorizonTime) / pyo.value(model.ProductionAmount[i])
        )

        assert model.cycleTime_log[i].lb == pytest.approx(expected_lb)
        assert model.cycleTime_log[i].ub == pytest.approx(expected_ub)


def test_batch_processing_fbbt_handles_cycle_time_bounds():
    model = build_model()

    fbbt(model)

    for i in model.PRODUCTS:
        assert math.isfinite(model.cycleTime_log[i].lb)
        assert math.isfinite(model.cycleTime_log[i].ub)


@pytest.mark.parametrize("transformation", ["gdp.bigm", "gdp.hull"])
def test_batch_processing_reformulates_with_supported_gdp_transformations(
    transformation,
):
    model = build_model()

    pyo.TransformationFactory(transformation).apply_to(model)

    # Forward guard: the model currently declares no LogicalConstraint (its
    # XORs are algebraic), so this only fires if future edits add one that a
    # transformation then fails to deactivate.
    assert not any(model.component_data_objects(pyo.LogicalConstraint, active=True))
    assert not any(model.component_data_objects(Disjunction, active=True))
    assert not any(model.component_data_objects(Disjunct, active=True))


@pytest.mark.skipif(
    os.environ.get("GDPLIB_RUN_GAMS_BARON_TESTS") != "1",
    reason="set GDPLIB_RUN_GAMS_BARON_TESTS=1 to run optional GAMS/BARON tests",
)
def test_batch_processing_gloa_runs_with_gams_baron():
    if not pyo.SolverFactory("gams").available(False):
        pytest.skip("GAMS solver interface is not available")

    model = build_model()
    kwargs = _gloa_gams_baron_kwargs(20)

    results = pyo.SolverFactory("gdpopt.gloa").solve(model, **kwargs)

    assert results.solver.termination_condition in {
        TerminationCondition.optimal,
        TerminationCondition.maxTimeLimit,
    }
    assert math.isfinite(results.problem.lower_bound)
    assert math.isfinite(results.problem.upper_bound)
    assert results.problem.lower_bound <= results.problem.upper_bound
