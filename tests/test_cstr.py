import os

import pytest
import pyomo.environ as pyo
from pyomo.gdp import Disjunction
from pyomo.opt import TerminationCondition

from gdplib.benchmark import _gdpopt_solve_kwargs
from gdplib.cstr import build_model


def test_cstr_activation_logic_uses_indicator_binary_constraints():
    model = build_model()

    assert isinstance(model.YF, pyo.Expression)
    assert isinstance(model.cstr_if_recycle, pyo.Constraint)
    assert isinstance(model.one_unreacted_feed, pyo.Constraint)
    assert isinstance(model.one_recycle, pyo.Constraint)
    assert isinstance(model.unit_in_n, pyo.Constraint)
    assert not any(model.component_data_objects(pyo.LogicalConstraint, active=True))
    assert any(model.component_data_objects(Disjunction, active=True))


@pytest.mark.skipif(
    os.environ.get("GDPLIB_RUN_GAMS_BARON_TESTS") != "1",
    reason="set GDPLIB_RUN_GAMS_BARON_TESTS=1 to run optional GAMS/BARON tests",
)
def test_cstr_gloa_matches_global_bigm_solution_with_gams_baron():
    if not pyo.SolverFactory("gams").available(False):
        pytest.skip("GAMS solver interface is not available")

    model = build_model()
    kwargs = _gdpopt_solve_kwargs(60, "gams", "baron")
    kwargs["tee"] = False
    for key in (
        "nlp_solver_args",
        "mip_solver_args",
        "minlp_solver_args",
        "local_minlp_solver_args",
    ):
        kwargs[key]["tee"] = False

    results = pyo.SolverFactory("gdpopt.gloa").solve(model, **kwargs)

    assert results.solver.termination_condition == TerminationCondition.optimal
    assert results.problem.lower_bound == pytest.approx(3.0620145766, abs=1e-6)
    assert results.problem.upper_bound == pytest.approx(3.0620145766, abs=1e-6)
