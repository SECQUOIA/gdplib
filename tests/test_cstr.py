import os
from itertools import product

import pytest
import pyomo.environ as pyo
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import TerminationCondition

from gdplib.benchmark import _gdpopt_solve_kwargs
from gdplib.cstr import build_model


def _indicator_binary(disjunct):
    return disjunct.indicator_var.get_associated_binary()


def _set_discrete_selection(model, cstr_values, recycle_values):
    for n, cstr_active, recycle_active in zip(model.N, cstr_values, recycle_values):
        _indicator_binary(model.YP_is_cstr[n]).set_value(int(cstr_active))
        _indicator_binary(model.YP_is_bypass[n]).set_value(int(not cstr_active))
        _indicator_binary(model.YR_is_recycle[n]).set_value(int(recycle_active))
        _indicator_binary(model.YR_is_not_recycle[n]).set_value(int(not recycle_active))


def _constraint_satisfied(constraint, tol=1e-8):
    body = pyo.value(constraint.body)
    if constraint.has_lb() and body < pyo.value(constraint.lower) - tol:
        return False
    if constraint.has_ub() and body > pyo.value(constraint.upper) + tol:
        return False
    return True


def _activation_constraints_satisfied(model):
    constraints = (
        list(model.cstr_if_recycle.values())
        + [model.one_unreacted_feed, model.one_recycle]
        + list(model.unit_in_n.values())
    )
    return all(_constraint_satisfied(constraint) for constraint in constraints)


def _active_prefix_length(cstr_values):
    prefix_length = 0
    seen_bypass = False
    for cstr_active in cstr_values:
        if cstr_active:
            if seen_bypass:
                return None
            prefix_length += 1
        else:
            seen_bypass = True
    return prefix_length or None


def _single_recycle_location(recycle_values):
    locations = [
        n for n, recycle_active in enumerate(recycle_values, start=1) if recycle_active
    ]
    if len(locations) != 1:
        return None
    return locations[0]


def test_cstr_activation_logic_uses_indicator_binary_constraints():
    model = build_model()

    assert isinstance(model.YF, pyo.Expression)
    assert isinstance(model.cstr_if_recycle, pyo.Constraint)
    assert isinstance(model.one_unreacted_feed, pyo.Constraint)
    assert isinstance(model.one_recycle, pyo.Constraint)
    assert isinstance(model.unit_in_n, pyo.Constraint)
    assert not any(model.component_data_objects(pyo.LogicalConstraint, active=True))
    assert any(model.component_data_objects(Disjunction, active=True))


def test_cstr_activation_logic_admits_exactly_prefix_recycle_combinations():
    model = build_model()

    for cstr_values in product([False, True], repeat=len(model.N)):
        prefix_length = _active_prefix_length(cstr_values)
        for recycle_values in product([False, True], repeat=len(model.N)):
            recycle_location = _single_recycle_location(recycle_values)
            _set_discrete_selection(model, cstr_values, recycle_values)

            expected = (
                prefix_length is not None
                and recycle_location is not None
                and recycle_location <= prefix_length
            )
            assert _activation_constraints_satisfied(model) is expected, (
                cstr_values,
                recycle_values,
            )

            if expected:
                for n in model.N:
                    assert pyo.value(model.YF[n]) == pytest.approx(
                        int(n == prefix_length)
                    )


@pytest.mark.parametrize("transformation", ["gdp.bigm", "gdp.hull"])
def test_cstr_activation_logic_reformulates_with_supported_gdp_transformations(
    transformation,
):
    model = build_model()

    pyo.TransformationFactory(transformation).apply_to(model)

    assert not any(model.component_data_objects(pyo.LogicalConstraint, active=True))
    assert not any(model.component_data_objects(Disjunction, active=True))
    assert not any(model.component_data_objects(Disjunct, active=True))


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
