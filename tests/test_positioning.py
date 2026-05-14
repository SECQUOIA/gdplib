import math
import os

import pytest
import pyomo.environ as pyo
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import TerminationCondition

from gdplib.benchmark import _gdpopt_model_initialization_kwargs, _gdpopt_solve_kwargs
from gdplib.positioning import build_model


def _consumer_best_existing_distance(model, consumer):
    return min(
        sum(
            pyo.value(model.weights[consumer, location])
            * (
                pyo.value(model.existing_products[product, location])
                - pyo.value(model.ideal_points[consumer, location])
            )
            ** 2
            for location in model.locations
        )
        for product in model.products
    )


def _consumer_max_new_product_distance(model, consumer):
    return sum(
        pyo.value(model.weights[consumer, location])
        * max(
            (model.x[location].lb - pyo.value(model.ideal_points[consumer, location]))
            ** 2,
            (model.x[location].ub - pyo.value(model.ideal_points[consumer, location]))
            ** 2,
        )
        for location in model.locations
    )


def test_positioning_slack_bounds_are_source_derived():
    model = build_model()
    expected_bounds = {}

    for consumer in model.consumers:
        expected_bounds[consumer] = max(
            0,
            _consumer_max_new_product_distance(model, consumer)
            - _consumer_best_existing_distance(model, consumer),
        )

        assert pyo.value(model.consumer_slack_bound[consumer]) == pytest.approx(
            expected_bounds[consumer]
        )

    assert model.U.lb == 0
    assert model.U.ub == pytest.approx(max(expected_bounds.values()))
    assert model.U.ub < 5000


def test_positioning_gdpopt_initialization_set_matches_verified_solution():
    model = build_model()
    (initial_disjuncts,) = model.gdpopt_initial_disjuncts

    selected_consumers = [
        consumer
        for consumer in model.consumers
        if model.d[consumer].disjuncts[0] in initial_disjuncts
    ]

    assert selected_consumers == [1, 6, 8, 15, 17, 20, 25]
    for consumer in model.consumers:
        assert (
            sum(
                disjunct in initial_disjuncts
                for disjunct in model.d[consumer].disjuncts
            )
            == 1
        )


@pytest.mark.parametrize("transformation", ["gdp.bigm", "gdp.hull"])
def test_positioning_reformulates_with_supported_gdp_transformations(transformation):
    model = build_model()

    pyo.TransformationFactory(transformation).apply_to(model)

    assert not any(model.component_data_objects(pyo.LogicalConstraint, active=True))
    assert not any(model.component_data_objects(Disjunction, active=True))
    assert not any(model.component_data_objects(Disjunct, active=True))


@pytest.mark.skipif(
    os.environ.get("GDPLIB_RUN_GAMS_TESTS") != "1",
    reason="set GDPLIB_RUN_GAMS_TESTS=1 to run optional GAMS-backed tests",
)
def test_positioning_gloa_runs_with_gams_local_profile():
    if not pyo.SolverFactory("gams").available(False):
        pytest.skip("GAMS solver interface is not available")

    model = build_model()
    kwargs = _gdpopt_solve_kwargs(
        60,
        "gams",
        "dicopt",
        gams_nlp_solver="ipopth",
        gams_mip_solver="gurobi",
        gams_minlp_solver="dicopt",
        gams_local_minlp_solver="dicopt",
    )
    kwargs["tee"] = False
    kwargs.update(_gdpopt_model_initialization_kwargs(model))
    for key in (
        "nlp_solver_args",
        "mip_solver_args",
        "minlp_solver_args",
        "local_minlp_solver_args",
    ):
        kwargs[key]["tee"] = False

    results = pyo.SolverFactory("gdpopt.gloa").solve(model, **kwargs)

    assert results.solver.termination_condition == TerminationCondition.optimal
    assert math.isfinite(results.problem.lower_bound)
    assert math.isfinite(results.problem.upper_bound)
    assert results.problem.lower_bound == pytest.approx(-8.06413617, abs=1e-6)
    assert results.problem.upper_bound == pytest.approx(-8.06413617, abs=1e-6)
