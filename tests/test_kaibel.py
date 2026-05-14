import logging

import pytest
import pyomo.environ as pyo
from pyomo.core.expr.numeric_expr import PowExpression
from pyomo.core.expr.visitor import identify_components
from pyomo.gdp import Disjunct, Disjunction

from build_model_test_utils import is_missing_external_solver_error


def _build_kaibel_or_skip():
    from gdplib.kaibel import build_model

    try:
        return build_model()
    except Exception as err:
        if is_missing_external_solver_error(err):
            pytest.skip(f"kaibel build_model requires external solver: {err}")
        raise


def _pyomo_warning_messages(caplog):
    return "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and record.name.startswith("pyomo")
    )


def test_kaibel_total_flow_initial_values_respect_bounds(caplog):
    caplog.set_level(logging.WARNING)

    model = _build_kaibel_or_skip()

    for total_flow in (model.Vtotal, model.Ltotal):
        for flow_data in total_flow.values():
            value = pyo.value(flow_data)
            assert flow_data.lb <= value <= flow_data.ub

    for gap_data in model.TcGap.values():
        value = pyo.value(gap_data)
        assert gap_data.lb >= 0
        assert gap_data.lb <= value <= gap_data.ub

    messages = _pyomo_warning_messages(caplog)
    assert "Setting Var 'Vtotal" not in messages
    assert "Setting Var 'Ltotal" not in messages


@pytest.mark.parametrize("transformation", ["gdp.bigm", "gdp.hull"])
def test_kaibel_reformulates_with_supported_gdp_transformations(transformation):
    model = _build_kaibel_or_skip()

    assert all(disjunction.xor for disjunction in model.tray_exists_or_not.values())

    pyo.TransformationFactory(transformation).apply_to(model)

    assert not any(model.component_data_objects(pyo.LogicalConstraint, active=True))
    assert not any(model.component_data_objects(Disjunction, active=True))
    assert not any(model.component_data_objects(Disjunct, active=True))


def test_kaibel_vapor_pressure_avoids_fractional_power_expressions():
    model = _build_kaibel_or_skip()

    for disjunct in model.tray_exists.values():
        for constraint in disjunct.component_data_objects(pyo.Constraint):
            if "vapor_composition" not in constraint.local_name:
                continue
            for power in identify_components(constraint.body, (PowExpression,)):
                exponent = pyo.value(power.arg(1), exception=False)
                assert exponent in {3, 6}


def test_kaibel_hull_expressions_evaluate_for_gams_writer():
    model = _build_kaibel_or_skip()

    pyo.TransformationFactory("gdp.hull").apply_to(model)

    uninitialized_vars = []
    for var_data in model.component_data_objects(pyo.Var):
        if var_data.value is None:
            uninitialized_vars.append(var_data)
            var_data.set_value(0, skip_validation=True)

    try:
        for constraint in model.component_data_objects(pyo.Constraint, active=True):
            if not constraint.body.is_fixed():
                pyo.value(constraint.body)
        pyo.value(next(model.component_data_objects(pyo.Objective, active=True)).expr)
    finally:
        for var_data in uninitialized_vars:
            var_data.set_value(None)
