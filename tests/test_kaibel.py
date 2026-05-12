import logging

import pytest
import pyomo.environ as pyo
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
