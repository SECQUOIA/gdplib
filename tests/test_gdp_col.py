"""Regression tests for the gdp_col distillation column model."""

import logging

import pyomo.environ as pyo
from pyomo.gdp import Disjunction

import gdplib.gdp_col


def _pyomo_warning_messages(caplog):
    return "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and record.name.startswith("pyomo")
    )


def test_gdp_col_build_initializes_feed_enthalpy_within_bounds(caplog):
    caplog.set_level(logging.WARNING)

    model = gdplib.gdp_col.build_model()

    for var in (model.H_L_spec_feed, model.H_V_spec_feed):
        for component in model.comps:
            value = pyo.value(var[component])
            lower, upper = var[component].bounds
            assert lower <= value <= upper

    messages = _pyomo_warning_messages(caplog)
    assert "H_L_spec_feed" not in messages
    assert "H_V_spec_feed" not in messages


def test_gdp_col_reformulates_with_hull():
    model = gdplib.gdp_col.build_model()

    assert any(model.component_data_objects(Disjunction, active=True))

    pyo.TransformationFactory("gdp.hull").apply_to(model)

    assert not any(model.component_data_objects(Disjunction, active=True))
