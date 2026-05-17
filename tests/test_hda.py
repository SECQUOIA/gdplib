import logging

import pytest
import pyomo.environ as pyo
from pyomo.gdp import Disjunct, Disjunction

import gdplib.hda


def _pyomo_warning_messages(caplog):
    return "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and record.name.startswith("pyomo")
    )


def test_hda_build_uses_bounded_initial_values(caplog):
    caplog.set_level(logging.WARNING)

    gdplib.hda.build_model()

    messages = _pyomo_warning_messages(caplog)
    assert "outside the bounds" not in messages
    assert "Setting Var" not in messages


@pytest.mark.parametrize("transformation", ["gdp.bigm", "gdp.hull"])
def test_hda_reformulates_with_supported_gdp_transformations(transformation):
    model = gdplib.hda.build_model()

    pyo.TransformationFactory(transformation).apply_to(model)

    assert not any(model.component_data_objects(pyo.LogicalConstraint, active=True))
    assert not any(model.component_data_objects(Disjunction, active=True))
    assert not any(model.component_data_objects(Disjunct, active=True))
