import pytest
import pyomo.environ as pyo
from pyomo.gdp import Disjunct, Disjunction

from gdplib.ex1_linan_2023 import build_model


def test_ex1_linan_disjunctions_are_xor():
    model = build_model()

    assert model.Disjunction1.xor is True
    assert model.Disjunction2.xor is True


@pytest.mark.parametrize("transformation", ["gdp.bigm", "gdp.hull"])
def test_ex1_linan_reformulates_with_supported_gdp_transformations(transformation):
    model = build_model()

    pyo.TransformationFactory(transformation).apply_to(model)

    assert not any(model.component_data_objects(pyo.LogicalConstraint, active=True))
    assert not any(model.component_data_objects(Disjunction, active=True))
    assert not any(model.component_data_objects(Disjunct, active=True))
