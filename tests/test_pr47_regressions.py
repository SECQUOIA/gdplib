import pytest
from pyomo.environ import TransformationFactory, value

from gdplib.hda import HDA_model
from gdplib.kaibel import build_model as build_kaibel_model


def test_hda_model_constructs_with_scalar_heat_capacity_ratio():
    model = HDA_model()

    assert value(model.heat_capacity_ratio) == 1.3
    assert model.gamma.is_indexed()


def test_kaibel_reduced_temperature_initial_values_are_within_bounds():
    model = build_kaibel_model()

    out_of_bounds = [
        var.name
        for var in model.Tr.values()
        if value(var) < var.lb or value(var) > var.ub
    ]

    assert out_of_bounds == []


@pytest.mark.parametrize('transformation', ['gdp.hull', 'gdp.bigm'])
def test_kaibel_gdp_transformation_constructs(transformation):
    model = build_kaibel_model()

    TransformationFactory(transformation).apply_to(model)
