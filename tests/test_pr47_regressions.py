from math import exp

import pytest
from pyomo.environ import TransformationFactory, value

from gdplib.hda import HDA_model
from gdplib.kaibel import build_model as build_kaibel_model


def test_hda_model_constructs_with_scalar_heat_capacity_ratio():
    model = HDA_model()

    assert value(model.gam) == 1.3
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


def _original_vapor_composition_residual(model, section, tray, component):
    temperature = value(model.T[section, tray])
    reduced_temperature = value(model.prop[component, 'TC']) / temperature
    vapor_pressure_exponent = reduced_temperature * (
        value(model.prop[component, 'vpA'])
        * (1 - temperature / value(model.prop[component, 'TC']))
        + value(model.prop[component, 'vpB'])
        * (1 - temperature / value(model.prop[component, 'TC'])) ** 1.5
        + value(model.prop[component, 'vpC'])
        * (1 - temperature / value(model.prop[component, 'TC'])) ** 3
        + value(model.prop[component, 'vpD'])
        * (1 - temperature / value(model.prop[component, 'TC'])) ** 6
    )
    vapor_pressure = value(model.prop[component, 'PC']) * exp(vapor_pressure_exponent)
    original_rhs = (
        value(model.x[section, tray, component])
        * value(model.actv[section, tray, component])
        * vapor_pressure
        / value(model.P[section, tray])
    )
    return value(model.y[section, tray, component]) - original_rhs


@pytest.mark.parametrize(
    ('section', 'reduced_temperature_name', 'vapor_composition_name'),
    [
        (1, '_bottom_reduced_temperature', 'bottom_vapor_composition'),
        (2, '_feedside_reduced_temperature', 'feedside_vapor_composition'),
        (3, '_productside_reduced_temperature', 'productside_vapor_composition'),
        (4, '_top_reduced_temperature', 'top_vapor_composition'),
    ],
)
def test_kaibel_reduced_temperature_reformulation_matches_original_algebra(
    section, reduced_temperature_name, vapor_composition_name
):
    model = build_kaibel_model()
    tray = 1

    for component in model.comp:
        temperature = value(model.T[section, tray])
        model.Tr[section, tray, component].set_value(
            value(model.prop[component, 'TC']) / temperature
        )

        disjunct = model.tray_exists[section, tray]
        reduced_temperature = getattr(disjunct, reduced_temperature_name)[component]
        vapor_composition = getattr(disjunct, vapor_composition_name)[component]
        scaled_reformulation_residual = value(vapor_composition.body) / value(
            model.P[section, tray]
        )

        assert value(reduced_temperature.body) == pytest.approx(
            value(reduced_temperature.lower)
        )
        assert scaled_reformulation_residual == pytest.approx(
            _original_vapor_composition_residual(model, section, tray, component)
        )
