"""Tests for the modular HENS public builder."""

import os
import sys

import pytest
import pyomo.environ as pyo
from pyomo.gdp import Disjunction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gdplib.mod_hens import build_model

CANONICAL_MOD_HENS_CASES = [
    "conventional",
    "single_module_integer",
    "multiple_module_integer",
    "mixed_integer",
    "single_module_discrete",
    "multiple_module_discrete",
    "mixed_discrete",
]

MOD_HENS_CASES = [
    "conventional",
    "single_module_integer",
    "multiple_module_integer",
    "require_modular_integer",
    "mixed_integer",
    "modular_option_integer",
    "single_module_discrete",
    "multiple_module_discrete",
    "require_modular_discrete",
    "mixed_discrete",
    "modular_option_discrete",
]


@pytest.mark.parametrize("case", MOD_HENS_CASES)
def test_build_model_accepts_supported_cases(case):
    model = build_model(case, cafaro_approx=False)

    assert model is not None
    assert hasattr(model, "component_objects")


def test_build_model_rejects_invalid_case():
    with pytest.raises(ValueError, match="Invalid mod_hens case"):
        build_model("not-a-case")


@pytest.mark.parametrize("case", CANONICAL_MOD_HENS_CASES)
def test_structurally_absent_matches_are_not_active_gdp_choices(case):
    model = build_model(case, cafaro_approx=False)

    active_exchanger_choices = [
        disjunction
        for disjunction in model.component_data_objects(Disjunction, active=True)
        if disjunction.parent_component() is model.exchanger_exists_or_absent
    ]
    fixed_active_choices = [
        disjunction.name
        for disjunction in active_exchanger_choices
        if any(disjunct.indicator_var.fixed for disjunct in disjunction.disjuncts)
    ]
    forced_absent_choices = [
        index
        for index in model.exchanger_exists_or_absent
        if not model.exchanger_exists_or_absent[index].active
    ]

    assert len(active_exchanger_choices) == 12
    assert fixed_active_choices == []
    assert len(forced_absent_choices) == 20

    for hot, cold, stg in forced_absent_choices:
        assert not model.exchanger_exists[hot, cold, stg].active
        assert not model.exchanger_absent[hot, cold, stg].active
        assert model.heat_exchanged[hot, cold, stg].fixed
        assert pyo.value(model.heat_exchanged[hot, cold, stg]) == 0
        assert model.exchanger_area[stg, hot, cold].fixed
        assert pyo.value(model.exchanger_area[stg, hot, cold]) == 0
        assert model.exchanger_area_cost[stg, hot, cold].fixed
        assert pyo.value(model.exchanger_area_cost[stg, hot, cold]) == 0
        assert model.exchanger_fixed_cost[stg, hot, cold].fixed
        assert pyo.value(model.exchanger_fixed_cost[stg, hot, cold]) == 0


@pytest.mark.parametrize("case", CANONICAL_MOD_HENS_CASES)
@pytest.mark.parametrize("transformation", ["gdp.bigm", "gdp.hull"])
def test_supported_cases_transform_after_absent_match_pruning(case, transformation):
    model = build_model(case, cafaro_approx=False, num_stages=2)

    pyo.TransformationFactory(transformation).apply_to(model)

    assert list(model.component_data_objects(Disjunction, active=True)) == []
