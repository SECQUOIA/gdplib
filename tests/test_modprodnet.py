"""Tests for modular production network model variants."""

import pytest
import pyomo.environ as pyo
from pyomo.gdp import Disjunction

from gdplib.modprodnet import build_model

MODPRODNET_CASES = ["Growth", "Dip", "Decay", "Distributed", "QuarterDistributed"]
DISTRIBUTED_CASES = ["Distributed", "QuarterDistributed"]


@pytest.mark.parametrize("case", MODPRODNET_CASES)
def test_build_model_accepts_supported_cases(case):
    model = build_model(case)

    assert model is not None
    assert hasattr(model, "component_objects")


def test_build_model_rejects_invalid_case():
    with pytest.raises(ValueError, match="Invalid case"):
        build_model("not-a-case")


@pytest.mark.parametrize("case", MODPRODNET_CASES)
@pytest.mark.parametrize("transformation", ["gdp.bigm", "gdp.hull"])
def test_supported_cases_reformulate(case, transformation):
    model = build_model(case)

    pyo.TransformationFactory(transformation).apply_to(model)

    assert not any(model.component_data_objects(Disjunction, active=True))


def _time_periods(model):
    return model.months if hasattr(model, "months") else model.quarters


def _constraints_satisfied(block, tolerance=1e-8):
    for constraint in block.component_data_objects(pyo.Constraint, active=True):
        body = pyo.value(constraint.body)
        if (
            constraint.lower is not None
            and body < pyo.value(constraint.lower) - tolerance
        ):
            return False
        if (
            constraint.upper is not None
            and body > pyo.value(constraint.upper) + tolerance
        ):
            return False
    return True


@pytest.mark.parametrize("case", DISTRIBUTED_CASES)
@pytest.mark.parametrize(
    ("site1_active", "site2_active", "expected"),
    [(0, 0, True), (1, 0, True), (0, 1, True), (1, 1, False)],
)
def test_pair_inactive_represents_not_both_sites_active(
    case, site1_active, site2_active, expected
):
    model = build_model(case)
    site1, site2 = 1, 2
    model.site_active[site1].binary_indicator_var.set_value(site1_active)
    model.site_active[site2].binary_indicator_var.set_value(site2_active)
    for period in _time_periods(model):
        model.modules_transferred[site1, site2, period].set_value(0)

    assert _constraints_satisfied(model.pair_inactive[site1, site2]) is expected
