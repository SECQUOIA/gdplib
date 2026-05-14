"""Tests for the stranded gas public builder."""

import math
import os
import sys

import pytest
from pyomo.environ import TransformationFactory, value

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gdplib.stranded_gas import build_model

STRANDED_GAS_CASES = {
    "Gas_100": {"U100"},
    "Gas_250": {"U250"},
    "Gas_500": {"U500"},
    "Gas_small": {"U100", "U250"},
    "Gas_large": {"U250", "U500"},
}


@pytest.mark.parametrize("case,module_types", STRANDED_GAS_CASES.items())
def test_build_model_accepts_supported_cases(case, module_types):
    model = build_model(case)

    assert set(model.module_types) == module_types
    assert hasattr(model, "component_objects")


def test_build_model_without_case_keeps_unrestricted_module_set():
    model = build_model()

    assert {"A500", "U100", "U250", "U500"}.issubset(set(model.module_types))


def test_build_model_rejects_invalid_case():
    with pytest.raises(ValueError, match="Invalid stranded_gas case"):
        build_model("not-a-case")


def _set_purchase_count(model, mtype, count):
    purchase_vars = [
        model.modules_purchased[mtype, site, time]
        for site in model.potential_sites
        for time in model.time
    ]
    for var in purchase_vars:
        var.set_value(0)
    for var in purchase_vars[:count]:
        var.set_value(1)

    for option in model.purchase_count_options:
        model.purchase_count_selected[mtype, option].set_value(option == count)

    model.learning_factor[mtype].set_value(
        value(model.learning_factor_by_purchase_count[count])
    )


def test_module_learning_factor_is_tabulated_over_valid_purchase_counts():
    model = build_model("Gas_100")
    mtype = next(iter(model.module_types))

    assert list(model.purchase_count_options) == [0, 1, 2, 3, 4, 5]
    assert value(model.learning_factor_by_purchase_count[0]) == pytest.approx(1)
    for count in range(1, 6):
        expected = (1 - value(model.learning_rate)) ** (math.log(count) / math.log(2))
        assert value(model.learning_factor_by_purchase_count[count]) == pytest.approx(
            expected
        )

    for count in model.purchase_count_options:
        _set_purchase_count(model, mtype, count)

        assert value(model.select_one_purchase_count[mtype].body) == pytest.approx(1)
        assert value(model.match_purchase_count[mtype].body) == pytest.approx(0)
        if count == 0:
            assert value(
                model.mtype_absent[mtype].no_module_purchases.body
            ) == pytest.approx(0)
            assert value(
                model.mtype_absent[mtype].constant_learning_factor.body
            ) == pytest.approx(1)
        else:
            assert value(
                model.mtype_exists[mtype].require_module_purchases.body
            ) == pytest.approx(count)
            assert value(
                model.mtype_exists[mtype].learning_factor_calc.body
            ) == pytest.approx(0)


@pytest.mark.parametrize("case", [None, "Gas_100"])
@pytest.mark.parametrize("transformation", ["gdp.bigm", "gdp.hull"])
def test_stranded_gas_reformulates_with_supported_gdp_transformations(
    case, transformation
):
    model = build_model(case)

    TransformationFactory(transformation).apply_to(model)
