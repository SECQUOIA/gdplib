"""Tests for the stranded gas public builder."""

import os
import sys

import pytest

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
