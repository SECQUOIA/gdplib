"""Tests for the modular HENS public builder."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gdplib.mod_hens import build_model

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
