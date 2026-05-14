import pytest
import pyomo.environ as pyo
from pyomo.core.expr.visitor import identify_variables
from pyomo.gdp import Disjunct, Disjunction

from gdplib.biofuel import build_model

MODULE_COMPONENT_NAMES = {"num_modules", "modules_purchased", "modules_sold"}


def _module_variable_sites(constraint):
    sites = set()
    for var in identify_variables(constraint.body, include_fixed=False):
        if var.parent_component().local_name in MODULE_COMPONENT_NAMES:
            sites.add(var.index()[0])
    return sites


@pytest.mark.parametrize(
    "disjunct_name,constraint_name",
    [("site_inactive", "no_modules"), ("conventional", "no_modules")],
)
def test_biofuel_site_disjunct_module_constraints_are_site_local(
    disjunct_name, constraint_name
):
    model = build_model()

    for site in model.potential_sites:
        disjunct = getattr(model, disjunct_name)[site]
        constraint = getattr(disjunct, constraint_name)

        assert _module_variable_sites(constraint) == {site}


@pytest.mark.parametrize("transformation", ["gdp.bigm", "gdp.hull"])
def test_biofuel_reformulates_with_supported_gdp_transformations(transformation):
    model = build_model()

    pyo.TransformationFactory(transformation).apply_to(model)

    assert not any(model.component_data_objects(Disjunction, active=True))
    assert not any(model.component_data_objects(Disjunct, active=True))
