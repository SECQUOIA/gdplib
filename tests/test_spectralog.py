import pytest
import pyomo.environ as pyo
from pyomo.gdp import Disjunct, Disjunction

from gdplib.spectralog import build_model

EXPECTED_VAL_UPPER_BOUNDS = {
    1: 157341.7613,
    2: 807704.4181,
    3: 511037.7934,
    4: 476646.6045,
    5: 802891.9736,
    6: 569936.876,
    7: 429417.6493,
    8: 186408.5249,
}


def test_spectralog_objective_value_variables_have_source_bounds():
    model = build_model()

    for j, upper_bound in EXPECTED_VAL_UPPER_BOUNDS.items():
        assert model.val[j].lb == 0
        assert model.val[j].ub == pytest.approx(upper_bound)


@pytest.mark.parametrize("transformation", ["gdp.bigm", "gdp.hull"])
def test_spectralog_reformulates_with_supported_gdp_transformations(transformation):
    model = build_model()

    pyo.TransformationFactory(transformation).apply_to(model)

    assert not any(model.component_data_objects(pyo.LogicalConstraint, active=True))
    assert not any(model.component_data_objects(Disjunction, active=True))
    assert not any(model.component_data_objects(Disjunct, active=True))
