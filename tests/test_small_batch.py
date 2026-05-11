import math

import pytest
import pyomo.environ as pyo
from pyomo.environ import value
from pyomo.gdp import Disjunction

import gdplib.small_batch


def test_small_batch_log_variables_have_finite_source_bounds():
    model = gdplib.small_batch.build_model()

    for j in model.j:
        assert model.n[j].lb == 0
        assert model.n[j].ub == pytest.approx(math.log(len(model.k)))

    for i in model.i:
        expected_batch_size_ub = min(
            math.log(value(model.vupp) / value(model.s[i, j])) for j in model.j
        )
        expected_cycle_time_ub = (
            math.log(value(model.h) / value(model.q[i])) + expected_batch_size_ub
        )

        assert model.b[i].lb == 0
        assert model.b[i].ub == pytest.approx(expected_batch_size_ub)
        assert model.tl[i].lb == 0
        assert model.tl[i].ub == pytest.approx(expected_cycle_time_ub)


@pytest.mark.parametrize("transformation", ["gdp.bigm", "gdp.hull"])
def test_small_batch_reformulates_with_bounded_log_variables(transformation):
    model = gdplib.small_batch.build_model()

    pyo.TransformationFactory(transformation).apply_to(model)

    assert not any(model.component_data_objects(pyo.LogicalConstraint, active=True))
    assert not any(model.component_data_objects(Disjunction, active=True))
