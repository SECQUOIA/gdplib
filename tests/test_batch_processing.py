import math

import pytest
import pyomo.environ as pyo
from pyomo.environ import value
from pyomo.gdp import Disjunction

import gdplib.batch_processing


def test_batch_processing_cycle_time_log_has_finite_source_bounds():
    model = gdplib.batch_processing.build_model()

    for i in model.PRODUCTS:
        expected_lb = max(
            math.log(value(model.ProcessingTime[i, j]))
            - model.batchSize_log[i, j].ub
            - value(model.unitsOutOfPhaseUB[j])
            for j in model.STAGES
        )
        expected_ub = math.log(
            value(model.HorizonTime) / value(model.ProductionAmount[i])
        )

        assert math.isfinite(model.cycleTime_log[i].lb)
        assert math.isfinite(model.cycleTime_log[i].ub)
        assert model.cycleTime_log[i].lb == pytest.approx(expected_lb)
        assert model.cycleTime_log[i].ub == pytest.approx(expected_ub)


@pytest.mark.parametrize("transformation", ["gdp.bigm", "gdp.hull"])
def test_batch_processing_reformulates_with_bounded_cycle_time(transformation):
    model = gdplib.batch_processing.build_model()

    pyo.TransformationFactory(transformation).apply_to(model)

    assert not any(model.component_data_objects(pyo.LogicalConstraint, active=True))
    assert not any(model.component_data_objects(Disjunction, active=True))
