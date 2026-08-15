"""
Tests for the grid GDP model (event-constrained optimal power flow).
"""

import pytest
import numpy as np
from math import ceil
from pyomo.environ import (
    ConcreteModel,
    TransformationFactory,
    BooleanVar,
    Var,
    Objective,
    LogicalConstraint,
    value,
)
from pyomo.gdp import Disjunction

from gdplib.grid import build_model

NUM_SAMPLES = 10


@pytest.fixture(scope='module')
def model():
    return build_model(num_samples=NUM_SAMPLES)


class TestBuildModel:
    def test_returns_concrete_model(self, model):
        assert isinstance(model, ConcreteModel)

    def test_index_sets(self, model):
        assert len(model.K) == NUM_SAMPLES
        assert len(model.G) == 5
        assert len(model.L) == 20

    def test_variables_present(self, model):
        assert isinstance(model.z_gen, Var)
        assert isinstance(model.z_line, Var)
        assert isinstance(model.d_gen, Var)
        assert isinstance(model.d_line, Var)

    def test_event_boolean_var(self, model):
        assert isinstance(model.w, BooleanVar)
        assert len(model.w) == NUM_SAMPLES

    def test_objective_present(self, model):
        assert isinstance(model.obj, Objective)

    def test_disjunctions_present(self, model):
        assert isinstance(model.gen_constrs_satisfy_or_not, Disjunction)
        assert isinstance(model.line_constrs_satisfy_or_not, Disjunction)
        assert len(model.gen_constrs_satisfy_or_not) == 5 * NUM_SAMPLES
        assert len(model.line_constrs_satisfy_or_not) == 20 * NUM_SAMPLES

    def test_event_logical_constraint(self, model):
        assert isinstance(model.event_logic, LogicalConstraint)
        assert len(model.event_logic) == NUM_SAMPLES

    def test_chance_constraint_threshold(self, model):
        assert value(model.min_constrs) == ceil(0.9 * NUM_SAMPLES)

    def test_does_not_mutate_global_rng(self):
        rng_state_before = np.random.get_state()[1].copy()
        build_model(num_samples=NUM_SAMPLES)
        rng_state_after = np.random.get_state()[1]
        assert np.array_equal(rng_state_before, rng_state_after)


class TestTransformationPath:
    def test_logical_to_linear(self):
        m = build_model(num_samples=NUM_SAMPLES)
        TransformationFactory('core.logical_to_linear').apply_to(m)

    def test_bigm_after_logical_to_linear(self):
        m = build_model(num_samples=NUM_SAMPLES)
        TransformationFactory('core.logical_to_linear').apply_to(m)
        TransformationFactory('gdp.bigm').apply_to(m)

    def test_hull_after_logical_to_linear(self):
        m = build_model(num_samples=NUM_SAMPLES)
        TransformationFactory('core.logical_to_linear').apply_to(m)
        TransformationFactory('gdp.hull').apply_to(m)


class TestParameterValidation:
    def test_invalid_num_samples_zero(self):
        with pytest.raises(ValueError, match='num_samples'):
            build_model(num_samples=0)

    def test_invalid_num_samples_negative(self):
        with pytest.raises(ValueError, match='num_samples'):
            build_model(num_samples=-1)

    def test_invalid_num_samples_float(self):
        with pytest.raises(ValueError, match='num_samples'):
            build_model(num_samples=10.0)

    def test_invalid_active_gens_out_of_range(self):
        with pytest.raises(ValueError, match='active_gens'):
            build_model(active_gens=6, num_samples=NUM_SAMPLES)

    def test_invalid_active_gens_float(self):
        with pytest.raises(ValueError, match='active_gens'):
            build_model(active_gens=2.0, num_samples=NUM_SAMPLES)

    def test_invalid_active_lines_out_of_range(self):
        with pytest.raises(ValueError, match='active_lines'):
            build_model(active_lines=21, num_samples=NUM_SAMPLES)

    def test_invalid_active_lines_float(self):
        with pytest.raises(ValueError, match='active_lines'):
            build_model(active_lines=5.0, num_samples=NUM_SAMPLES)

    def test_valid_boundary_values(self):
        m = build_model(active_gens=0, active_lines=0, num_samples=1)
        assert isinstance(m, ConcreteModel)
