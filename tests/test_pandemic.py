"""
Tests for the pandemic GDP model (event-constrained optimal disease control).
"""

import pytest
from pyomo.environ import (
    ConcreteModel,
    TransformationFactory,
    Var,
    Objective,
    Constraint,
)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.dae import ContinuousSet

from gdplib.pandemic import build_model

NUM_TIMES = 11


@pytest.fixture(scope='module')
def model():
    return build_model(num_times=NUM_TIMES)


class TestBuildModel:
    def test_returns_concrete_model(self, model):
        assert isinstance(model, ConcreteModel)

    def test_time_set(self, model):
        # num_times equidistant + 10 extra early points, duplicates removed
        assert len(model.t) > NUM_TIMES

    def test_state_variables_present(self, model):
        for var in ('s', 'e', 'i', 'r', 'u'):
            assert isinstance(getattr(model, var), Var)

    def test_objective_present(self, model):
        assert isinstance(model.obj, Objective)

    def test_disjunctions_present(self, model):
        assert isinstance(model.constrs_satisfy_or_not, Disjunction)
        assert len(model.constrs_satisfy_or_not) == len(model.t)

    def test_event_constraint_present(self, model):
        assert isinstance(model.event_constr, Constraint)

    def test_no_dae_warning(self, recwarn):
        # nfe=len(ts)-1 must suppress the "More finite elements were found" Pyomo warning
        build_model(num_times=NUM_TIMES)
        dae_warnings = [
            w for w in recwarn.list if 'finite elements' in str(w.message).lower()
        ]
        assert len(dae_warnings) == 0


class TestTransformationPath:
    def test_bigm(self):
        m = build_model(num_times=NUM_TIMES)
        TransformationFactory('gdp.bigm').apply_to(m)

    def test_hull(self):
        m = build_model(num_times=NUM_TIMES)
        TransformationFactory('gdp.hull').apply_to(m)
