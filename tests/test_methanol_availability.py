from math import prod
from types import SimpleNamespace

import pytest
import pyomo.environ as pe
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.core.expr.visitor import identify_variables
from pyomo.gdp import Disjunct, Disjunction

import gdplib.methanol_availability.methanol_availability as methanol_availability

EQUIPMENT = (3, 4, 6, 16, 17, 19, 5, 12, 18, 7, 8, 14, 15, 11)
ALWAYS_INSTALLED = (7, 8, 11, 12, 14, 15)
FEED_COMPRESSOR_UNITS = {False: (3,), True: (4, 5, 6)}
RECYCLE_COMPRESSOR_UNITS = {False: (16,), True: (17, 18, 19)}


def _fix_configuration(m, two_stage_feed, two_stage_recycle):
    m.cheap_feed_disjunct.indicator_var.fix(True)
    m.expensive_feed_disjunct.indicator_var.fix(False)
    m.cheap_reactor.indicator_var.fix(True)
    m.expensive_reactor.indicator_var.fix(False)
    m.single_stage_feed_compressor_disjunct.indicator_var.fix(not two_stage_feed)
    m.two_stage_feed_compressor_disjunct.indicator_var.fix(two_stage_feed)
    m.single_stage_recycle_compressor_disjunct.indicator_var.fix(not two_stage_recycle)
    m.two_stage_recycle_compressor_disjunct.indicator_var.fix(two_stage_recycle)
    pe.TransformationFactory("gdp.fix_disjuncts").apply_to(m)


def test_build_model_structure():
    m = methanol_availability.build_model()

    assert m.objective.sense == pe.minimize
    assert hasattr(m, "profit")
    assert len(list(m.component_data_objects(Disjunction, active=True))) == 4
    assert len(list(m.component_data_objects(Disjunct, active=True))) == 8
    for eq in EQUIPMENT:
        equipment = getattr(m, "equipment_" + str(eq))
        assert equipment.capacity.bounds == (0, 60)
        assert equipment.availability.bounds == (0, 1)
    assert m.expensive_reactor.reactor_9.reactor_volume.bounds == (30, 150)
    assert m.cheap_reactor.reactor_10.reactor_volume.bounds == (30, 150)
    first_disjunct = next(m.component_data_objects(Disjunct, active=True))
    assert len(first_disjunct.BigM) > 0


def test_objective_uses_binary_indicator_variables():
    m = methanol_availability.build_model()

    objective_vars = {id(v) for v in identify_variables(m.objective.expr)}
    for disjunct in (
        m.single_stage_feed_compressor_disjunct,
        m.two_stage_feed_compressor_disjunct,
        m.single_stage_recycle_compressor_disjunct,
        m.two_stage_recycle_compressor_disjunct,
    ):
        assert id(disjunct.binary_indicator_var) in objective_vars
    assert id(m.system_availability) in objective_vars


@pytest.mark.parametrize("two_stage_feed", [False, True])
@pytest.mark.parametrize("two_stage_recycle", [False, True])
def test_system_availability_matches_installed_units(two_stage_feed, two_stage_recycle):
    m = methanol_availability.build_model()
    model_data = methanol_availability.MethanolModelAvailability()
    _fix_configuration(m, two_stage_feed, two_stage_recycle)

    for c in m.component_data_objects(pe.Constraint, active=True):
        if c.local_name in ("available_con", "zero_capacity_con"):
            fbbt(c)
    fbbt(m.system_available_con)

    installed = (
        ALWAYS_INSTALLED
        + FEED_COMPRESSOR_UNITS[two_stage_feed]
        + RECYCLE_COMPRESSOR_UNITS[two_stage_recycle]
    )
    expected = prod(model_data.probability[eq] for eq in installed)
    assert m.system_availability.lb == pytest.approx(expected)
    assert m.system_availability.ub == pytest.approx(expected)
    for eq in (
        FEED_COMPRESSOR_UNITS[not two_stage_feed]
        + RECYCLE_COMPRESSOR_UNITS[not two_stage_recycle]
    ):
        equipment = getattr(m, "equipment_" + str(eq))
        assert equipment.availability.bounds == (1, 1)
        assert equipment.capacity.bounds == (0, 0)


@pytest.mark.parametrize("transformation", ["gdp.bigm", "gdp.hull"])
def test_gdp_transformations(transformation):
    m = methanol_availability.build_model()

    pe.TransformationFactory(transformation).apply_to(m)

    assert not any(m.component_data_objects(Disjunction, active=True))


def test_solve_with_gdp_opt_passes_solver_options(monkeypatch):
    captured = {}

    class _FakeGDPopt:
        def solve(self, model, **kwargs):
            captured["model"] = model
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                solver=SimpleNamespace(termination_condition="optimal")
            )

    def fake_solver_factory(name):
        captured["solver_name"] = name
        return _FakeGDPopt()

    monkeypatch.setattr(methanol_availability.pe, "SolverFactory", fake_solver_factory)

    model, results = methanol_availability.solve_with_gdp_opt(
        tee=False,
        mip_solver_args={"solver": "gurobi"},
        nlp_solver_args={"solver": "conopt4"},
        time_limit=60,
        return_results=True,
        print_results=False,
    )

    assert captured["solver_name"] == "gdpopt"
    assert captured["model"] is model
    assert captured["kwargs"] == {
        "algorithm": "LOA",
        "mip_solver": "gams",
        "nlp_solver": "gams",
        "tee": False,
        "mip_solver_args": {"solver": "gurobi"},
        "nlp_solver_args": {"solver": "conopt4"},
        "time_limit": 60,
    }
    assert results.solver.termination_condition == "optimal"
