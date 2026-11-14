from math import prod
from types import SimpleNamespace

import pytest
import pyomo.environ as pe
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.gdp import Disjunct, Disjunction

import gdplib.methanol_stochastic.methanol_stochastic as methanol_stochastic

EQUIPMENT = (3, 4, 6, 16, 17, 19, 5, 12, 18, 7, 8, 14, 15, 11)
LEVEL_PROBABILITY = {"low": 0.3, "mid": 0.4, "high": 0.3}


@pytest.fixture(scope="module")
def models():
    return {
        redundancy: methanol_stochastic.build_model(redundancy=redundancy)
        for redundancy in (False, True)
    }


@pytest.mark.parametrize("redundancy", [False, True])
def test_build_model_structure(models, redundancy):
    m = models[redundancy]

    assert m.objective.sense == pe.minimize
    assert hasattr(m, "profit")
    assert len(m.scenarios) == 9
    assert len(list(m.component_data_objects(Disjunction, active=True))) == 4
    assert len(list(m.component_data_objects(Disjunct, active=True))) == 8
    first_disjunct = next(m.component_data_objects(Disjunct, active=True))
    assert len(first_disjunct.BigM) > 0

    # First-stage design variables are shared by all scenarios ...
    assert not m.cheap_reactor.reactor_10.reactor_volume.is_indexed()
    assert not m.expensive_reactor.reactor_9.reactor_volume.is_indexed()
    for eq in EQUIPMENT:
        equipment = getattr(m, "equipment_" + str(eq))
        assert not equipment.capacity.is_indexed()
        assert not equipment.availability.is_indexed()
    # ... while operating variables are indexed by scenario.
    assert m.cheap_reactor.reactor_10.temp.index_set() is m.scenarios
    assert m.cooler_7.heat_duty.index_set() is m.scenarios

    binaries = [
        v
        for v in m.component_data_objects(pe.Var, descend_into=(pe.Block, Disjunct))
        if v.is_binary()
    ]
    # 8 disjunct indicators, plus 3 candidate parallel units per compressor
    # and 4 per heat exchanger and valve with redundancy.
    assert len(binaries) == (8 + 6 * 3 + 8 * 4 if redundancy else 8)


def test_scenario_data(models):
    m = models[False]

    assert sum(pe.value(m.probs[s]) for s in m.scenarios) == pytest.approx(1)
    for s in m.scenarios:
        demand_level, price_level = s.split()
        assert pe.value(m.probs[s]) == pytest.approx(
            LEVEL_PROBABILITY[demand_level] * LEVEL_PROBABILITY[price_level]
        )
        assert (
            pe.value(m.demands[s])
            == {"low": 0.8, "mid": 1.0, "high": 1.2}[demand_level]
        )
        assert (
            pe.value(m.electricity_cost[s])
            == {"low": 200, "mid": 255, "high": 300}[price_level]
        )


def test_system_availability_without_redundancy():
    m = methanol_stochastic.build_model()
    model_data = methanol_stochastic.MethanolModelStochastic()
    m.cheap_feed_disjunct.indicator_var.fix(True)
    m.expensive_feed_disjunct.indicator_var.fix(False)
    m.cheap_reactor.indicator_var.fix(True)
    m.expensive_reactor.indicator_var.fix(False)
    m.single_stage_feed_compressor_disjunct.indicator_var.fix(False)
    m.two_stage_feed_compressor_disjunct.indicator_var.fix(True)
    m.single_stage_recycle_compressor_disjunct.indicator_var.fix(True)
    m.two_stage_recycle_compressor_disjunct.indicator_var.fix(False)
    pe.TransformationFactory("gdp.fix_disjuncts").apply_to(m)

    for c in m.component_data_objects(pe.Constraint, active=True):
        if c.local_name in ("available_con", "zero_capacity_con"):
            fbbt(c)
    fbbt(m.system_available_con)

    installed = (7, 8, 11, 12, 14, 15, 4, 5, 6, 16)
    expected = prod(model_data.probability[eq][0] for eq in installed)
    assert m.system_availability.lb == pytest.approx(expected)
    assert m.system_availability.ub == pytest.approx(expected)
    assert m.equipment_3.capacity.bounds == (0, 0)


@pytest.mark.parametrize(
    "selection", [(1, 0, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1)], ids=str
)
def test_redundant_units_availability_and_capacity(models, selection):
    m = models[True]
    model_data = methanol_stochastic.MethanolModelStochastic(redundancy=True)
    equipment = m.equipment_3
    selected_block = m.single_stage_feed_compressor_disjunct.equipment_3
    capacity = 40.0

    equipment.capacity.set_value(capacity)
    for i, y in zip(equipment.I, selection):
        equipment.ys[i].set_value(y)
        equipment.capacity_ys[i].set_value(capacity * y)
    expected = 1 - prod(1 - p * y for p, y in zip(model_data.probability[3], selection))
    equipment.availability.set_value(expected)

    assert pe.value(selected_block.available_con.body) == pytest.approx(
        pe.value(selected_block.available_con.upper)
    )
    assert pe.value(selected_block.at_least_one_con.body) >= 1
    for name in ("McCormick_1_con", "McCormick_2_con", "McCormick_3_con"):
        for c in getattr(selected_block, name).values():
            lower, body, upper = (
                pe.value(c.lower),
                pe.value(c.body),
                pe.value(c.upper),
            )
            assert lower is None or body >= lower - 1e-9
            assert upper is None or body <= upper + 1e-9
    for c in selected_block.McCormick_4_con.values():
        assert pe.value(c.body) <= pe.value(c.upper) + 1e-9


@pytest.mark.parametrize("redundancy", [False, True])
def test_bigm_transformation(redundancy):
    m = methanol_stochastic.build_model(redundancy=redundancy)

    pe.TransformationFactory("gdp.bigm").apply_to(m)

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

    monkeypatch.setattr(methanol_stochastic.pe, "SolverFactory", fake_solver_factory)

    model, results = methanol_stochastic.solve_with_gdp_opt(
        redundancy=True,
        tee=False,
        minlp_solver="gams",
        minlp_solver_args={"solver": "dicopt"},
        return_results=True,
        print_results=False,
    )

    assert captured["solver_name"] == "gdpopt"
    assert captured["model"] is model
    assert hasattr(model, "equipment_3") and hasattr(model.equipment_3, "ys")
    assert captured["kwargs"] == {
        "algorithm": "LOA",
        "mip_solver": "gams",
        "nlp_solver": "gams",
        "tee": False,
        "minlp_solver": "gams",
        "minlp_solver_args": {"solver": "dicopt"},
    }
    assert results.solver.termination_condition == "optimal"
