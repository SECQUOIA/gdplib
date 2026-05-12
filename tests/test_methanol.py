from types import SimpleNamespace

from pyomo.gdp import Disjunct

import gdplib.methanol.methanol as methanol


class _FakeGDPopt:
    def __init__(self, captured):
        self._captured = captured

    def solve(self, model, **kwargs):
        self._captured["model"] = model
        self._captured["kwargs"] = kwargs
        return SimpleNamespace(solver=SimpleNamespace(termination_condition="optimal"))


def _patch_gdpopt(monkeypatch):
    captured = {}

    def fake_solver_factory(name):
        captured["solver_name"] = name
        return _FakeGDPopt(captured)

    monkeypatch.setattr(methanol.pe, "SolverFactory", fake_solver_factory)
    return captured


def test_solve_with_gdp_opt_passes_solver_options(monkeypatch):
    captured = _patch_gdpopt(monkeypatch)

    model, results = methanol.solve_with_gdp_opt(
        tee=False,
        mip_solver_args={"solver": "gurobi"},
        nlp_solver_args={"solver": "baron"},
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
        "nlp_solver_args": {"solver": "baron"},
        "time_limit": 60,
    }
    assert results.solver.termination_condition == "optimal"


def test_solve_with_gdp_opt_preserves_model_return(monkeypatch):
    captured = _patch_gdpopt(monkeypatch)

    model = methanol.solve_with_gdp_opt(tee=False, print_results=False)

    assert model is captured["model"]
    assert "mip_solver_args" not in captured["kwargs"]
    assert "nlp_solver_args" not in captured["kwargs"]
    first_disjunct = next(model.component_data_objects(Disjunct, active=True))
    assert len(first_disjunct.BigM) > 0


def test_build_model_exposes_profit_and_minimizes_negative_profit():
    model = methanol.build_model()

    assert model.objective.sense == methanol.pe.minimize
    assert hasattr(model, "profit")
