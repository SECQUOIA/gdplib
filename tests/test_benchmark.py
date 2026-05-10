import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

benchmark_path = Path(__file__).resolve().parents[1] / "benchmark.py"
benchmark_spec = spec_from_file_location("benchmark_module", benchmark_path)
assert benchmark_spec is not None
assert benchmark_spec.loader is not None
benchmark_module = module_from_spec(benchmark_spec)
benchmark_spec.loader.exec_module(benchmark_module)


class DummyModel:
    def clone(self):
        return self


class DummyTransformation:
    def apply_to(self, model):
        return None


class FailingSolver:
    def solve(self, *args, **kwargs):
        raise RuntimeError("solver failed")


def test_benchmark_restores_stdout_when_direct_solver_raises(tmp_path, monkeypatch):
    original_stdout = sys.stdout
    monkeypatch.setattr(
        benchmark_module,
        "TransformationFactory",
        lambda strategy: DummyTransformation(),
    )
    monkeypatch.setattr(
        benchmark_module, "SolverFactory", lambda subsolver: FailingSolver()
    )

    with pytest.raises(RuntimeError, match="solver failed"):
        benchmark_module.benchmark(DummyModel(), "gdp.bigm", 1, str(tmp_path))

    assert sys.stdout is original_stdout
    assert (tmp_path / "gdp.bigm_scip.log").exists()
    assert not (tmp_path / "gdp.bigm_scip.json").exists()


def test_benchmark_restores_stdout_when_gdpopt_solver_raises(tmp_path, monkeypatch):
    original_stdout = sys.stdout
    monkeypatch.setattr(
        benchmark_module, "SolverFactory", lambda strategy: FailingSolver()
    )

    with pytest.raises(RuntimeError, match="solver failed"):
        benchmark_module.benchmark(DummyModel(), "gdpopt.loa", 1, str(tmp_path))

    assert sys.stdout is original_stdout
    assert (tmp_path / "gdpopt.loa_scip.log").exists()
    assert not (tmp_path / "gdpopt.loa_scip.json").exists()


def test_benchmark_rejects_unknown_strategy(tmp_path):
    with pytest.raises(
        ValueError, match="Unknown benchmark strategy: no.such.strategy"
    ):
        benchmark_module.benchmark(DummyModel(), "no.such.strategy", 1, str(tmp_path))
