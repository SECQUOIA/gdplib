# Repository Instructions for Coding Agents

These instructions apply to the whole repository.

## Project Scope

- GDPlib is a Python model library for Generalized Disjunctive Programming examples and benchmarks built with Pyomo.
- GDPlib is licensed under the BSD 3-clause license. Keep new contributions
  compatible with that license.
- Supported Python versions are 3.10, 3.11, and 3.12. Keep `pyproject.toml`, `setup.py`, CI, and environment manifests aligned when changing support.
- Treat optimization solvers as optional external tools. Do not make model construction, imports, or default tests require GAMS, IPOPT, or another solver unless the task explicitly asks for solver-specific behavior.
- Models often represent chemical-engineering or operations-research
  optimization problems. Preserve the model's reference source and benchmarking
  intent when making changes.

## Environment

- Prefer the committed Pixi environment for reproducible Linux development
  (`linux-64`):
  ```bash
  pixi install
  pixi run test
  pixi run lint
  ```
- If `pixi` is not on `PATH`, check whether a local Pixi binary exists (for
  example `$HOME/.pixi/bin/pixi`) before falling back to the pip workflow.
- On macOS or Windows, use the pip workflow below unless the task explicitly
  expands Pixi platform support and regenerates `pixi.lock`.
- Do not add Pixi platforms casually. Adding `osx-64`, `osx-arm64`, `win-64`,
  or another platform should also regenerate `pixi.lock` and verify
  `pixi install`, `pixi run test`, and `pixi run lint` on that platform, or be
  tracked as follow-up work.
- The legacy pip workflow is still supported:
  ```bash
  pip install -r requirements.txt
  pip install -r requirements-dev.txt
  pip install -e .
  ```
- Do not commit generated environments, caches, coverage output, or benchmark output.

## Testing and Linting

- Run targeted tests for the code you change, then run broader tests when practical.
- Default test command:
  ```bash
  pytest tests/ -v --tb=short
  ```
- For model import and construction changes, include:
  ```bash
  pytest tests/test_module_imports.py -v --tb=short
  ```
- Match CI formatting and linting:
  ```bash
  black -S -C --target-version py310 --check --diff .
  flake8 gdplib/ --count --select=E9,F63,F7,F82 --show-source --statistics
  flake8 gdplib/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
  typos --config ./.github/workflows/typos.toml
  ```

## Coding and Documentation Style

- Follow the style of the surrounding module, while keeping changed Python code
  compatible with the repository Black and flake8 checks.
- Use type hints where they clarify public helper APIs, but do not introduce
  large annotation-only refactors in model files.
- Keep docstrings accurate. Use NumPy/SciPy-style `Parameters` and `Returns`
  sections only for actual callable signatures; document constructor-created
  model attributes as implementation notes, module documentation, or model
  README content.
- For new or substantially changed models, include relevant mathematical
  formulation details, usage examples, and source references in the model
  README or module documentation.
- Use `### Solution` consistently for README solution sections. For a single
  verified objective value, prefer:
  `Best known objective value: <value> (optimal)`. For multiple formulations or
  instances, use `Best known objective values:` followed by one entry per case.
  If a value is not proven optimal, say "best known" rather than "optimal" and
  include the solver status or gap when known.
- When adding or changing documented solution values, try to reproduce them with
  the model's existing solve path or a focused Pyomo script, preferably inside
  the Pixi environment. Record the model instance/formulation, transformation,
  solver interface, solver, termination condition, objective value, and any gap
  or infeasibility evidence in the PR or issue discussion.
- Keep solver-backed solution verification separate from default model
  construction tests. GAMS, BARON, IPOPT, and similar tools are optional, so do
  not make default imports, model construction, or CI tests depend on them
  unless the task explicitly asks for solver-specific behavior.
- If a model cannot be solved because of a missing solver, timeout, infeasible
  result, or runtime error, search existing GitHub issues first. Update the
  relevant issue when one exists; otherwise create a new issue with the exact
  command, environment, solver stack, termination condition or traceback, and
  model/formulation attempted.
- A future structured solution-record format is tracked in
  https://github.com/SECQUOIA/gdplib/issues/105. The intended direction is
  inspired by MINLPLib solution metadata: objective sense/value, best primal and
  dual bounds, infeasibility, solver and solve metadata, model instance or
  formulation, and optionally nonzero primal variable values.
- For Pyomo code, follow the import style already used by the local module.

## Model and API Conventions

- Model packages live under `gdplib/<model_name>/`.
- New model packages should include a `README.md`, expose a callable `build_model()` from `gdplib/<model_name>/__init__.py`, and be imported from `gdplib/__init__.py`.
- `build_model()` should construct and return a Pyomo model. Avoid solving, network access, filesystem writes, or other side effects during model construction.
- Package modules may use relative imports. Scripts intended to run as `__main__` should use absolute imports, as described in the README.
- Keep data files package-local and access them with paths relative to the module file.
- Preserve existing public import paths and model-builder names unless the user explicitly asks for an API break.
- When a change claims to simplify or reduce a GDP/Pyomo model, distinguish
  between Pyomo modeling-layer objects and the solver-facing transformed model.
  Report both when relevant: logical components such as `BooleanVarData`,
  numeric variables, binary variables, constraints, disjunctions, disjuncts,
  and transformed model size after the intended GDP transformation.
- For logical or GDP rewrites, test the supported transformation path in the
  committed environment. Prefer direct `gdp.bigm` smoke tests when that is the
  intended path; do not add optional dependencies such as `sympy` solely to test
  a transformation path that the project does not otherwise require.
- For scalable model changes, verify the documented/default instance and at
  least one larger instance when practical. If semantics should be unchanged,
  compare transformed model counts or generated algebraic representations where
  feasible.

## Packaging

- Runtime dependencies belong in `pyproject.toml`, `setup.py`, and `requirements.txt`.
- Development-only tools belong in `requirements-dev.txt` and the Pixi environment.
- Add solver packages only to optional environments or documentation unless they are required for default imports and tests.
- Packaging tests can regenerate `gdplib/_version.py` and the editable package
  entry in `pixi.lock`. Do not commit that churn unless the task is explicitly
  about versioning, release metadata, or regenerating the Pixi lock.
