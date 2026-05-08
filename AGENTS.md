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
- On macOS or Windows, use the pip workflow below unless the task explicitly
  expands Pixi platform support and regenerates `pixi.lock`.
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
- For Pyomo code, follow the import style already used by the local module.

## Model and API Conventions

- Model packages live under `gdplib/<model_name>/`.
- New model packages should include a `README.md`, expose a callable `build_model()` from `gdplib/<model_name>/__init__.py`, and be imported from `gdplib/__init__.py`.
- `build_model()` should construct and return a Pyomo model. Avoid solving, network access, filesystem writes, or other side effects during model construction.
- Package modules may use relative imports. Scripts intended to run as `__main__` should use absolute imports, as described in the README.
- Keep data files package-local and access them with paths relative to the module file.
- Preserve existing public import paths and model-builder names unless the user explicitly asks for an API break.

## Packaging

- Runtime dependencies belong in `pyproject.toml`, `setup.py`, and `requirements.txt`.
- Development-only tools belong in `requirements-dev.txt` and the Pixi environment.
- Add solver packages only to optional environments or documentation unless they are required for default imports and tests.
