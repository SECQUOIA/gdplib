# Repository Instructions for Coding Agents

These instructions apply to the whole repository.

- Keep this file curated: fold new lessons into the most specific existing
  section, replace obsolete detail, and prefer reusable guidance.

## Project Scope

- GDPlib is a Python model library for Generalized Disjunctive Programming examples and benchmarks built with Pyomo.
- GDPlib is licensed under the BSD 3-clause license. Keep new contributions
  compatible with that license.
- Supported Python versions are 3.10, 3.11, and 3.12. Keep `pyproject.toml`, `setup.py`, CI, and environment manifests aligned when changing support.
- Treat optimization solvers as optional external tools. Do not make model construction, imports, or default tests require GAMS, IPOPT, or another solver unless the task explicitly asks for solver-specific behavior.
- Direct Gurobi Python access is available through the optional Pixi
  environment `gurobi`, which adds `gurobipy` without making it part of the
  default development environment. Keep Gurobi license paths local; document
  `GRB_LICENSE_FILE` rather than hard-coding machine-specific paths.
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
- When Pixi is available, run tests, lint, and documentation generators through
  `pixi run ...` so the Pixi-managed `.venv` supplies the Python version and
  dependencies. Avoid using the system Python for PR verification unless Pixi is
  unavailable and the fallback is called out explicitly.
- If `pixi` is not on `PATH`, check whether a local Pixi binary exists (for
  example `$HOME/.pixi/bin/pixi`) before falling back to the pip workflow.
- The committed Pixi support surface is exactly the platform list in
  `pixi.toml` and the matching entries in `pixi.lock`; it is currently
  `linux-64`. Keep README, `pixi.toml`, and `pixi.lock` aligned when changing
  that policy.
- On macOS or Windows, use the pip workflow below unless the task explicitly
  expands Pixi platform support.
- Do not add Pixi platforms casually. Adding `osx-64`, `osx-arm64`, `win-64`,
  or another platform must also regenerate `pixi.lock` and verify
  `pixi install`, `pixi run test`, and `pixi run lint` on that platform before
  committing manifest or lock-file changes. If the platform cannot be verified,
  leave the manifest and lock unchanged and document the work as follow-up.
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
- When changing `build_model()` tests, make unexpected construction failures
  fail the test. Inspect builder signatures before calling models that may
  require arguments; only skip required-argument builders or unavailable
  optional solvers. Do not catch `Exception` or `TypeError` from
  `build_model()` and turn it into a skip.
- For line-ending-heavy PRs, inspect `git diff --ignore-cr-at-eol`, still run
  `git diff --check`, and avoid unrelated normalization.
- For GitHub issue or PR inspection through `gh`, prefer explicit `--json`
  field lists when default views request deprecated GraphQL fields such as
  Projects Classic `projectCards`. GitHub rejects approvals from the PR author;
  submit a `COMMENT` review instead and note that an eligible reviewer is still
  required.
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
- For model documentation audits, prefer documentation, module docstrings, and
  inline comments over behavior changes. Do not commit source PDFs, extracted
  paper text, or other local research artifacts unless they are intentional
  package data.
- Trace units, scaling, constants, and source references to the paper or a
  clearly named related implementation. If a quantity is only known in source
  scaling, label it `[source scale]` or `[dimensionless]`; distinguish
  equivalent formulations from merely useful references.
- If `typos` flags a correct domain/project name, add the exact accepted term
  to `.github/workflows/typos.toml` rather than misspelling, avoiding the
  correct name, or renaming a public API/model component solely for spell-check.
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
- When escalating a Pyomo or solver behavior upstream, include a compact MWE,
  the exact environment and solver roles, and links to the affected GDPlib
  issue, PR, and benchmark comment. Prefer a solver-free or open-solver MWE
  when it can isolate the behavior; otherwise keep licensed-solver evidence as
  downstream context rather than the only reproduction.
- A future structured solution-record format is tracked in
  https://github.com/SECQUOIA/gdplib/issues/105. The intended direction is
  inspired by MINLPLib solution metadata: objective sense/value, best primal and
  dual bounds, infeasibility, solver and solve metadata, model instance or
  formulation, and optionally nonzero primal variable values.
- For Pyomo code, follow the import style already used by the local module.

## Pyomo Deprecations and Warnings

- When fixing Pyomo or Pyomo.GDP deprecations, keep the change narrowly tied
  to the deprecated API. Do not bundle unrelated warning classes, numerical
  initialization changes, or formulation changes into the same PR unless they
  are required for correctness.
- Set `BooleanVar` and GDP `indicator_var` values with `True`/`False`, not
  numeric `1`/`0`. When associating a `BooleanVar` with a disjunct indicator,
  use the indicator's associated binary variable instead of the Boolean
  `indicator_var` itself.
- For GDPOpt dispatcher calls through `SolverFactory("gdpopt").solve(...)`,
  use `algorithm=...` instead of the deprecated top-level `strategy=...`.
  Verify current Pyomo option names in the committed environment before
  updating less common GDPOpt options.
- For ordered Pyomo `Set` positional access, use `set.at(index)` when the
  intent is positional lookup. Do not rely on deprecated `set[index]` behavior.
- Add focused warning-regression tests for deprecation cleanup. Prefer
  capturing Pyomo warning logs for the smallest relevant build or transform
  path, and assert against the specific deprecated warning patterns being
  removed. Do not make tests fail on unrelated existing warnings; track those
  separately in the relevant model issue.
- If a warning cleanup touches model construction, GDP transformations, or
  solver-facing expressions, compare the affected benchmark instances against
  existing baseline artifacts when practical. Record objective values,
  termination conditions, bounds, and unchanged failures in the PR or issue
  discussion, but do not commit generated benchmark outputs.
- Treat Pyomo/FBBT runtime warnings as triage signals, not automatic model
  defects. First fix invalid domains, missing finite bounds, or unsafe algebra
  in the model. If the model has valid bounds and the warning is isolated to
  Pyomo internals such as scalar interval arithmetic, report a focused upstream
  MWE instead of hiding it with unrelated model changes.

## Benchmark Campaigns

- Use the committed benchmark CLI for solver-backed campaigns:
  ```bash
  pixi run gdplib-benchmark preflight
  pixi run gdplib-benchmark run
  pixi run gdplib-benchmark warnings
  ```
- Treat `benchmark_cases/*.csv` and `benchmark_cases/*.json` as reusable,
  version-controlled campaign inputs, not benchmark results. Commit a case file
  only when it captures a generally useful campaign configuration.
- Do not commit generated benchmark artifacts. Keep run outputs under ignored
  paths such as `gdplib/*/benchmark_result/`, `benchmark_runs/`, and
  `benchmark_summary/`. Record important results in the relevant PR or issue
  discussion instead.
- Benchmark and reporting helpers that redirect solver output must restore
  `stdout`/`stderr` even when solves fail.
- Preflight before long campaigns. Start with small, bounded runs using
  explicit `--instances`, `--strategies`, `--timelimit`, `--solver-profile`,
  and `--run-id` values, then broaden coverage once failures are understood.
- Choose solvers according to transformed model class and benchmark goal: use
  LP/MIP solvers such as Gurobi or HiGHS for direct MIP reformulations, IPOPT or
  GAMS IPOPTH for local NLP roles, DICOPT for local MINLP roles, and BARON only
  when global evidence is intended. For nonlinear GDPs that Pyomo's direct
  Gurobi writers reject, use the GAMS/Gurobi profile and say why.
- Avoid `gdpopt.enumerate` for large-disjunction models such as `biofuel`;
  compare direct `gdp.bigm` and `gdp.hull` solves first. Direct Hull through
  GAMS/Gurobi can be the best quick evidence for some large GDPs, but report the
  actual GAMS `optcr`/gap so a 1% certificate is not mistaken for a 1e-6 run.
- Interpret GDPOpt results conservatively on nonconvex models. Do not treat
  `gdpopt.loa` `optimal`, `LB = UB`, or missing-bound results as rigorous global
  certificates unless convexity and valid OA bounds are established; cross-check
  against GLOA, RIC, direct transformed solves, or explicit feasible points.
- For `gdpopt.lbb` failures, separate model issues from GDPOpt control-flow or
  solver-role issues. Known patterns include MINLP-only solvers receiving
  continuous nodes and time-limit finalization failures; isolate with role-solver
  probes or MWEs before changing the model.
- Trust solver logs over wrapper summaries when they disagree. If a GAMS/Pyomo
  wrapper reports an objective or bound but the solver log says no incumbent was
  found, report the solver-log status and note the wrapper discrepancy.
- When reporting benchmark results, include the exact command, run id,
  instance, strategy, solver interface, GAMS solver and GDPOpt role solvers,
  time limit, termination condition, objective, primal/dual bounds, gap or
  infeasibility evidence, and failure log path or traceback.
- Keep specialized diagnostics focused: use BARON `CompIIS` with symbolic GAMS
  labels for infeasibility work, keep external solver handoff packages under
  ignored paths such as `/tmp`, and use `gdplib-benchmark warnings` with the
  narrowest useful mode to capture warning/deprecation evidence.

## Model Size Reports

- `README.md` and `gdplib/*/model_size_report.md` are generated by
  `generate_model_size_report.py` and should remain reproducible from that
  script. Do not ignore generated model-size reports; commit changed report
  outputs together with generator changes.
- When changing `generate_model_size_report.py` or model construction that
  affects report output, run:
  ```bash
  pixi run python generate_model_size_report.py
  ```
  Confirm the resulting diff contains only intentional generated updates.
- Preserve the `MODEL_INSTANCES` coverage for models with multiple documented
  formulations or instances. Add labels, args, or kwargs for new cases instead
  of collapsing them to a single default `build_model()` call.
- Model-size report generation should not require optional solver stacks. If a
  model normally solves a subproblem during construction, provide and test a
  solver-free construction path for the report generator.
- For generator changes, include focused tests for idempotent README updates,
  per-model report generation, and any special-case instance or solver-free
  paths added to `MODEL_INSTANCES`.

## Model and API Conventions

- Model packages live under `gdplib/<model_name>/`.
- New model packages should include a `README.md`, expose a callable `build_model()` from `gdplib/<model_name>/__init__.py`, and be imported from `gdplib/__init__.py`.
- `build_model()` should construct and return a Pyomo model. Avoid solving, network access, filesystem writes, or other side effects during model construction.
- Package modules may use relative imports. Scripts intended to run as `__main__` should use absolute imports, as described in the README.
- Keep data files package-local and access them with paths relative to the module file.
- Preserve public import paths, model-builder names, and Pyomo model component
  names such as `m.foo` unless the user explicitly asks for an API break.
- When a change claims to simplify or reduce a GDP/Pyomo model, distinguish
  between Pyomo modeling-layer objects and the solver-facing transformed model.
  Report both when relevant: logical components such as `BooleanVarData`,
  numeric variables, binary variables, constraints, disjunctions, disjuncts,
  and transformed model size after the intended GDP transformation.
- For logical or GDP rewrites, test the supported transformation path in the
  committed environment. Prefer direct `gdp.bigm` smoke tests when that is the
  intended path; do not add optional dependencies such as `sympy` solely to test
  a transformation path that the project does not otherwise require.
- When model semantics require exactly one disjunct, declare the `Disjunction`
  with `xor=True`; do not rely only on a separate `LogicalConstraint` because
  `gdp.hull` requires XOR disjunctions.
- For algebraic GDP reformulations, add deterministic equivalence tests on
  representative initialized data and smoke-test supported transformations such
  as `gdp.bigm` and `gdp.hull`. Keep this separate from solver-backed
  optimality evidence.
- When Pyomo/GDPopt FBBT or interval propagation fails, fix the narrow
  modeling cause before changing solver configuration: derive finite bounds from
  source constraints and parameters for missing-bound failures, and use explicit
  products for small integer powers when negative-bounded variables break
  `PowExpression`. Keep algebra equivalent, verify the affected transformation
  or GDPOpt path, and re-solve previously converged methods when practical to
  confirm bound tightening did not change the solution.
- For ordered-location superstructures or activation-prefix rewrites, test the
  discrete semantics directly on a small representative instance. Enumerate the
  relevant binary activation/location choices and assert the intended valid
  combinations, including any derived feed, recycle, or terminal-location
  expressions. Component-type checks and transformation smoke tests are useful
  but are not enough to protect the modeling semantics.
- For scalable model changes, verify the documented/default instance and at
  least one larger instance when practical. If semantics should be unchanged,
  compare transformed model counts or generated algebraic representations where
  feasible.

## Packaging

- Runtime dependencies belong in `pyproject.toml`, `setup.py`, and `requirements.txt`.
- Development-only tools belong in `requirements-dev.txt` and the Pixi environment.
- Keep optional external solver stacks and licensed solver bindings out of the
  default Pixi environment unless they are required for default imports and
  tests. Put GAMS, BARON, IPOPT, Gurobi, HiGHS, and similar tools in optional
  local environments, optional Pixi features, benchmark profiles, or
  documentation.
- Use `pixi run -e gurobi ...` when local work needs direct `gurobipy` or
  Pyomo's direct Gurobi solver interfaces. Pyomo's direct Gurobi writers do not
  support every nonlinear transformed GDP expression; for those cases, keep
  using the GAMS/Gurobi profile and record the limitation.
- Packaging tests, Pixi runs, and benchmark runs can regenerate
  `gdplib/_version.py` or the editable package entry in `pixi.lock`. Do not
  commit that churn unless the task is explicitly about versioning, release
  metadata, or regenerating the Pixi lock.
- When changing release workflow or Pixi support policy docs, update
  `tests/test_release_workflow.py` so README, `publish.yml`, `pixi.toml`, and
  AGENTS.md stay aligned.
- PyPI publishing is intentionally release-driven. Keep publish workflows tied
  to published GitHub Releases or another explicit maintainer release action,
  not ordinary pushes or pull requests. Publishing should use PyPI Trusted
  Publishing through the protected `pypi` environment unless maintainers choose
  and document another credential path.
- Release workflow changes should build and validate real artifacts before any
  publish step. Include `python -m build`, `python -m twine check dist/*`, and
  a wheel install/import smoke test across the supported Python versions when
  practical. Keep these checks in CI or document why a local-only check is the
  best available evidence.
- `setuptools_scm` needs intact git metadata to infer release versions. CI
  builds should check out full history, and local artifact checks should run
  from a real git checkout or set an explicit pretend version only when the
  purpose is isolated packaging validation. Do not rely on source copies without
  `.git` metadata as release evidence.
- Pixi is the preferred Linux development environment, not a runtime
  requirement for PyPI users. A release should remain installable with standard
  pip tooling from the wheel or sdist, and package metadata should not depend on
  Pixi-only files.
- Treat packaging deprecation warnings from `setuptools`, `setuptools_scm`, or
  PyPA tools as useful follow-up work. Keep warning cleanup in a focused
  metadata PR unless the warning causes the release build, artifact validation,
  or installation smoke test to fail.
