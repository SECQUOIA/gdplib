# Benchmark Case Files

This directory contains version-controlled inputs for `gdplib-benchmark
--cases-file`. Generated benchmark outputs stay ignored under
`gdplib/<model>/benchmark_result/` and `benchmark_runs/`.

Case files may be CSV or JSON. CSV files use one benchmark case per row. Common
columns are:

- `instance`: GDPlib model package name.
- `strategy`: one of `gdp.bigm`, `gdp.hull`, `gdpopt.enumerate`, `gdpopt.loa`,
  `gdpopt.gloa`, `gdpopt.lbb`, or `gdpopt.ric`.
- `solver_profile`: named profile from `gdplib.benchmark.SOLVER_PROFILES`.
- `gams_solver`: GAMS solver for direct transformed solves and result naming.
- `timelimit`: per-case time limit in seconds.
- `label`: optional suffix for result file names.
- `gams_nlp_solver`, `gams_mip_solver`, `gams_minlp_solver`,
  `gams_local_minlp_solver`: GDPopt role solvers when using GAMS.

Current tracked campaigns:

- `pr58_local.csv`: full PR #58 model/strategy matrix using the local nonlinear
  GAMS profile: DICOPT for transformed/local MINLP roles, IPOPTH for NLP roles,
  and Gurobi for MIP roles. Rows may override the direct GAMS solver or GDPopt
  role solvers when a model/strategy is known to produce a linear MIP under this
  profile.

Run it with:

```bash
pixi run gdplib-benchmark preflight --cases-file benchmark_cases/pr58_local.csv
pixi run gdplib-benchmark run --cases-file benchmark_cases/pr58_local.csv --run-id pr58_local
```
