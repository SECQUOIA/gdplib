import os
import sys
from inspect import signature

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_model_size_report import (
    MODEL_INSTANCES,
    build_combined_report,
    render_readme_section,
    update_readme_model_size_section,
)


def test_readme_section_insert_is_idempotent():
    readme = "# GDPlib\n\nIntro text.\n\n## Installation\n\nInstall steps.\n"
    section = "## Model Size Comparison\n\n| Component | Number |\n"

    once = update_readme_model_size_section(readme, section)
    twice = update_readme_model_size_section(once, section)

    assert once == twice
    assert once.count("## Model Size Comparison") == 1
    assert once.index("## Model Size Comparison") < once.index("## Installation")


def test_readme_section_replaces_legacy_heading():
    readme = "# GDPlib\n\n## Model Size Example\n\nold table\n\n## Installation\n"
    section = "## Model Size Comparison\n\nnew table\n"

    updated = update_readme_model_size_section(readme, section)

    assert "## Model Size Example" not in updated
    assert "old table" not in updated
    assert updated.count("## Model Size Comparison") == 1
    assert "new table" in updated


def test_combined_report_links_columns_without_mutating_inputs():
    index = pd.Index(["Variables"], name="Component")
    reports = {
        "jobshop": pd.DataFrame({"Number": [10]}, index=index),
        "mod_hens": pd.DataFrame(
            {"conventional": [338], "mixed_integer": [498]}, index=index
        ),
    }
    original_columns = {name: list(report.columns) for name, report in reports.items()}

    combined = build_combined_report(reports)

    assert {name: list(report.columns) for name, report in reports.items()} == (
        original_columns
    )
    assert "[jobshop](./gdplib/jobshop/)" in combined.columns
    assert "[mod_hens: conventional](./gdplib/mod_hens/)" in combined.columns
    assert "[mod_hens: mixed_integer](./gdplib/mod_hens/)" in combined.columns


def test_render_readme_section_uses_stable_heading():
    index = pd.Index(["Variables"], name="Component")
    combined = pd.DataFrame({"[jobshop](./gdplib/jobshop/)": [10]}, index=index)

    section = render_readme_section(combined)

    assert section.startswith("## Model Size Comparison\n\n")
    assert "generate_model_size_report.py" in section


def test_reverse_electrodialysis_report_uses_solver_free_build_path():
    from gdplib.reverse_electrodialysis import build_model

    assert MODEL_INSTANCES["reverse_electrodialysis"] == [
        {"label": "Number", "kwargs": {"solve_stack": False}}
    ]
    assert "solve_stack" in signature(build_model).parameters


def test_reverse_electrodialysis_solver_free_build_skips_stack_solve(monkeypatch):
    import gdplib.reverse_electrodialysis.REDprocess as redprocess

    def fail_if_called():
        raise AssertionError("stand-alone RED stack solve should not be called")

    monkeypatch.setattr(redprocess, "build_REDstack", fail_if_called)

    model = redprocess.build_model(solve_stack=False)

    assert model is not None
    assert len(list(model.component_objects())) > 0
