from importlib import import_module
from pathlib import Path

from pyomo.util.model_size import build_model_size_report
import pandas as pd

README_SECTION_HEADING = "## Model Size Comparison"
LEGACY_SECTION_HEADING = "## Model Size Example"
README_PATH = Path("README.md")

REPORT_INDEX = [
    "Variables",
    "Binary variables",
    "Integer variables",
    "Continuous variables",
    "Disjunctions",
    "Disjuncts",
    "Constraints",
    "Nonlinear constraints",
]

MODEL_INSTANCES = {
    "batch_processing": [{"label": "Number"}],
    "biofuel": [{"label": "Number"}],
    "cstr": [{"label": "Number"}],
    "disease_model": [{"label": "Number"}],
    "ex1_linan_2023": [{"label": "Number"}],
    "gdp_col": [{"label": "Number"}],
    "hda": [{"label": "Number"}],
    "jobshop": [{"label": "Number"}],
    "kaibel": [{"label": "Number"}],
    "med_term_purchasing": [{"label": "Number"}],
    "methanol": [{"label": "Number"}],
    "mod_hens": [
        {
            "label": "conventional",
            "args": ("conventional",),
            "kwargs": {"cafaro_approx": False},
        },
        {
            "label": "single_module_integer",
            "args": ("single_module_integer",),
            "kwargs": {"cafaro_approx": False},
        },
        {
            "label": "multiple_module_integer",
            "args": ("multiple_module_integer",),
            "kwargs": {"cafaro_approx": False},
        },
        {
            "label": "mixed_integer",
            "args": ("mixed_integer",),
            "kwargs": {"cafaro_approx": False},
        },
        {
            "label": "single_module_discrete",
            "args": ("single_module_discrete",),
            "kwargs": {"cafaro_approx": False},
        },
        {
            "label": "multiple_module_discrete",
            "args": ("multiple_module_discrete",),
            "kwargs": {"cafaro_approx": False},
        },
        {
            "label": "mixed_discrete",
            "args": ("mixed_discrete",),
            "kwargs": {"cafaro_approx": False},
        },
    ],
    "modprodnet": [
        {"label": "Growth", "args": ("Growth",)},
        {"label": "Dip", "args": ("Dip",)},
        {"label": "Decay", "args": ("Decay",)},
        {"label": "Distributed", "args": ("Distributed",)},
        {"label": "QuarterDistributed", "args": ("QuarterDistributed",)},
    ],
    "positioning": [{"label": "Number"}],
    "reverse_electrodialysis": [{"label": "Number"}],
    "small_batch": [{"label": "Number"}],
    "spectralog": [{"label": "Number"}],
    "stranded_gas": [
        {"label": "Gas_100", "args": ("Gas_100",)},
        {"label": "Gas_250", "args": ("Gas_250",)},
        {"label": "Gas_500", "args": ("Gas_500",)},
        {"label": "Gas_small", "args": ("Gas_small",)},
        {"label": "Gas_large", "args": ("Gas_large",)},
    ],
    "syngas": [{"label": "Number"}],
    "water_network": [
        {"label": "none"},
        {
            "label": "quadratic_zero_origin",
            "kwargs": {"approximation": "quadratic_zero_origin"},
        },
        {
            "label": "quadratic_nonzero_origin",
            "kwargs": {"approximation": "quadratic_nonzero_origin"},
        },
        {"label": "piecewise", "kwargs": {"approximation": "piecewise"}},
    ],
    "multiperiod_blending": [{"label": "mpbp_6"}],
}


def build_report_dataframe(build_model, model_instances):
    report_df = pd.DataFrame()
    for model_instance in model_instances:
        model = build_model(
            *model_instance.get("args", ()), **model_instance.get("kwargs", {})
        )
        report = build_model_size_report(model)
        temp_df = pd.DataFrame(report.overall, index=[0]).T
        temp_df.columns = [model_instance["label"]]
        report_df = pd.concat([report_df, temp_df], axis=1)

    report_df.index = REPORT_INDEX
    report_df.index.name = "Component"
    return report_df


def build_model_reports(model_instances=MODEL_INSTANCES):
    reports = {}
    for instance, instances in model_instances.items():
        print("Generating model size report: " + instance)
        build_model = import_module("gdplib." + instance).build_model
        reports[instance] = build_report_dataframe(build_model, instances)
    return reports


def write_model_reports(reports):
    for instance, report_df in reports.items():
        report_path = Path("gdplib") / instance / "model_size_report.md"
        report_df.to_markdown(report_path)


def linked_column_names(instance, report_df):
    if list(report_df.columns) == ["Number"]:
        return [f"[{instance}](./gdplib/{instance}/)"]
    return [
        f"[{instance}: {column}](./gdplib/{instance}/)" for column in report_df.columns
    ]


def build_combined_report(reports):
    linked_reports = []
    for instance, report_df in reports.items():
        linked_df = report_df.copy()
        linked_df.columns = linked_column_names(instance, report_df)
        linked_reports.append(linked_df)

    return pd.concat(linked_reports, axis=1).sort_index(axis=1)


def render_readme_section(combined_df):
    section = README_SECTION_HEADING + "\n\n"
    section += "The following table shows the size metrics for GDPlib models:\n\n"
    section += combined_df.to_markdown()
    section += (
        "\n\nThis table was automatically generated using the "
        "`generate_model_size_report.py` script.\n"
    )
    return section


def find_model_size_section(readme_content):
    section_positions = [
        position
        for position in (
            readme_content.find(README_SECTION_HEADING),
            readme_content.find(LEGACY_SECTION_HEADING),
        )
        if position != -1
    ]
    if not section_positions:
        return None

    section_start = min(section_positions)
    next_section_start = readme_content.find("\n## ", section_start + 1)
    if next_section_start == -1:
        return section_start, len(readme_content)
    return section_start, next_section_start + 1


def update_readme_model_size_section(readme_content, section):
    normalized_section = section.rstrip() + "\n\n"
    section_span = find_model_size_section(readme_content)

    if section_span is not None:
        section_start, section_end = section_span
        return (
            readme_content[:section_start].rstrip()
            + "\n\n"
            + normalized_section
            + readme_content[section_end:].lstrip()
        )

    installation_pos = readme_content.find("## Installation")
    if installation_pos != -1:
        return (
            readme_content[:installation_pos].rstrip()
            + "\n\n"
            + normalized_section
            + readme_content[installation_pos:].lstrip()
        )

    return readme_content.rstrip() + "\n\n" + normalized_section.rstrip() + "\n"


def write_readme_report(combined_df, readme_path=README_PATH):
    readme_content = readme_path.read_text()
    section = render_readme_section(combined_df)
    readme_path.write_text(update_readme_model_size_section(readme_content, section))


def main():
    reports = build_model_reports()
    if not reports:
        print("No reports were generated. Skipping README report generation.")
        return

    write_model_reports(reports)
    write_readme_report(build_combined_report(reports))


if __name__ == "__main__":
    main()
