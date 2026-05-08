from importlib import import_module
from pyomo.util.model_size import build_model_size_report
import pandas as pd

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
}


if __name__ == "__main__":
    for instance, model_instances in MODEL_INSTANCES.items():
        print("Generating model size report: " + instance)
        build_model = import_module("gdplib." + instance).build_model

        report_df = pd.DataFrame()
        for model_instance in model_instances:
            model = build_model(
                *model_instance.get("args", ()), **model_instance.get("kwargs", {})
            )
            report = build_model_size_report(model)
            temp_df = pd.DataFrame(report.overall, index=[0]).T
            temp_df.columns = [model_instance["label"]]
            report_df = pd.concat([report_df, temp_df], axis=1)

        report_df.index = [
            "Variables",
            "Binary variables",
            "Integer variables",
            "Continuous variables",
            "Disjunctions",
            "Disjuncts",
            "Constraints",
            "Nonlinear constraints",
        ]
        report_df.index.name = "Component"

        # Generate the model size report (Markdown)
        report_df.to_markdown("gdplib/" + instance + "/" + "model_size_report.md")
