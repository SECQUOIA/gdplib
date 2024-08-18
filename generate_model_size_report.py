from datetime import datetime
from importlib import import_module
from pyomo.util.model_size import build_model_size_report
import pandas as pd


if __name__ == "__main__":
    instance_dict = {
        "batch_processing": [],
        "biofuel": [],
        "cstr": [],
        "disease_model": [],
        "gdp_col": [],
        "hda": [],
        "jobshop": [],
        # "kaibel", # next step
        "med_term_purchasing": [],
        "methanol": [],
        "mod_hens": [
            "conventional",
            "single_module_integer",
            "multiple_module_integer",
            "mixed_integer",
            "single_module_discrete",
            "multiple_module_discrete",
            "mixed_discrete",
        ],
        "modprodnet": [],
        "positioning": [],
        "spectralog": [],
        "stranded_gas": [],
        "syngas": [],
    }
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timelimit = 600

    for instance, cases in instance_dict.items():
        print("Generating model size report: " + instance)

        if cases == []:
            model = import_module("gdplib." + instance).build_model()
            report = build_model_size_report(model)
            report_df = pd.DataFrame(report.overall, index=[0]).T

            report_df.columns = ["Number"]
        else:
            report_df = pd.DataFrame()
            for case in cases:
                model = import_module("gdplib." + instance).build_model(case)
                case_report = build_model_size_report(model)
                temp_df = pd.DataFrame(case_report.overall, index=[0]).T
                temp_df.columns = [case]
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
