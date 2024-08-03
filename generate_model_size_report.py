from datetime import datetime
from importlib import import_module
from pyomo.util.model_size import build_model_size_report
import pandas as pd


if __name__ == "__main__":
    instance_list = [
        "batch_processing",
        "biofuel",
        "cstr",
        "disease_model",
        "gdp_col",
        "hda",
        "jobshop",
        # "kaibel", # next step
        "med_term_purchasing",
        "methanol",
        # "mod_hens", # next step
        "modprodnet",
        "positioning",
        "spectralog",
        "stranded_gas",
        "syngas",
    ]
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timelimit = 600

    for instance in instance_list:
        print("Generating model size report: " + instance)

        model = import_module("gdplib." + instance).build_model()
        report = build_model_size_report(model)
        report_df = pd.DataFrame(report.overall, index=[0]).T

        report_df.columns = ["Number"]
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
