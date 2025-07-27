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
        "ex1_linan_2023",
        "gdp_col",
        "hda",
        "jobshop",
        "kaibel",
        "med_term_purchasing",
        "methanol",
        "mod_hens",
        "modprodnet",
        "positioning",
        "small_batch",
        "spectralog",
        "stranded_gas",
        "syngas",
        "water_network"
    ]
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timelimit = 600

    # Dictionary to store all model reports
    all_reports = {}

    for instance in instance_list:
        print("Generating model size report: " + instance)
        try:
            model = import_module("gdplib." + instance).build_model()
            report = build_model_size_report(model)
            report_df = pd.DataFrame(report.overall, index=[0]).T
            report_df.index.name = "Component"
            report_df.columns = [instance]  # Use model name as column

            # Store the report
            all_reports[instance] = report_df

            # Generate individual model size report (Markdown)
            individual_report_df = report_df.copy()
            individual_report_df.columns = ["Number"]  # Change column name for individual report
            individual_report_df.to_markdown("gdplib/" + instance + "/" + "model_size_report.md")
        except Exception as e:
            print(f"Error processing {instance}: {str(e)}")
            continue

    # Combine all reports into a single table
    if not all_reports:
        print("No reports were generated. Skipping combined report generation.")
    else:
        combined_df = pd.concat([df for df in all_reports.values()], axis=1)

        # Sort columns alphabetically
        combined_df = combined_df.sort_index(axis=1)

        # Generate the combined report
        combined_report = "## Model Size Comparison\n\n"
        combined_report += "The following table shows the size metrics for all models in GDPlib:\n\n"
        combined_report += combined_df.to_markdown()
        combined_report += "\n\nThis table was automatically generated using the `generate_model_size_report.py` script.\n"

        # Read current README content
        with open("README.md", "r") as f:
            readme_content = f.read()

        # Find the position to insert the table (after "## Model Size Example")
        size_example_pos = readme_content.find("## Model Size Example")
        
        if size_example_pos == -1:
            print("Warning: '## Model Size Example' section not found in README.md. Appending report to the end of the file.")
            new_readme = readme_content + "\n\n" + combined_report
        else:
            next_section_pos = readme_content.find("##", size_example_pos + 1)
            # Create new README content
            new_readme = (
                readme_content[:size_example_pos] +
                combined_report +
                "\n\n" +
                readme_content[next_section_pos:]
            )
        
        # Write updated README
        with open("README.md", "w") as f:
            f.write(new_readme)
