import os

from gdplib.benchmark_summary import generate_benchmark_summary

if __name__ == "__main__":
    instance_list = [
        "batch_processing",
        "biofuel",  # enumeration got stuck
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
    ]

    folders = []

    for instance in instance_list:
        # Make a directory to store the summary
        os.makedirs(
            os.path.join("gdplib", instance, "benchmark_result", "summary"),
            exist_ok=True,
        )
        # # Navigate to the benchmark_result folder for the current instance
        # os.chdir(os.path.join('gdplib', instance, 'benchmark_result'))
        # # Make a directory to store the summary
        # os.makedirs('summary', exist_ok=True)
        # # Navigate back to the root directory
        # os.chdir('../../../')
        # List of folders containing JSON files for the current instance
        folders.append(os.path.join("gdplib", instance, "benchmark_result", "summary"))
        try:
            generate_benchmark_summary(folders)
        except Exception as e:
            print(f"Error processing instance {instance}: {e}")
            continue
        print(f"Generated benchmark summary for instance: {instance}")
