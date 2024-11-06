import pandas as pd
import json
import os
import glob
import plotly.graph_objects as go


def generate_benchmark_summary(folders):
    # Search for benchmark folders in each instance folder and date folder to generate a summary
    # of all benchmark results in a single table.
    # The summary includes the "Problem" and "Solver" data for each benchmark result.

    # # List of folders containing JSON files
    # folders = [
    #     os.path.join('gdplib',instance,'benchmark_result','summary'),
    # ]

    # Initialize empty lists to store DataFrames
    problem_dataframes = []
    solver_dataframes = []

    # Iterate over each folder
    for folder in folders:
        # Find all JSON files in the current folder
        json_files = glob.glob(os.path.join(folder, '*.json'))

        # Iterate over each JSON file
        for file_path in json_files:
            # Read the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)

            # Normalize JSON structure and create DataFrame
            problem_df = pd.json_normalize(data, 'Problem')
            solver_df = pd.json_normalize(data, 'Solver')

            # Append DataFrames to the respective lists
            problem_dataframes.append(problem_df)
            solver_dataframes.append(solver_df)

    # Concatenate all DataFrames into single DataFrames for "Problem" and "Solver" do not ignore the index
    combined_problem_df = pd.concat(problem_dataframes, ignore_index=False)
    combined_solver_df = pd.concat(solver_dataframes, ignore_index=False)

    # Rename the 'Name' column to 'id' in the 'Problem' DataFrame
    combined_problem_df.rename(columns={'Name': 'id'}, inplace=True)
    # Add the 'id' column to the 'Solver' DataFrame
    combined_solver_df['id'] = combined_problem_df['id']
    # Set the 'id' column as the index for the 'Problem' and 'Solver' DataFrames
    combined_problem_df.set_index('id', inplace=True)
    combined_solver_df.set_index('id', inplace=True)

    # Join the "Problem" and "Solver" DataFrames on the 'id' column to create a single DataFrame
    # combined_df = combined_problem_df.merge(combined_solver_df, on='id')

    # Select desired columns from each DataFrame
    problem_columns = ['Lower bound', 'Upper bound', 'Sense']
    solver_columns = ['Name', 'Termination condition', 'User time']

    selected_problem_df = combined_problem_df[problem_columns]
    selected_solver_df = combined_solver_df[solver_columns]

    # Join the selected columns from the "Problem" and "Solver" DataFrames
    combined_df = pd.concat([selected_problem_df, selected_solver_df], axis="columns")

    # Compute the gap or user time based on the termination status
    def compute_gap_or_user_time(row):
        if row['Termination condition'] != 'infeasible':
            if row['Termination condition'] == 'maxTimeLimit':
                if row['Sense'] == 'minimize':
                    return abs(
                        abs(row['Upper bound'] - row['Lower bound'])
                        / row['Upper bound']
                        # * 100
                    )
                elif row['Sense'] == 'maximize':
                    return abs(
                        abs(row['Upper bound'] - row['Lower bound'])
                        / row['Lower bound']
                        # * 100
                    )
            elif row['Termination condition'] == 'optimal':
                return row['User time']
        return None

    combined_df['Gap [%] or Time [s]'] = combined_df.apply(
        compute_gap_or_user_time, axis=1
    )

    # Create new column with the objective value
    def compute_objective_value(row):
        if row['Termination condition'] != 'infeasible':
            if row['Sense'] == 'minimize':
                return row['Upper bound']
            elif row['Sense'] == 'maximize':
                return row['Lower bound']
        return None

    combined_df['Objective value'] = combined_df.apply(compute_objective_value, axis=1)

    # Optionally, export the combined DataFrames to Markdown tables in folder "benchmark_summary"
    # Ensure the 'benchmark_summary' folder exists
    os.makedirs('benchmark_summary', exist_ok=True)

    # Export the combined DataFrames to Markdown tables in the 'benchmark_summary' folder
    combined_problem_df.to_markdown(
        os.path.join('benchmark_summary', 'combined_problem_data.md'), index=True
    )
    combined_solver_df.to_markdown(
        os.path.join('benchmark_summary', 'combined_solver_data.md'), index=True
    )

    # Export the combined DataFrame to a single Markdown table in the 'benchmark_summary' folder
    combined_df.to_markdown(
        os.path.join('benchmark_summary', 'combined_data.md'), index=True
    )

    # Optionally, export the combined DataFrames to CSV files
    combined_problem_df.to_csv(
        os.path.join('benchmark_summary', 'combined_problem_data.csv'), index=True
    )
    combined_solver_df.to_csv(
        os.path.join('benchmark_summary', 'combined_solver_data.csv'), index=True
    )

    # Optionally, export the combined DataFrame to a single CSV file
    combined_df.to_csv('benchmark_summary/combined_data.csv', index=True)


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
            os.path.join('gdplib', instance, 'benchmark_result', 'summary'),
            exist_ok=True,
        )
        # # Navigate to the benchmark_result folder for the current instance
        # os.chdir(os.path.join('gdplib', instance, 'benchmark_result'))
        # # Make a directory to store the summary
        # os.makedirs('summary', exist_ok=True)
        # # Navigate back to the root directory
        # os.chdir('../../../')
        # List of folders containing JSON files for the current instance
        folders.append(os.path.join('gdplib', instance, 'benchmark_result', 'summary'))
        try:
            generate_benchmark_summary(folders)
        except Exception as e:
            print(f"Error processing instance {instance}: {e}")
            continue
        print(f"Generated benchmark summary for instance: {instance}")
