import glob
import json
import os

import pandas as pd


def _gap_or_user_time(row):
    termination_condition = row['Termination condition']
    if termination_condition == 'infeasible':
        return None
    if termination_condition == 'optimal':
        return row['User time']
    if termination_condition != 'maxTimeLimit':
        return None

    if row['Sense'] == 'minimize':
        denominator = row['Upper bound']
    elif row['Sense'] == 'maximize':
        denominator = row['Lower bound']
    else:
        return None

    return abs(abs(row['Upper bound'] - row['Lower bound']) / denominator)


def _objective_value(row):
    if row['Termination condition'] == 'infeasible':
        return None
    if row['Sense'] == 'minimize':
        return row['Upper bound']
    if row['Sense'] == 'maximize':
        return row['Lower bound']
    return None


def generate_benchmark_summary(folders):
    """Generate combined benchmark summary tables for result folders."""
    problem_dataframes = []
    solver_dataframes = []

    for folder in folders:
        json_files = glob.glob(os.path.join(folder, '*.json'))

        for file_path in json_files:
            with open(file_path, 'r') as file:
                data = json.load(file)

            problem_dataframes.append(pd.json_normalize(data, 'Problem'))
            solver_dataframes.append(pd.json_normalize(data, 'Solver'))

    combined_problem_df = pd.concat(problem_dataframes, ignore_index=False)
    combined_solver_df = pd.concat(solver_dataframes, ignore_index=False)

    combined_problem_df.rename(columns={'Name': 'id'}, inplace=True)
    combined_solver_df['id'] = combined_problem_df['id']
    combined_problem_df.set_index('id', inplace=True)
    combined_solver_df.set_index('id', inplace=True)

    problem_columns = ['Lower bound', 'Upper bound', 'Sense']
    solver_columns = ['Name', 'Termination condition', 'User time']

    selected_problem_df = combined_problem_df[problem_columns]
    selected_solver_df = combined_solver_df[solver_columns]
    combined_df = pd.concat([selected_problem_df, selected_solver_df], axis='columns')

    combined_df['Gap [%] or Time [s]'] = combined_df.apply(_gap_or_user_time, axis=1)
    combined_df['Objective value'] = combined_df.apply(_objective_value, axis=1)

    os.makedirs('benchmark_summary', exist_ok=True)

    combined_problem_df.to_markdown(
        os.path.join('benchmark_summary', 'combined_problem_data.md'), index=True
    )
    combined_solver_df.to_markdown(
        os.path.join('benchmark_summary', 'combined_solver_data.md'), index=True
    )
    combined_df.to_markdown(
        os.path.join('benchmark_summary', 'combined_data.md'), index=True
    )

    combined_problem_df.to_csv(
        os.path.join('benchmark_summary', 'combined_problem_data.csv'), index=True
    )
    combined_solver_df.to_csv(
        os.path.join('benchmark_summary', 'combined_solver_data.csv'), index=True
    )
    combined_df.to_csv('benchmark_summary/combined_data.csv', index=True)
