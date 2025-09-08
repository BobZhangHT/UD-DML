# evaluation.py
#
# This script loads the saved simulation results, calculates evaluation metrics,
# and generates plots and tables for the manuscript.
# -----------------------------------------------------------------------------

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import SIM_RESULTS_DIR, ANALYSIS_RESULTS_DIR, scenarios

def load_results():
    """Load all simulation results into a single pandas DataFrame."""
    results = []
    if not os.path.exists(SIM_RESULTS_DIR):
        return pd.DataFrame(results)

    for scenario_name in os.listdir(SIM_RESULTS_DIR):
        scenario_dir = os.path.join(SIM_RESULTS_DIR, scenario_name)
        if not os.path.isdir(scenario_dir) or scenario_name not in scenarios:
            continue
            
        true_beta_scenario = scenarios[scenario_name]['params']['true_beta']
        
        for file_name in os.listdir(scenario_dir):
            if file_name.endswith(".pkl"):
                try:
                    parts = file_name.replace(".pkl", "").split("_")
                    sim_id = int(parts[1])
                    method_name = parts[3]
                    
                    with open(os.path.join(scenario_dir, file_name), 'rb') as f:
                        res = pickle.load(f)
                    
                    # Skip if result is malformed or contains NaNs
                    if 'point_est' not in res or np.isnan(res['point_est']).any():
                        continue
                        
                    # Calculate squared error for the first parameter (theta_1)
                    squared_error_theta1 = (res['point_est'][0] - true_beta_scenario[0])**2
                    
                    ci_width = np.mean(res['ci_upper'] - res['ci_lower'])
                    # Coverage for the first, most prominent parameter
                    coverage = (res['ci_lower'][0] <= true_beta_scenario[0]) and \
                               (res['ci_upper'][0] >= true_beta_scenario[0])
                    
                    results.append({
                        "scenario": scenario_name,
                        "sim_id": sim_id,
                        "method": method_name,
                        "squared_error_theta1": squared_error_theta1,
                        "ci_width": ci_width,
                        "coverage": int(coverage),
                        "running_time": res['running_time']
                    })
                except (IOError, pickle.PickleError, IndexError) as e:
                    print(f"Warning: Could not process file {file_name}. Reason: {e}")
    
    return pd.DataFrame(results)

def generate_summary_table(df):
    """Generate a LaTeX summary table of the results."""
    # Define a custom aggregation for RMSE
    def rmse(x):
        return np.sqrt(x.mean())

    summary = df.groupby(['scenario', 'method']).agg(
        RMSE_theta1=('squared_error_theta1', rmse),
        CI_Width=('ci_width', 'mean'),
        Coverage=('coverage', 'mean'),
        Time=('running_time', 'mean')
    ).reset_index()
    
    # Format for LaTeX
    summary_latex = summary.to_latex(
        index=False,
        float_format="%.3f",
        caption="Summary of Simulation Results across Scenarios and Methods.",
        label="tab:summary",
        column_format="llrrrr"
    )
    
    table_path = os.path.join(ANALYSIS_RESULTS_DIR, "summary_table.tex")
    with open(table_path, "w") as f:
        f.write(summary_latex)
    print(f"Summary table saved to {table_path}")
    return summary

def generate_plots(df):
    """Generate improved boxplots for each metric, faceted by scenario."""
    
    # Calculate RMSE from squared error
    df['rmse_theta1'] = np.sqrt(df['squared_error_theta1'])
    metrics = ['rmse_theta1', 'ci_width', 'coverage', 'running_time']
    metric_titles = {
        'rmse_theta1': 'RMSE for Theta 1',
        'ci_width': 'Average CI Width',
        'coverage': 'Coverage Rate for Theta 1',
        'running_time': 'Running Time (seconds)'
    }
    
    scenarios_list = sorted(df['scenario'].unique())
    n_scenarios = len(scenarios_list)
    n_cols = 3
    n_rows = (n_scenarios + n_cols - 1) // n_cols

    for metric in metrics:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5), squeeze=False)
        axes = axes.flatten()

        for i, scenario_name in enumerate(scenarios_list):
            ax = axes[i]
            scenario_df = df[df['scenario'] == scenario_name]
            
            sns.boxplot(x='method', y=metric, data=scenario_df, ax=ax, palette="viridis")
            ax.set_title(scenario_name.replace('_', ' ').title(), fontsize=12)
            ax.set_xlabel('')
            ax.set_ylabel(metric_titles[metric])
            ax.tick_params(axis='x', rotation=45)
            
        # Hide unused subplots
        for i in range(n_scenarios, len(axes)):
            fig.delaxes(axes[i])
            
        fig.suptitle(f'Comparison of Methods: {metric_titles[metric]}', fontsize=18, y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        plot_path = os.path.join(ANALYSIS_RESULTS_DIR, f"plot_{metric}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Plot saved to {plot_path}")

def main():
    """Main function to run the evaluation."""
    print("Loading simulation results...")
    results_df = load_results()
    
    if results_df.empty:
        print("Simulation results directory is empty or contains no valid results. Please run simulation.ipynb first.")
        return

    # Create analysis directory if it doesn't exist
    if not os.path.exists(ANALYSIS_RESULTS_DIR):
        os.makedirs(ANALYSIS_RESULTS_DIR)

    print("\nGenerating summary table...")
    summary_table = generate_summary_table(results_df)
    print("\n--- Summary Table ---")
    print(summary_table.to_string())
    
    print("\nGenerating plots...")
    generate_plots(results_df)
    
    print(f"\nEvaluation complete. Results are in the '{ANALYSIS_RESULTS_DIR}' directory.")

if __name__ == "__main__":
    main()

