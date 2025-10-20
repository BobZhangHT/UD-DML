# -*- coding: utf-8 -*-
"""
evaluation.py

Loads results, calculates metrics, and generates summary tables.
Now includes AMSE Ratio and Time-Normalized MSE (TNMSE).
All content is in English.
"""
import os
import pickle
import pandas as pd
import numpy as np
import config

def load_results(results_dir, scenarios):
    """Loads simulation results from a specific directory into a pandas DataFrame."""
    # ... (code from previous turn, no changes needed here)
    results_list = []
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found.")
        return pd.DataFrame()

    for scenario_name in os.listdir(results_dir):
        scenario_dir = os.path.join(results_dir, scenario_name)
        if not os.path.isdir(scenario_dir): continue

        true_ate = np.nan
        for file_name in os.listdir(scenario_dir):
             if file_name.endswith(".pkl"):
                try:
                    with open(os.path.join(scenario_dir, file_name), 'rb') as f:
                        res_sample = pickle.load(f)
                        true_ate = res_sample.get('true_ate', np.nan)
                        break 
                except Exception:
                    continue

        for file_name in os.listdir(scenario_dir):
            if not file_name.endswith(".pkl"): continue
            
            try:
                with open(os.path.join(scenario_dir, file_name), 'rb') as f:
                    res = pickle.load(f)
                
                est_ate, ci_lower, ci_upper = res.get('est_ate'), res.get('ci_lower'), res.get('ci_upper')
                
                if est_ate is not None and not np.isnan(est_ate) and ci_lower is not None and not np.isnan(ci_lower):
                    res_dict = {
                        "scenario": res.get('scenario'), "method": res.get('method'),
                        "Coverage": (ci_lower <= true_ate) and (true_ate <= ci_upper),
                        "CI_Width": ci_upper - ci_lower, "Bias": est_ate - true_ate,
                        "Sq_Error": (est_ate - true_ate)**2,
                        "Runtime": res.get('runtime', 0)
                    }
                    if 'pilot_ratio' in res: res_dict['pilot_ratio'] = res['pilot_ratio']
                    if 'misspecification' in res: res_dict['misspecification'] = res['misspecification']
                    results_list.append(res_dict)
            except Exception as e:
                print(f"Warning: Could not process {file_name}. Reason: {e}")

    return pd.DataFrame(results_list)


def generate_summary_table(df, experiment_name):
    """Generates a summary table with standard and new efficiency metrics."""
    if df.empty: return None
        
    grouping_vars = ['scenario', 'method']
    if 'pilot_ratio' in df.columns: grouping_vars = ['pilot_ratio', 'method', 'scenario']
    if 'misspecification' in df.columns: grouping_vars = ['misspecification', 'method']

    summary = df.groupby(grouping_vars).agg(
        Coverage=('Coverage', 'mean'), 
        CI_Width=('CI_Width', 'mean'),
        Bias=('Bias', 'mean'), 
        RMSE=('Sq_Error', lambda x: np.sqrt(x.mean())),
        Runtime=('Runtime', 'mean')
    ).reset_index()
    
    # --- New Metrics ---
    if 'method' in summary.columns and 'scenario' in summary.columns and experiment_name == "experiment_2_main_comparison":
        # 1. AMSE Ratio (using RMSE as a proxy for asymptotic variance)
        unif_rmse = summary[summary['method'] == 'UNIF'].set_index('scenario')['RMSE']
        summary['AMSE_Ratio'] = summary.apply(
            lambda row: (unif_rmse.get(row['scenario'], np.nan) / row['RMSE'])**2 if row['method'] != 'UNIF' else 1.0,
            axis=1
        )
        
        # 2. Time-Normalized MSE (TNMSE)
        summary['TNMSE'] = summary['RMSE']**2 * summary['Runtime']

    analysis_dir = f"./analysis_results/{experiment_name}"
    os.makedirs(analysis_dir, exist_ok=True)
    
    method_order = ['OS', 'UNIF', 'LSS', 'FULL']
    if 'method' in summary.columns:
        summary['method'] = pd.Categorical(summary['method'], categories=method_order, ordered=True)
        summary = summary.sort_values(grouping_vars)
    
    table_path = os.path.join(analysis_dir, "summary_table.csv")
    summary.to_csv(table_path, index=False, float_format="%.4f")
    print(f"\nSummary table saved to {table_path}")
    return summary

def process_results(results_list):
    """Process results list and generate summary table."""
    if not results_list:
        return pd.DataFrame()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results_list)
    
    if df.empty:
        return pd.DataFrame()
    
    # Calculate metrics
    df['Coverage'] = (df['ci_lower'] <= df['true_ate']) & (df['true_ate'] <= df['ci_upper'])
    df['CI_Width'] = df['ci_upper'] - df['ci_lower']
    df['Bias'] = df['est_ate'] - df['true_ate']
    df['Sq_Error'] = (df['est_ate'] - df['true_ate'])**2
    
    # Group by scenario and method
    grouping_vars = ['scenario', 'method']
    if 'misspecification' in df.columns:
        grouping_vars = ['misspecification', 'method']
    
    summary = df.groupby(grouping_vars).agg(
        Coverage=('Coverage', 'mean'),
        CI_Width=('CI_Width', 'mean'),
        Bias=('Bias', 'mean'),
        RMSE=('Sq_Error', lambda x: np.sqrt(x.mean())),
        Runtime=('runtime', 'mean')
    ).reset_index()
    
    # Add AMSE Ratio and TNMSE for experiment 2
    if 'scenario' in summary.columns and 'method' in summary.columns:
        # AMSE Ratio (using RMSE as proxy for asymptotic variance)
        unif_rmse = summary[summary['method'] == 'UNIF'].set_index('scenario')['RMSE']
        summary['AMSE_Ratio'] = summary.apply(
            lambda row: (unif_rmse.get(row['scenario'], np.nan) / row['RMSE'])**2 if row['method'] != 'UNIF' else 1.0,
            axis=1
        )
        
        # Time-Normalized MSE (TNMSE)
        summary['TNMSE'] = summary['RMSE']**2 * summary['Runtime']
    
    # Sort by method order
    method_order = ['OS', 'UNIF', 'LSS', 'FULL']
    if 'method' in summary.columns:
        summary['method'] = pd.Categorical(summary['method'], categories=method_order, ordered=True)
        summary = summary.sort_values(grouping_vars)
    
    return summary

def generate_latex_table(summary_df, experiment_name, output_dir):
    """Generate LaTeX table for Overleaf."""
    if summary_df.empty:
        return
    
    # Create LaTeX table
    latex_content = []
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{" + experiment_name.replace('_', ' ').title() + " Results}")
    latex_content.append("\\label{tab:" + experiment_name + "}")
    
    # Determine columns based on experiment
    if 'scenario' in summary_df.columns:
        # Experiment 2: Main Comparison
        latex_content.append("\\begin{tabular}{l" + "c" * (len(summary_df.columns) - 2) + "}")
        latex_content.append("\\toprule")
        
        # Header
        header = "Scenario & Method"
        for col in summary_df.columns:
            if col not in ['scenario', 'method']:
                if col == 'Coverage':
                    header += " & Coverage"
                elif col == 'RMSE':
                    header += " & RMSE"
                elif col == 'Bias':
                    header += " & Bias"
                elif col == 'CI_Width':
                    header += " & CI Width"
                elif col == 'Runtime':
                    header += " & Runtime (s)"
                elif col == 'AMSE_Ratio':
                    header += " & AMSE Ratio"
                elif col == 'TNMSE':
                    header += " & TNMSE"
                else:
                    header += f" & {col}"
        header += " \\\\"
        latex_content.append(header)
        latex_content.append("\\midrule")
        
        # Data rows
        for _, row in summary_df.iterrows():
            line = f"{row['scenario']} & {row['method']}"
            for col in summary_df.columns:
                if col not in ['scenario', 'method']:
                    if col in ['Coverage', 'RMSE', 'Bias', 'CI_Width', 'AMSE_Ratio', 'TNMSE']:
                        line += f" & {row[col]:.4f}"
                    elif col == 'Runtime':
                        line += f" & {row[col]:.2f}"
                    else:
                        line += f" & {row[col]:.4f}"
            line += " \\\\"
            latex_content.append(line)
    
    elif 'misspecification' in summary_df.columns:
        # Experiment 3: Robustness Check
        latex_content.append("\\begin{tabular}{l" + "c" * (len(summary_df.columns) - 2) + "}")
        latex_content.append("\\toprule")
        
        # Header
        header = "Misspecification & Method"
        for col in summary_df.columns:
            if col not in ['misspecification', 'method']:
                if col == 'Coverage':
                    header += " & Coverage"
                elif col == 'RMSE':
                    header += " & RMSE"
                elif col == 'Bias':
                    header += " & Bias"
                elif col == 'CI_Width':
                    header += " & CI Width"
                elif col == 'Runtime':
                    header += " & Runtime (s)"
                else:
                    header += f" & {col}"
        header += " \\\\"
        latex_content.append(header)
        latex_content.append("\\midrule")
        
        # Data rows
        for _, row in summary_df.iterrows():
            line = f"{row['misspecification']} & {row['method']}"
            for col in summary_df.columns:
                if col not in ['misspecification', 'method']:
                    if col in ['Coverage', 'RMSE', 'Bias', 'CI_Width']:
                        line += f" & {row[col]:.4f}"
                    elif col == 'Runtime':
                        line += f" & {row[col]:.2f}"
                    else:
                        line += f" & {row[col]:.4f}"
            line += " \\\\"
            latex_content.append(line)
    
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")
    
    # Save to file
    latex_file = output_dir / f"{experiment_name}_table.tex"
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_content))
    
    print(f"✓ LaTeX table saved to: {latex_file}")

def main():
    """Main function to run the evaluation for all experiments."""
    scenarios, _, experiments = config.get_experiments()
    
    for exp_name, exp_config in experiments.items():
        print(f"\n--- Evaluating: {exp_name} ---")
        results_df = load_results(exp_config['base_dir'], scenarios)
        
        if results_df.empty:
            print(f"No valid simulation results found for '{exp_name}'.")
            continue

        print("Generating summary table...")
        summary_table = generate_summary_table(results_df, exp_name)
        
        if summary_table is not None:
            print(f"\n--- {exp_name} Summary Table ---")
            print(summary_table.to_string(index=False, float_format="%.4f"))

    print(f"\nEvaluation complete.")

def generate_robustness_tables(results_list, output_dir):
    """Generate separate robustness tables for each scenario and a combined table."""
    if not results_list:
        return
    
    # Convert results to DataFrame
    df = pd.DataFrame(results_list)
    if df.empty:
        return
    
    # Calculate metrics
    df['Coverage'] = (df['ci_lower'] <= df['true_ate']) & (df['true_ate'] <= df['ci_upper'])
    df['CI_Width'] = df['ci_upper'] - df['ci_lower']
    df['Bias'] = df['est_ate'] - df['true_ate']
    df['Sq_Error'] = (df['est_ate'] - df['true_ate'])**2
    
    # Get unique scenarios
    scenarios = df['scenario'].unique()
    
    # Generate table for each scenario
    for scenario in scenarios:
        scenario_df = df[df['scenario'] == scenario]
        
        # Group by misspecification and method
        summary = scenario_df.groupby(['misspecification', 'method']).agg(
            Coverage=('Coverage', 'mean'),
            CI_Width=('CI_Width', 'mean'),
            Bias=('Bias', 'mean'),
            RMSE=('Sq_Error', lambda x: np.sqrt(x.mean())),
            Runtime=('runtime', 'mean')
        ).reset_index()
        
        # Generate LaTeX table for this scenario
        generate_robustness_latex_table(summary, scenario, output_dir)
        
        # Generate formatted CSV table for this scenario
        generate_robustness_csv_table(summary, scenario, output_dir)
    
    # Generate combined table
    combined_summary = df.groupby(['scenario', 'misspecification', 'method']).agg(
        Coverage=('Coverage', 'mean'),
        CI_Width=('CI_Width', 'mean'),
        Bias=('Bias', 'mean'),
        RMSE=('Sq_Error', lambda x: np.sqrt(x.mean())),
        Runtime=('runtime', 'mean')
    ).reset_index()
    
    # Generate combined LaTeX table
    generate_combined_robustness_latex_table(combined_summary, output_dir)
    
    # Generate combined CSV table with proper formatting
    generate_combined_robustness_csv_table(combined_summary, output_dir)

def generate_robustness_latex_table(summary_df, scenario, output_dir):
    """Generate LaTeX table for a single scenario robustness check."""
    latex_content = []
    
    # Table title
    latex_content.append(f"\\begin{{table}}[h]")
    latex_content.append(f"\\centering")
    latex_content.append(f"\\caption{{Double robustness check for OS-DML under the {scenario} scenario.}}")
    latex_content.append(f"\\label{{tab:robustness_{scenario.lower()}}}")
    latex_content.append(f"\\begin{{tabular}}{{lccccc}}")
    latex_content.append(f"\\toprule")
    
    # Header with multirow
    latex_content.append(f"\\multirow{{2}}{{*}}{{Nuisance Models}} & \\multirow{{2}}{{*}}{{Method}} & \\multicolumn{{5}}{{c}}{{Performance Metrics}} \\\\")
    latex_content.append(f"\\cmidrule(lr){{3-7}}")
    latex_content.append(f"Outcome & Propensity & Coverage & CI Width & Bias & RMSE & Runtime (s) \\\\")
    latex_content.append(f"\\midrule")
    
    # Data rows
    for _, row in summary_df.iterrows():
        misspec = row['misspecification']
        method = row['method']
        
        # Parse misspecification
        if misspec == 'correct_correct':
            outcome, propensity = 'Correct', 'Correct'
        elif misspec == 'correct_wrong':
            outcome, propensity = 'Correct', 'Wrong'
        elif misspec == 'wrong_correct':
            outcome, propensity = 'Wrong', 'Correct'
        elif misspec == 'wrong_wrong':
            outcome, propensity = 'Wrong', 'Wrong'
        else:
            outcome, propensity = misspec, misspec
        
        line = f"{outcome} & {propensity} & {method} & {row['Coverage']:.2f} & {row['CI_Width']:.4f} & {row['Bias']:.4f} & {row['RMSE']:.4f} & {row['Runtime']:.2f} \\\\"
        latex_content.append(line)
    
    latex_content.append(f"\\bottomrule")
    latex_content.append(f"\\end{{tabular}}")
    latex_content.append(f"\\end{{table}}")
    
    # Save to file
    latex_file = output_dir / f"robustness_{scenario}_table.tex"
    with open(latex_file, 'w') as f:
        f.write('\n'.join(latex_content))
    
    print(f"✓ Generated LaTeX table for {scenario}: {latex_file}")

def generate_robustness_csv_table(summary_df, scenario, output_dir):
    """Generate CSV table for a single scenario robustness check with proper formatting."""
    # Create a formatted DataFrame for CSV output
    formatted_df = summary_df.copy()
    
    # Add Outcome and Propensity columns
    formatted_df['Outcome'] = formatted_df['misspecification'].apply(
        lambda x: 'Correct' if x in ['correct_correct', 'correct_wrong'] else 'Wrong'
    )
    formatted_df['Propensity'] = formatted_df['misspecification'].apply(
        lambda x: 'Correct' if x in ['correct_correct', 'wrong_correct'] else 'Wrong'
    )
    
    # Reorder columns to match the desired format
    csv_df = formatted_df[['Outcome', 'Propensity', 'method', 'Coverage', 'CI_Width', 'Bias', 'RMSE', 'Runtime']].copy()
    
    # Rename columns for better readability
    csv_df.columns = ['Outcome', 'Propensity', 'Method', 'Coverage', 'CI Width', 'Bias', 'RMSE', 'Runtime (s)']
    
    # Save CSV
    csv_file = output_dir / f"robustness_{scenario}_table.csv"
    csv_df.to_csv(csv_file, index=False)
    print(f"✓ Generated CSV table for {scenario}: {csv_file}")
    
    return csv_df

def generate_combined_robustness_latex_table(summary_df, output_dir):
    """Generate combined LaTeX table for all scenarios robustness check."""
    latex_content = []
    
    # Table title
    latex_content.append(f"\\begin{{table}}[h]")
    latex_content.append(f"\\centering")
    latex_content.append(f"\\caption{{Combined double robustness check for OS-DML under RCT-2 and OBS-2 scenarios.}}")
    latex_content.append(f"\\label{{tab:robustness_combined}}")
    latex_content.append(f"\\begin{{tabular}}{{llccccc}}")
    latex_content.append(f"\\toprule")
    
    # Header
    latex_content.append(f"Scenario & Nuisance Models & Method & Coverage & CI Width & Bias & RMSE & Runtime (s) \\\\")
    latex_content.append(f"\\midrule")
    
    # Data rows
    for _, row in summary_df.iterrows():
        scenario = row['scenario']
        misspec = row['misspecification']
        method = row['method']
        
        # Parse misspecification
        if misspec == 'correct_correct':
            nuisance_models = 'Correct-Correct'
        elif misspec == 'correct_wrong':
            nuisance_models = 'Correct-Wrong'
        elif misspec == 'wrong_correct':
            nuisance_models = 'Wrong-Correct'
        elif misspec == 'wrong_wrong':
            nuisance_models = 'Wrong-Wrong'
        else:
            nuisance_models = misspec
        
        line = f"{scenario} & {nuisance_models} & {method} & {row['Coverage']:.2f} & {row['CI_Width']:.4f} & {row['Bias']:.4f} & {row['RMSE']:.4f} & {row['Runtime']:.2f} \\\\"
        latex_content.append(line)
    
    latex_content.append(f"\\bottomrule")
    latex_content.append(f"\\end{{tabular}}")
    latex_content.append(f"\\end{{table}}")
    
    # Save to file
    latex_file = output_dir / "robustness_combined_table.tex"
    with open(latex_file, 'w') as f:
        f.write('\n'.join(latex_content))
    
    print(f"✓ Generated combined LaTeX table: {latex_file}")

def generate_combined_robustness_csv_table(summary_df, output_dir):
    """Generate combined CSV table for all scenarios robustness check with proper formatting."""
    # Create a formatted DataFrame for CSV output
    formatted_df = summary_df.copy()
    
    # Add Outcome and Propensity columns
    formatted_df['Outcome'] = formatted_df['misspecification'].apply(
        lambda x: 'Correct' if x in ['correct_correct', 'correct_wrong'] else 'Wrong'
    )
    formatted_df['Propensity'] = formatted_df['misspecification'].apply(
        lambda x: 'Correct' if x in ['correct_correct', 'wrong_correct'] else 'Wrong'
    )
    
    # Reorder columns to match the desired format
    csv_df = formatted_df[['scenario', 'Outcome', 'Propensity', 'method', 'Coverage', 'CI_Width', 'Bias', 'RMSE', 'Runtime']].copy()
    
    # Rename columns for better readability
    csv_df.columns = ['Scenario', 'Outcome', 'Propensity', 'Method', 'Coverage', 'CI Width', 'Bias', 'RMSE', 'Runtime (s)']
    
    # Save CSV
    csv_file = output_dir / "robustness_combined_table.csv"
    csv_df.to_csv(csv_file, index=False)
    print(f"✓ Generated combined CSV table: {csv_file}")
    
    return csv_df

if __name__ == "__main__":
    main()

