import pandas as pd
import numpy as np
import os

# === Read CSV ===
csv_path = "projects/biofoundation/results/latex/beans.csv" 
df = pd.read_csv(csv_path, sep=',')

# Rename for convenience
df = df.rename(columns={
    'datamodule.dataset.dataset_name': 'Dataset',
    'module.network.model_name': 'Model',
    'tags': 'Tags',
    'test/AUROC': 'AUROC',
    'test/MulticlassAccuracy': 'Top1'
})

# Convert scores to percentage
df['AUROC'] *= 100
df['Top1'] *= 100

# List of unique models and datasets
models = sorted(df['Model'].unique(), key=lambda x: x.lower()) # This will change often

#datasets = df['Dataset'].unique()
datasets = ["beans_watkins", "beans_bats", "beans_cbi", "beans_dogs", "beans_humbugdb"]  # This will not change often

# Format row with LaTeX
def format_values(values):
    rounded = np.round(values, 1)
    max_idx = np.argmax(rounded)
    second_max_idx = np.argsort(rounded)[-2]
    formatted = [f"{val:.1f}" for val in rounded]
    formatted[max_idx] = f"\\textbf{{{rounded[max_idx]:.1f}}}"
    formatted[second_max_idx] = f"\\underline{{{rounded[second_max_idx]:.1f}}}"
    return formatted

# Delete old LaTeX file if it exists
output_path = "projects/biofoundation/results/latex/beans_table.tex"
if os.path.exists(output_path):
    os.remove(output_path)

# === Top table part ===
with open(output_path, "a") as f:
    f.write("\\renewcommand{\\arraystretch}{0.6} % Increase row height\n")
    f.write("\\setlength{\\tabcolsep}{2pt}\n")
    f.write("% Color gradient for heatmap col\n\n")
    f.write("\\begin{tabular}{p{2cm} p{1.2cm} | ccccc | ccccc !{\\vrule width 1.3pt}cc}\n")
    f.write("    \\toprule\n")
    f.write("    \\multicolumn{2}{c}{}                      & \\multicolumn{5}{c}{\\textbf{Linearprobing}} & \\multicolumn{5}{c}{\\textbf{Finetuning}}  & \\multicolumn{2}{c}{\\textbf{Score}} \\\\\n")
    f.write("    \\addlinespace[2pt]\n")
    f.write("    \\cline{3-14}\n")
    f.write("    \\addlinespace[2pt]\n\n")
    f.write("    \\multicolumn{1}{c}{}                      &                                            & \\cellcolor{gray!25}\\textbf{\\textsc{WTK}} & \\cellcolor{gray!25}\\textbf{\\textsc{BAT}} & \\cellcolor{gray!25}\\textbf{\\textsc{CBI}} & \\cellcolor{gray!25}\\textbf{\\textsc{DOG}} & \\cellcolor{gray!25}\\textbf{\\textsc{HDB}} & \\cellcolor{gray!25}\\textbf{\\textsc{WTK}} & \\cellcolor{gray!25}\\textbf{\\textsc{BAT}} & \\cellcolor{gray!25}\\textbf{\\textsc{CBI}} & \\cellcolor{gray!25}\\textbf{\\textsc{DOG}} & \\cellcolor{gray!25}\\textbf{\\textsc{HDB}} & \\cellcolor{gray!25}\\textbf{\\textsc{LP}} & \\cellcolor{gray!25}\\textbf{\\textsc{FT}} \\\\\n")
    f.write("    \\midrule \\rule{0pt}{0.8em}\n")


# === Build table rows ===
for model in models:
    top1_lp, top1_ft = [], []
    auroc_lp, auroc_ft = [], []

    for dataset in datasets:
        lp_rows = df[(df['Model'] == model) & (df['Dataset'] == dataset) & (df['Tags'].str.contains('linearprobing'))]
        ft_rows = df[(df['Model'] == model) & (df['Dataset'] == dataset) & (df['Tags'].str.contains('finetune'))]
        
        top1_lp.append(lp_rows['Top1'].max() if not lp_rows.empty else 0)  # Max value for LP Top1
        top1_ft.append(ft_rows['Top1'].max() if not ft_rows.empty else 0)  # Max value for FT Top1
        auroc_lp.append(lp_rows['AUROC'].max() if not lp_rows.empty else 0)  # Max value for LP AUROC
        auroc_ft.append(ft_rows['AUROC'].max() if not ft_rows.empty else 0)  # Max value for FT AUROC

    # Format with LaTeX
    fmt_top1_lp = format_values(top1_lp)
    fmt_top1_ft = format_values(top1_ft)
    fmt_auroc_lp = format_values(auroc_lp)
    fmt_auroc_ft = format_values(auroc_ft)

    # Averages
    avg_top1_lp = int(np.round(np.mean(top1_lp), 0))
    avg_top1_ft = int(np.round(np.mean(top1_ft), 0))
    avg_auroc_lp = int(np.round(np.mean(auroc_lp), 0))
    avg_auroc_ft = int(np.round(np.mean(auroc_ft), 0))

    # Write LaTeX to a file
    with open(output_path, "a") as f:
        f.write(f"\\multirow{{2}}{{*}}{{\\textbf{{{model.replace('_', ' ').title()}}}}} & {{Top-1}} & " +
            " & ".join(fmt_top1_lp) + f" & " + " & ".join(fmt_top1_ft) +
            f" & \\heatgreen{{{avg_top1_lp}}} & \\heatgreen{{{avg_top1_ft}}} \\\\ [0.1em]\n")

        f.write(f" & {{AUROC}} & " +
                " & ".join(fmt_auroc_lp) + f" & " + " & ".join(fmt_auroc_ft) +
                f" & \\heatblue{{{avg_auroc_lp}}} & \\heatblue{{{avg_auroc_ft}}} \\\\ [0.1em]")
        if model != models[-1]:
            f.write(f"\\hline \\rule{{0pt}}{{0.8em}}\n")

# === Table end part ===
with open(output_path, "a") as f:
    f.write("    \\bottomrule\n")
    f.write("\\end{tabular}\n")