import pandas as pd
import numpy as np
import os
import sys

# === Functions ===
# Format row with LaTeX
def format_values(values):
    # Transpose the list of lists to treat each position as a column
    columns = np.array(values).T
    formatted_columns = []

    for col in columns:
        rounded = np.round(col, 1)
        max_idx = np.argmax(rounded)
        second_max_idx = np.argsort(rounded)[-2]
        formatted = [f"{val:.1f}" for val in rounded]
        formatted[max_idx] = f"\\textbf{{{rounded[max_idx]:.1f}}}"
        formatted[second_max_idx] = f"\\underline{{{rounded[second_max_idx]:.1f}}}"
        formatted_columns.append(formatted)

    # Transpose back to match the original structure
    return np.array(formatted_columns).T.tolist()

def format_name(name):
    # Convert to lowercase and replace underscores with spaces
    name = name.lower().replace("_", "")
    # Capitalize the first letter of each word
    name = name.title()
    # If the name is longer than 4 characters, split into two rows
    if len(name) > 4:
        mid = len(name) // 2
        name = name[:mid] + " " + name[mid:]
        name = (
            "\\rotatebox[origin=c]{90}{\\begin{tabular}{@{}c@{}}" +
            " \\\\ ".join(name.split()) +
            "\\end{tabular}}\n"
        )
    else: 
        name = (
            "\\rotatebox[origin=c]{90}{" + name + "}\n"
        )    
    return name


# === BEANS ===
def beans_table(path):
    df = pd.read_csv(path, sep=',')

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

    datasets = ["beans_watkins", "beans_bats", "beans_cbi", "beans_dogs", "beans_humbugdb"]  # This will not change often



    # Delete old LaTeX file if it exists
    output_path = "projects/biofoundation/results/latex/beans_table.tex"
    if os.path.exists(output_path):
        os.remove(output_path)

    # === Top table part ===
    with open(output_path, "a") as f:
        f.write(
            "\\renewcommand{\\arraystretch}{0.6} % Increase row height\n"
            "\\setlength{\\tabcolsep}{2pt}\n"
            "% Color gradient for heatmap col\n\n"
            "\\begin{tabular}{p{2cm} p{1.2cm} | ccccc | ccccc !{\\vrule width 1.3pt}cc}\n"
            "    \\toprule\n"
            "    \\multicolumn{2}{c}{}                      & \\multicolumn{5}{c}{\\textbf{Linearprobing}} & \\multicolumn{5}{c}{\\textbf{Finetuning}}  & \\multicolumn{2}{c}{\\textbf{Score}} \\\\\n"
            "    \\addlinespace[2pt]\n"
            "    \\cline{3-14}\n"
            "    \\addlinespace[2pt]\n\n"
            "    \\multicolumn{1}{c}{}                      &                                            & \\cellcolor{gray!25}\\textbf{\\textsc{WTK}} & \\cellcolor{gray!25}\\textbf{\\textsc{BAT}} & \\cellcolor{gray!25}\\textbf{\\textsc{CBI}} & \\cellcolor{gray!25}\\textbf{\\textsc{DOG}} & \\cellcolor{gray!25}\\textbf{\\textsc{HDB}} & \\cellcolor{gray!25}\\textbf{\\textsc{WTK}} & \\cellcolor{gray!25}\\textbf{\\textsc{BAT}} & \\cellcolor{gray!25}\\textbf{\\textsc{CBI}} & \\cellcolor{gray!25}\\textbf{\\textsc{DOG}} & \\cellcolor{gray!25}\\textbf{\\textsc{HDB}} & \\cellcolor{gray!25}\\textbf{\\textsc{LP}} & \\cellcolor{gray!25}\\textbf{\\textsc{FT}} \\\\\n"
            "    \\midrule \\rule{0pt}{0.8em}\n"
        )


    # === Build table rows ===
    # Initialize lists to store all values for later processing
    all_top1_lp, all_top1_ft = [], []
    all_auroc_lp, all_auroc_ft = [], []
    all_avg_top1_lp, all_avg_top1_ft = [], []
    all_avg_auroc_lp, all_avg_auroc_ft = [], []

    # Collect data for all models
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

        # Averages
        avg_top1_lp = int(np.round(np.mean([x for x in top1_lp if x > 0]), 0)) if any(x > 0 for x in top1_lp) else 0
        avg_top1_ft = int(np.round(np.mean([x for x in top1_ft if x > 0]), 0)) if any(x > 0 for x in top1_ft) else 0
        avg_auroc_lp = int(np.round(np.mean([x for x in auroc_lp if x > 0]), 0)) if any(x > 0 for x in auroc_lp) else 0
        avg_auroc_ft = int(np.round(np.mean([x for x in auroc_ft if x > 0]), 0)) if any(x > 0 for x in auroc_ft) else 0

        # Store all values for later processing
        all_top1_lp.append(top1_lp)
        all_top1_ft.append(top1_ft)
        all_auroc_lp.append(auroc_lp)
        all_auroc_ft.append(auroc_ft)
        all_avg_top1_lp.append(avg_top1_lp)
        all_avg_top1_ft.append(avg_top1_ft)
        all_avg_auroc_lp.append(avg_auroc_lp)
        all_avg_auroc_ft.append(avg_auroc_ft)

    # Determine the highest and second highest values and write LaTeX
    all_top1_lp = format_values(all_top1_lp)
    all_top1_ft = format_values(all_top1_ft)
    all_auroc_lp = format_values(all_auroc_lp)
    all_auroc_ft = format_values(all_auroc_ft)

    with open(output_path, "a") as f:
        for i, model in enumerate(models):

            # Write LaTeX to a file
            f.write(f"\\multirow{{2}}{{*}}{{\\textbf{{{model.replace('_', ' ').title()}}}}} & {{Top-1}} & " +
                    " & ".join(all_top1_lp[i]) + f" & " + " & ".join(all_top1_ft[i]) +
                    f" & \\heatgreen{{{all_avg_top1_lp[i]}}} & \\heatgreen{{{all_avg_top1_ft[i]}}} \\\\ [0.1em]\n")

            f.write(f" & {{AUROC}} & " +
                    " & ".join(all_auroc_lp[i]) + f" & " + " & ".join(all_auroc_ft[i]) +
                    f" & \\heatblue{{{all_avg_auroc_lp[i]}}} & \\heatblue{{{all_avg_auroc_ft[i]}}} \\\\ [0.1em]")
            if model != models[-1]:
                f.write(f"\\hline \\rule{{0pt}}{{0.8em}}\n")

    # === Table end part ===
    with open(output_path, "a") as f:
        f.write("    \\bottomrule\n")
        f.write("\\end{tabular}\n")


# === BirdSet ===
def birdset_table(path):
    df = pd.read_csv(path, sep=',')

    # Rename for convenience
    df = df.rename(columns={
        'datamodule.dataset.dataset_name': 'Dataset',
        'module.network.model_name': 'Model',
        'tags': 'Tags',
        'test/MultilabelAUROC': 'AUROC',
        'test/T1Accuracy': 'Top1',
        'test/cmAP5': 'Cmap',
    })

    # Convert scores to percentage
    df['AUROC'] *= 100
    df['Top1'] *= 100
    df['Cmap'] *= 100

    # List of unique models and datasets
    df['Model'] = df['Model'].str.replace('eat_bs', 'eat') #! Name inconsistency
    models = sorted(df['Model'].unique(), key=lambda x: x.lower()) # This will change often

    datasets = ["PER", "POW", "NES", "UHH", "HSN", "NBP", "SSW", "SNE"]  # This will not change often



    # Delete old LaTeX file if it exists
    output_path = "projects/biofoundation/results/latex/birdset_table.tex"
    if os.path.exists(output_path):
        os.remove(output_path)

    # === Top table part ===
    with open(output_path, "a") as f:
        f.write(
            "\\renewcommand{\\arraystretch}{0.55} % Increase row height\n"
            "\\setlength{\\tabcolsep}{2pt}\n\n"
            "\\begin{tabular}{p{0.55cm} p{1.2cm} | cccccccc | cccccccc !{\\vrule width 1.3pt} cc}\n"
            "     \\toprule\n"
            "    \\multicolumn{2}{c}{}                     & \\multicolumn{8}{c}{\\textbf{Linearprobing}} & \\multicolumn{8}{c}{\\textbf{Finetuning}}  & \\textbf{Score} \\\\\n"
            "    \\addlinespace[2pt]\n"
            "    \\cline{3-20} % Ensuring cline matches actual columns\n"
            "    \\addlinespace[2pt]\n\n"
            "    \\multicolumn{2}{c}{}                     & \\cellcolor{gray!25}\\textbf{\\textsc{PER}}   & \\cellcolor{gray!25}\\textbf{\\textsc{POW}} & \\cellcolor{gray!25}\\textbf{\\textsc{NES}} & \\cellcolor{gray!25}\\textbf{\\textsc{UHH}} & \\cellcolor{gray!25}\\textbf{\\textsc{HSN}} & \\cellcolor{gray!25}\\textbf{\\textsc{NBP}} & \\cellcolor{gray!25}\\textbf{\\textsc{SSW}} & \\cellcolor{gray!25}\\textbf{\\textsc{SNE}} &\n"
            "    \\cellcolor{gray!25}\\textbf{\\textsc{PER}} & \\cellcolor{gray!25}\\textbf{\\textsc{POW}}   & \\cellcolor{gray!25}\\textbf{\\textsc{NES}} & \\cellcolor{gray!25}\\textbf{\\textsc{UHH}} & \\cellcolor{gray!25}\\textbf{\\textsc{HSN}} & \\cellcolor{gray!25}\\textbf{\\textsc{NBP}} & \\cellcolor{gray!25}\\textbf{\\textsc{SSW}} & \\cellcolor{gray!25}\\textbf{\\textsc{SNE}} & \\cellcolor{gray!25}\\textbf{LP}           & \\cellcolor{gray!25}\\textbf{FT}            \\\\\n"
            "    \\addlinespace[2pt]\n"
            "    \\cline{3-20} % Adjusting cline to match new column numbers\n"
            "    \\addlinespace[2pt]\n"
        )


    # === Build table rows ===
    # Initialize lists to store all values for later processing
    all_top1_lp, all_top1_ft = [], []
    all_auroc_lp, all_auroc_ft = [], []
    all_cmap_lp, all_cmap_ft = [], []
    all_avg_top1_lp, all_avg_top1_ft = [], []
    all_avg_auroc_lp, all_avg_auroc_ft = [], []
    all_avg_cmap_lp, all_avg_cmap_ft = [], []

    # Collect data for all models
    for model in models:
        top1_lp, top1_ft = [], []
        auroc_lp, auroc_ft = [], []
        cmap_lp, cmap_ft = [], []

        for dataset in datasets:
            lp_rows = df[(df['Model'] == model) & (df['Dataset'] == dataset) & (df['Tags'].str.contains('linearprobing'))]
            ft_rows = df[(df['Model'] == model) & (df['Dataset'] == dataset) & (df['Tags'].str.contains('finetune'))]
            
            top1_lp.append(lp_rows['Top1'].max() if not lp_rows.empty else 0)  # Max value for LP Top1
            top1_ft.append(ft_rows['Top1'].max() if not ft_rows.empty else 0)  # Max value for FT Top1
            auroc_lp.append(lp_rows['AUROC'].max() if not lp_rows.empty else 0)  # Max value for LP AUROC
            auroc_ft.append(ft_rows['AUROC'].max() if not ft_rows.empty else 0)  # Max value for FT AUROC
            cmap_lp.append(lp_rows['Cmap'].max() if not lp_rows.empty else 0)  # Max value for LP Cmap
            cmap_ft.append(ft_rows['Cmap'].max() if not ft_rows.empty else 0)  # Max value for FT Cmap

        # Averages
        avg_top1_lp = int(np.round(np.mean(top1_lp), 0))
        avg_top1_ft = int(np.round(np.mean(top1_ft), 0))
        avg_auroc_lp = int(np.round(np.mean(auroc_lp), 0))
        avg_auroc_ft = int(np.round(np.mean(auroc_ft), 0))
        avg_cmap_lp = int(np.round(np.mean(cmap_lp), 0))
        avg_cmap_ft = int(np.round(np.mean(cmap_ft), 0))

        # Store all values for later processing
        all_top1_lp.append(top1_lp)
        all_top1_ft.append(top1_ft)
        all_auroc_lp.append(auroc_lp)
        all_auroc_ft.append(auroc_ft)
        all_avg_top1_lp.append(avg_top1_lp)
        all_avg_top1_ft.append(avg_top1_ft)
        all_avg_auroc_lp.append(avg_auroc_lp)
        all_avg_auroc_ft.append(avg_auroc_ft)
        all_cmap_lp.append(cmap_lp)
        all_cmap_ft.append(cmap_ft)
        all_avg_cmap_lp.append(avg_cmap_lp)
        all_avg_cmap_ft.append(avg_cmap_ft)

    # Determine the highest and second highest values and write LaTeX
    all_top1_lp = format_values(all_top1_lp)
    all_top1_ft = format_values(all_top1_ft)
    all_auroc_lp = format_values(all_auroc_lp)
    all_auroc_ft = format_values(all_auroc_ft)
    all_cmap_lp = format_values(all_cmap_lp)
    all_cmap_ft = format_values(all_cmap_ft)

    with open(output_path, "a") as f:
        for i, model in enumerate(models):

            # Write LaTeX to a file
            f.write(f"\\multirow{{3}}{{*}}{{\\textbf{{{format_name(model)}}}}} & {{cmAP}} & " +
                    " & ".join(all_cmap_lp[i]) + f" & " + " & ".join(all_cmap_ft[i]) +
                    f" & \\heatred{{{all_avg_cmap_lp[i]}}} & \\heatred{{{all_avg_cmap_ft[i]}}} \\\\ [0.1em]\n")

            f.write(f" & {{AUROC}} & " +
                    " & ".join(all_auroc_lp[i]) + f" & " + " & ".join(all_auroc_ft[i]) +
                    f" & \\heatblue{{{all_avg_auroc_lp[i]}}} & \\heatblue{{{all_avg_auroc_ft[i]}}} \\\\ [0.1em]\n")
            
            f.write(f" & {{Top-1}} & " +
                    " & ".join(all_top1_lp[i]) + f" & " + " & ".join(all_top1_ft[i]) +
                    f" & \\heatgreen{{{all_avg_top1_lp[i]}}} & \\heatgreen{{{all_avg_top1_ft[i]}}} \\\\ [0.1em]")
            
            if model != models[-1]:
                f.write(f"\\hline \\rule{{0pt}}{{0.8em}}\n")

    # === Table end part ===
    with open(output_path, "a") as f:
        f.write("    \\bottomrule\n")
        f.write("\\end{tabular}\n")        

# === Command parser ===
# Get the first argument (after the script name)
arg = sys.argv[1]

print(f"Creating {arg} Latex table...")
if arg == "beans":
    CSV_PATH = "projects/biofoundation/results/latex/beans.csv"
    beans_table(CSV_PATH)

elif arg == "birdset":
    CSV_PATH = "projects/biofoundation/results/latex/birdset.csv"
    birdset_table(CSV_PATH)

else: 
    print("Invalid argument. Use 'beans' or 'birdset'.")
    sys.exit(1)