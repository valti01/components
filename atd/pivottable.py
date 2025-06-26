import pandas as pd


file_path = "atd_auc.xlsx"
sheet_name = "Sheet1"


df = pd.read_excel(file_path, sheet_name=sheet_name)


pivot_table = pd.pivot_table(
    df,
    values=['min_distance', 'max_distance', 'average_dsitance', 'clean_auc', 'in_auc', 'out_auc', 'a-auc'],
    index=['adaptive_min_distance', 'targetlabel'],
    aggfunc='mean'
)


output_path = "pivot_table_output.xlsx"
pivot_table.to_excel(output_path)

print(f"Pivot table created and saved to {output_path}")