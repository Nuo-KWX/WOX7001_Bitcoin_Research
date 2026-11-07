
import os
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

# 1. Define file paths
results_folder = "outputs/model_results"

output_file = os.path.join(
    results_folder,
    "Bitcoin_Research_Results.xlsx"
)

# 2. Define CSV files and Excel sheet names
files = {
    "Descriptive Stats": "descriptive_statistics.csv",
    "Normality Test": "normality_test.csv",
    "Final Model Comparison": "final_model_comparison.csv",
    "Volatility Regime": "volatility_regime_results.csv",
    "Permutation Importance": "feature_importance.csv",
    "SHAP Importance": "shap_feature_importance.csv",
    "LR Predictions": "final_linear_regression_predictions.csv",
    "RF Predictions": "final_random_forest_predictions.csv",
    "XGBoost Predictions": "final_xgboost_predictions.csv"
}

# 3. Create Excel workbook
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

    for sheet_name, csv_file in files.items():

        file_path = os.path.join(results_folder, csv_file)

        if not os.path.exists(file_path):
            print(f"WARNING: File not found: {csv_file}")
            continue

        df = pd.read_csv(file_path)

        # Remove unnecessary exported index columns
        df = df.loc[
            :, ~df.columns.astype(str).str.startswith("Unnamed:")
        ]

        df.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False
        )

        print(f"Added sheet: {sheet_name}")

# 4. Format Excel workbook
workbook = load_workbook(output_file)

for worksheet in workbook.worksheets:

    # Freeze the header row
    worksheet.freeze_panes = "A2"

    # Add filter
    worksheet.auto_filter.ref = worksheet.dimensions

    # Format header
    for cell in worksheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill(
            start_color="1F4E78",
            end_color="1F4E78",
            fill_type="solid"
        )
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center"
        )

    # Adjust column widths
    for column in worksheet.columns:

        max_length = 0
        column_letter = get_column_letter(
            column[0].column
        )

        for cell in column:
            if cell.value is not None:
                max_length = max(
                    max_length,
                    len(str(cell.value))
                )

        worksheet.column_dimensions[
            column_letter
        ].width = min(max_length + 3, 35)

workbook.save(output_file)

print("\n================================")
print("EXCEL EXPORT COMPLETED!")
print(f"Saved to: {output_file}")
print("================================")
