
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "climate"

comparison_file = RESULTS_DIR / "climate_method_comparison.csv"
verification_file = RESULTS_DIR / "climate_method_comparison_verified.csv"
rolling_file = RESULTS_DIR / "tt_dmd_rolling_validation_summary.csv"

output_csv = RESULTS_DIR / "final_climate_comparison.csv"
output_plot = RESULTS_DIR / "final_climate_comparison.png"


# --------------------------------------------------
# 2. LOAD EXISTING RESULTS
# --------------------------------------------------

print("\nLoading existing experimental results...")

comparison = pd.read_csv(comparison_file)
verification = pd.read_csv(verification_file)
rolling = pd.read_csv(rolling_file)


# --------------------------------------------------
# 3. VERIFY METHOD ALIGNMENT
# --------------------------------------------------

print("\nChecking frame alignment...")

if len(verification) != 5:
    raise ValueError("Expected results for five methods.")

if not (verification["Frames evaluated"] == 11).all():
    raise ValueError("Not all methods evaluated 11 frames.")

if not (verification["First target frame"] == 2).all():
    raise ValueError("First target frame mismatch.")

if not (verification["Last target frame"] == 12).all():
    raise ValueError("Last target frame mismatch.")

if set(comparison["Method"]) != set(verification["Method"]):
    raise ValueError("Method names do not match.")

print("PASS: All five methods evaluated frames 2-12.")


# --------------------------------------------------
# 4. PREPARE FINAL METHOD COMPARISON
# --------------------------------------------------

final_df = comparison.copy()

final_df = final_df.rename(columns={
    "Method": "Method",
    "Mean MAE": "Mean_MAE",
    "Mean RMSE": "Mean_RMSE",
    "Mean frame relative error (%)":
        "Mean_frame_relative_error_percent",
    "Worst frame relative error (%)":
        "Worst_frame_relative_error_percent",
})

# Add common evaluation information
final_df["Frames_evaluated"] = 11
final_df["First_target_frame"] = 2
final_df["Last_target_frame"] = 12

# Sort by mean frame relative error
final_df = final_df.sort_values(
    "Mean_frame_relative_error_percent"
).reset_index(drop=True)

final_df.to_csv(output_csv, index=False)


# --------------------------------------------------
# 5. DISPLAY FINAL METHOD COMPARISON
# --------------------------------------------------

print("\nFINAL CLIMATE METHOD COMPARISON")
print("=" * 95)

print(final_df.to_string(index=False))


# --------------------------------------------------
# 6. PLOT COMPARISON RESULTS
# --------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

methods = final_df["Method"]

# Mean MAE
axes[0].bar(methods, final_df["Mean_MAE"])
axes[0].set_title("Mean MAE")
axes[0].set_ylabel("MAE")
axes[0].tick_params(axis="x", rotation=45)
axes[0].grid(axis="y", alpha=0.3)

# Mean RMSE
axes[1].bar(methods, final_df["Mean_RMSE"])
axes[1].set_title("Mean RMSE")
axes[1].set_ylabel("RMSE")
axes[1].tick_params(axis="x", rotation=45)
axes[1].grid(axis="y", alpha=0.3)

# Mean relative error
axes[2].bar(
    methods,
    final_df["Mean_frame_relative_error_percent"]
)
axes[2].set_title("Mean Frame Relative Error")
axes[2].set_ylabel("Relative Error (%)")
axes[2].tick_params(axis="x", rotation=45)
axes[2].grid(axis="y", alpha=0.3)

fig.suptitle(
    "Comparison of Five Tensor-DMD Methods",
    fontsize=15
)

plt.tight_layout()
plt.savefig(output_plot, dpi=300)
plt.close()


# --------------------------------------------------
# 7. DISPLAY ROLLING VALIDATION RESULTS
# --------------------------------------------------

print("\nTT-DMD ROLLING-ORIGIN VALIDATION")
print("=" * 75)

print(rolling.to_string(index=False))


# --------------------------------------------------
# 8. FINAL MESSAGES
# --------------------------------------------------

print("\nFinal comparison completed successfully!")

print("\nGenerated files:")
print(output_csv)
print(output_plot)

print("\nNote:")
print(
    "Reconstruction comparison uses original frames 2-12. "
    "Rolling-origin forecasting results are reported separately."
)
