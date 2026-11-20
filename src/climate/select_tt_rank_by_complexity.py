from pathlib import Path
import pandas as pd

# ------------------------------------------------------------
# 1. Paths and configuration
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "climate"

INPUT_FILE = (
    RESULTS_DIR / "tt_dmd_rolling_validation_splits.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR / "tt_dmd_complexity_selection_first3.csv"
)

ACCEPTANCE_FACTOR = 1.05
BASELINE_NAME = "Full"

# Climate tensor spatial dimensions
N = 60
M = 60


# ------------------------------------------------------------
# 2. TT-core parameter count
# ------------------------------------------------------------
def tt_parameter_count(rank1, rank2, training_frames):
    """
    Number of scalar entries in the three TT cores:

    G1: (N, rank1)
    G2: (M, rank1, rank2)
    G3: (training_frames - 1, rank2)
    """
    return (
        N * rank1
        + M * rank1 * rank2
        + training_frames * rank2
    )


# ------------------------------------------------------------
# 3. Load rolling-validation results
# ------------------------------------------------------------
def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "split",
        "training_frames",
        "configuration",
        "rank1",
        "rank2",
        "relative_error_percent",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Preserve the original chronological split order.
    all_splits = df["split"].drop_duplicates().tolist()
    selected_splits = all_splits[:3]

    if len(selected_splits) != 3:
        raise ValueError(
            "At least three chronological splits are required."
        )

    selection_data = df[
        df["split"].isin(selected_splits)
    ].copy()

    print("=" * 65)
    print("TT-DMD COMPLEXITY-BASED RANK SELECTION")
    print("=" * 65)

    print("\nSplits used for selection:")
    for split in selected_splits:
        print(f"  {split}")

    # Each configuration must have one result per selected split.
    counts = selection_data.groupby(
        "configuration"
    )["split"].nunique()

    incomplete = counts[counts != 3]

    if not incomplete.empty:
        raise ValueError(
            "Some configurations do not have results "
            f"for all three splits:\n{incomplete}"
        )

    # --------------------------------------------------------
    # 4. Calculate TT-core parameter counts
    # --------------------------------------------------------
    selection_data["tt_core_parameters"] = selection_data.apply(
        lambda row: tt_parameter_count(
            int(row["rank1"]),
            int(row["rank2"]),
            int(row["training_frames"]) - 1,
        ),
        axis=1,
    )

    summary = (
        selection_data.groupby("configuration")
        .agg(
            mean_relative_error_percent=(
                "relative_error_percent", "mean"
            ),
            mean_tt_core_parameters=(
                "tt_core_parameters", "mean"
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # 5. Apply the 5% acceptance criterion
    # --------------------------------------------------------
    baseline = summary[
        summary["configuration"] == BASELINE_NAME
    ]

    if len(baseline) != 1:
        raise ValueError(
            "Exactly one Full baseline is required."
        )

    baseline_error = float(
        baseline.iloc[0]["mean_relative_error_percent"]
    )

    acceptance_limit = (
        baseline_error * ACCEPTANCE_FACTOR
    )

    summary["acceptance_limit_percent"] = acceptance_limit

    summary["within_acceptance_limit"] = (
        summary["mean_relative_error_percent"]
        <= acceptance_limit
    )

    summary["parameter_reduction_percent"] = (
        1
        - summary["mean_tt_core_parameters"]
        / float(baseline.iloc[0]["mean_tt_core_parameters"])
    ) * 100

    eligible = summary[
        summary["within_acceptance_limit"]
    ].copy()

    if eligible.empty:
        raise RuntimeError(
            "No configuration satisfies the acceptance criterion."
        )

    # Choose the eligible configuration with the fewest parameters.
    selected = eligible.sort_values(
        by=[
            "mean_tt_core_parameters",
            "mean_relative_error_percent",
        ]
    ).iloc[0]

    summary["selected"] = (
        summary["configuration"]
        == selected["configuration"]
    )

    # --------------------------------------------------------
    # 6. Save and display results
    # --------------------------------------------------------
    summary = summary.sort_values(
        by="mean_tt_core_parameters"
    )

    summary.to_csv(OUTPUT_FILE, index=False)

    print(f"\nFull baseline error: {baseline_error:.6f}%")
    print(f"Acceptance limit:    {acceptance_limit:.6f}%")

    print("\nCandidate comparison:")
    print(
        summary.to_string(
            index=False,
            formatters={
                "mean_relative_error_percent": "{:.6f}".format,
                "mean_tt_core_parameters": "{:.2f}".format,
                "acceptance_limit_percent": "{:.6f}".format,
                "parameter_reduction_percent": "{:.2f}".format,
            },
        )
    )

    print("\nSelected configuration:")
    print(f"  Name: {selected['configuration']}")
    print(
        f"  Mean relative error: "
        f"{selected['mean_relative_error_percent']:.6f}%"
    )
    print(
        f"  Mean TT-core parameters: "
        f"{selected['mean_tt_core_parameters']:.2f}"
    )
    print(
        f"  Parameter reduction: "
        f"{selected['parameter_reduction_percent']:.2f}%"
    )

    print(f"\nSaved selection results to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()


