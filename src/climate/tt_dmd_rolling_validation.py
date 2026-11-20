
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tt_dmd import load_climate_tensor, DATA_PATH, block_circulant
from tt_dmd_rank_selection import (
    truncated_tt_svd,
    contract_spatial_cores,
)


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "climate"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. FIT TT-DMD USING TRAINING FRAMES ONLY
# --------------------------------------------------

def fit_tt_dmd(training_video, rank1, rank2):

    n, m, train_frames = training_video.shape
    r = train_frames - 1

    # Consecutive training snapshots
    X_t = training_video[:, :, :-1]
    Y_t = training_video[:, :, 1:]

    # Arrange data for block-circulant construction
    X_sw = np.transpose(X_t, (0, 2, 1))
    Y_sw = np.transpose(Y_t, (0, 2, 1))

    Xb = block_circulant(X_sw)
    Yb = block_circulant(Y_sw)

    Xmat = Xb[:, :r]
    Ymat = Yb[:, :r]

    # TT decomposition
    G1, G2, G3 = truncated_tt_svd(X_t, rank1, rank2)

    # Contract spatial TT cores
    U = contract_spatial_cores(G1, G2)

    # SVD of temporal core
    Us, singular_values, Vt = np.linalg.svd(
        G3, full_matrices=False
    )

    S_tt = np.diag(singular_values)
    Veff = Vt.T
    Ueff = U @ Us

    # Reduced DMD operator
    Atil = (
        Ueff.T
        @ (Ymat @ Veff)
        @ np.linalg.pinv(S_tt)
    )

    eigenvalues, W = np.linalg.eig(Atil)

    Phi = (
        (Ymat @ Veff)
        @ np.linalg.pinv(S_tt)
        @ W
    )

    b = np.linalg.pinv(Phi) @ Xmat[:, 0]

    return Phi, eigenvalues, b


# --------------------------------------------------
# 3. FORECAST FUTURE FRAMES
# --------------------------------------------------


def forecast_tt_dmd(Phi, eigenvalues, b, train_frames, horizon):

    predictions = []

    for k in range(horizon):

        # DMD power for the future frame
        t = train_frames + k

        # Apply eigenvalue powers to modal amplitudes
        modal_amplitudes = (eigenvalues ** t) * b

        # Reconstruct the predicted snapshot
        x_pred = Phi @ modal_amplitudes

        x_pred = np.real(x_pred)

        frame = x_pred.reshape((60, 60), order="F")

        predictions.append(frame)

    return np.stack(predictions, axis=2)

# --------------------------------------------------
# 4. EVALUATION METRICS
# --------------------------------------------------

def calculate_metrics(actual, predicted):

    difference = actual - predicted

    mae = np.mean(np.abs(difference))
    rmse = np.sqrt(np.mean(difference ** 2))

    relative_error = (
        np.linalg.norm(difference)
        / np.linalg.norm(actual)
    ) * 100

    return mae, rmse, relative_error


# --------------------------------------------------
# 5. LOAD CLIMATE DATA
# --------------------------------------------------

print("\nLoading climate dataset...")

video = load_climate_tensor(DATA_PATH)

n, m, total_frames = video.shape

print("Dataset shape:", video.shape)
print("Total frames:", total_frames)


# --------------------------------------------------
# 6. DEFINE ROLLING VALIDATION SPLITS
# --------------------------------------------------

splits = [
    (6, 2),
    (7, 2),
    (8, 2),
    (9, 2),
    (10, 2),
]

# Each rank2 is capped by the available temporal rank.
rank_configs = [
    ("Full", 60, None),
    ("Medium", 20, 7),
    ("Low spatial", 10, 7),
    ("Compact", 10, 5),
    ("Medium compact", 20, 5),
    ("Small", 5, 5),
]


# --------------------------------------------------
# 7. RUN ROLLING-ORIGIN VALIDATION
# --------------------------------------------------

split_results = []

print("\nStarting rolling-origin validation...\n")

for train_frames, horizon in splits:

    print("=" * 60)
    print(
        f"Training frames: 1-{train_frames} | "
        f"Validation frames: {train_frames + 1}-"
        f"{train_frames + horizon}"
    )
    print("=" * 60)

    training_video = video[:, :, :train_frames]

    actual = video[
        :, :,
        train_frames:train_frames + horizon
    ]

    r = train_frames - 1

    for label, requested_rank1, requested_rank2 in rank_configs:

        rank1 = requested_rank1

        if requested_rank2 is None:
            rank2 = r
        else:
            rank2 = min(requested_rank2, r)

        start_time = time.perf_counter()

        Phi, eigenvalues, b = fit_tt_dmd(
            training_video,
            rank1,
            rank2
        )

        predicted = forecast_tt_dmd(
            Phi,
            eigenvalues,
            b,
            train_frames,
            horizon
        )

        runtime = time.perf_counter() - start_time

        mae, rmse, relative_error = calculate_metrics(
            actual, predicted
        )

        split_results.append({
            "split": f"1-{train_frames}",
            "training_frames": train_frames,
            "validation_frames": (
                f"{train_frames + 1}-"
                f"{train_frames + horizon}"
            ),
            "configuration": label,
            "rank1": rank1,
            "rank2": rank2,
            "MAE": mae,
            "RMSE": rmse,
            "relative_error_percent": relative_error,
            "runtime_seconds": runtime,
        })

        print(
            f"{label:16s} "
            f"Ranks=({rank1},{rank2}) | "
            f"Relative Error={relative_error:.6f}% | "
            f"Time={runtime:.4f}s"
        )


# --------------------------------------------------
# 8. SAVE SPLIT-WISE RESULTS
# --------------------------------------------------

split_df = pd.DataFrame(split_results)

split_csv = OUTPUT_DIR / "tt_dmd_rolling_validation_splits.csv"

split_df.to_csv(split_csv, index=False)


# --------------------------------------------------
# 9. AGGREGATE RESULTS ACROSS SPLITS
# --------------------------------------------------

summary_df = (
    split_df
    .groupby("configuration", sort=False)
    .agg(
        mean_MAE=("MAE", "mean"),
        mean_RMSE=("RMSE", "mean"),
        mean_relative_error_percent=(
            "relative_error_percent", "mean"
        ),
        mean_runtime_seconds=("runtime_seconds", "mean"),
        number_of_splits=("split", "count"),
    )
    .reset_index()
)

summary_csv = OUTPUT_DIR / "tt_dmd_rolling_validation_summary.csv"

summary_df.to_csv(summary_csv, index=False)


# --------------------------------------------------
# 10. DISPLAY FINAL RESULTS
# --------------------------------------------------

print("\n\nROLLING-ORIGIN VALIDATION SUMMARY")
print("=" * 90)

print(summary_df.to_string(index=False))


# --------------------------------------------------
# 11. PLOT RESULTS
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    summary_df["configuration"],
    summary_df["mean_relative_error_percent"]
)

plt.xlabel("TT-DMD Rank Configuration")
plt.ylabel("Mean Relative Error (%)")
plt.title("TT-DMD Rolling-Origin Validation")

plt.xticks(rotation=25)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plot_path = OUTPUT_DIR / "tt_dmd_rolling_validation.png"

plt.savefig(plot_path, dpi=300)
plt.close()


# --------------------------------------------------
# 12. COMPLETION MESSAGE
# --------------------------------------------------

print("\nRolling-origin validation completed!")

print("Saved files:")
print(split_csv)
print(summary_csv)
print(plot_path)
