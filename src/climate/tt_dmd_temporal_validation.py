
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Paths and imports
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIMATE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(CLIMATE_DIR))

from tt_dmd import (
    load_climate_tensor,
    DATA_PATH,
    block_circulant,
)

from tt_dmd_rank_selection import (
    truncated_tt_svd,
    contract_spatial_cores,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "climate"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Train DMD using training frames only
# ------------------------------------------------------------
def fit_tt_dmd(training_video, rank1, rank2):
    n, m, train_frames = training_video.shape
    r = train_frames - 1

    # Training snapshot pairs
    X_t = training_video[:, :, :-1]
    Y_t = training_video[:, :, 1:]

    X_sw = np.transpose(X_t, (0, 2, 1))
    Y_sw = np.transpose(Y_t, (0, 2, 1))

    Xb = block_circulant(X_sw)
    Yb = block_circulant(Y_sw)

    Xmat = Xb[:, :r]
    Ymat = Yb[:, :r]

    # TT decomposition uses training data only
    G1, G2, G3 = truncated_tt_svd(
        X_t, rank1, rank2
    )

    U = contract_spatial_cores(G1, G2)

    # Reduced SVD of temporal core
    Us, singular_values, Vh = np.linalg.svd(
        G3, full_matrices=False
    )

    Vs = Vh.T
    S_tt = np.diag(singular_values)

    Ueff = U @ Us
    Veff = Vs

    S_pinv = np.linalg.pinv(S_tt)

    Atil = Ueff.T @ (Ymat @ Veff) @ S_pinv

    eigenvalues, W = np.linalg.eig(Atil)

    Phi = (Ymat @ Veff) @ S_pinv @ W

    # Initial amplitude from first training snapshot
    b = np.linalg.pinv(Phi) @ Xmat[:, 0]

    return eigenvalues, Phi, b


# ------------------------------------------------------------
# 3. Forecast future frames
# ------------------------------------------------------------
def forecast_tt_dmd(
    training_video,
    rank1,
    rank2,
    total_frames,
):
    n, m, train_frames = training_video.shape

    eigenvalues, Phi, b = fit_tt_dmd(
        training_video, rank1, rank2
    )

    # DMD time powers:
    # t=1 corresponds to original frame 2.
    # t=train_frames-1 corresponds to training frame 8.
    # Future frames continue with larger powers.
    predictions = np.zeros(
        (n, m, total_frames - train_frames),
        dtype=np.float64,
    )

    for frame_number in range(
        train_frames + 1, total_frames + 1
    ):
        t = frame_number - 1

        dynamics = np.power(eigenvalues, t) * b
        x = Phi @ dynamics

        predictions[:, :, frame_number - train_frames - 1] = (
            np.real(x.reshape((n, m), order="F"))
        )

    return predictions


# ------------------------------------------------------------
# 4. Calculate validation metrics
# ------------------------------------------------------------
def calculate_validation_metrics(prediction, truth):
    rows = []

    for i in range(truth.shape[2]):
        actual = truth[:, :, i]
        estimated = prediction[:, :, i]

        difference = estimated - actual

        mae = np.mean(np.abs(difference))
        rmse = np.sqrt(np.mean(difference ** 2))

        relative_error = (
            100 * np.linalg.norm(difference)
            / max(np.linalg.norm(actual), np.finfo(float).eps)
        )

        rows.append({
            "frame": i + 9,
            "MAE": float(mae),
            "RMSE": float(rmse),
            "relative_error_percent": float(relative_error),
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 5. Main experiment
# ------------------------------------------------------------
def main():
    start_time = time.perf_counter()

    print("=" * 65)
    print("TT-DMD TEMPORAL VALIDATION")
    print("=" * 65)

    video = load_climate_tensor(DATA_PATH)

    n, m, total_frames = video.shape

    # Chronological split
    train_frames = 8

    training_video = video[:, :, :train_frames]
    validation_video = video[:, :, train_frames:]

    print(f"\nComplete dataset: {video.shape}")
    print(f"Training data:    {training_video.shape}")
    print(f"Validation data:  {validation_video.shape}")

    print("\nTraining frames:   1-8")
    print("Validation frames: 9-12")

    # Full-rank baseline and truncated configurations.
    rank_pairs = [
        (60, 7),  # Full-rank baseline
        (20, 7),
        (10, 7),
        (10, 5),
        (20, 5),
        (5, 5),
    ]

    all_metrics = []
    summaries = []
    saved_predictions = {}

    for rank1, rank2 in rank_pairs:
        print("\n" + "-" * 55)
        print(f"Testing rank1={rank1}, rank2={rank2}")

        experiment_start = time.perf_counter()

        try:
            prediction = forecast_tt_dmd(
                training_video,
                rank1,
                rank2,
                total_frames,
            )

            metrics = calculate_validation_metrics(
                prediction,
                validation_video,
            )

            metrics["rank1"] = rank1
            metrics["rank2"] = rank2

            all_metrics.append(metrics)

            mean_mae = metrics["MAE"].mean()
            mean_rmse = metrics["RMSE"].mean()
            mean_relative = metrics[
                "relative_error_percent"
            ].mean()

            runtime = time.perf_counter() - experiment_start

            summaries.append({
                "rank1": rank1,
                "rank2": rank2,
                "mean_MAE": mean_mae,
                "mean_RMSE": mean_rmse,
                "mean_relative_error_percent": mean_relative,
                "runtime_seconds": runtime,
                "status": "success",
            })

            saved_predictions[(rank1, rank2)] = prediction

            print(f"Mean MAE: {mean_mae:.8f}")
            print(f"Mean RMSE: {mean_rmse:.8f}")
            print(f"Mean relative error: {mean_relative:.6f}%")
            print(f"Runtime: {runtime:.4f} seconds")

        except (np.linalg.LinAlgError, ValueError) as error:
            print(f"Experiment failed: {error}")

            summaries.append({
                "rank1": rank1,
                "rank2": rank2,
                "mean_MAE": np.nan,
                "mean_RMSE": np.nan,
                "mean_relative_error_percent": np.nan,
                "runtime_seconds": time.perf_counter() - experiment_start,
                "status": str(error),
            })

    # --------------------------------------------------------
    # 6. Save results
    # --------------------------------------------------------
    summary_df = pd.DataFrame(summaries)

    summary_path = (
        RESULTS_DIR / "tt_dmd_temporal_validation_summary.csv"
    )

    summary_df.to_csv(summary_path, index=False)

    if all_metrics:
        frame_metrics = pd.concat(
            all_metrics, ignore_index=True
        )

        frame_metrics_path = (
            RESULTS_DIR / "tt_dmd_temporal_validation_frames.csv"
        )

        frame_metrics.to_csv(
            frame_metrics_path, index=False
        )

    successful = summary_df[
        summary_df["status"] == "success"
    ]

    if successful.empty:
        raise RuntimeError("All validation experiments failed.")

    baseline = successful[
        (successful["rank1"] == 60)
        & (successful["rank2"] == 7)
    ]

    if baseline.empty:
        raise RuntimeError("Full-rank baseline was not completed.")

    baseline_error = float(
        baseline.iloc[0]["mean_relative_error_percent"]
    )

    print("\n" + "=" * 65)
    print("TEMPORAL VALIDATION RESULTS")
    print("=" * 65)

    print(summary_df.to_string(index=False))

    # Select the best rank using validation error.
    best = successful.loc[
        successful["mean_relative_error_percent"].idxmin()
    ]

    print("\nBest validation configuration:")
    print(f"rank1: {int(best['rank1'])}")
    print(f"rank2: {int(best['rank2'])}")
    print(
        "Validation relative error: "
        f"{best['mean_relative_error_percent']:.6f}%"
    )

    improvement = (
        (baseline_error - best["mean_relative_error_percent"])
        / max(baseline_error, np.finfo(float).eps)
    ) * 100

    print(
        f"Relative improvement over full-rank baseline: "
        f"{improvement:.4f}%"
    )

    # Save best validation prediction
    best_key = (int(best["rank1"]), int(best["rank2"]))

    np.save(
        RESULTS_DIR / "tt_dmd_temporal_validation_best.npy",
        saved_predictions[best_key],
    )

    # --------------------------------------------------------
    # 7. Plot validation errors
    # --------------------------------------------------------
    plt.figure(figsize=(10, 6))

    labels = [
        f"({int(row.rank1)}, {int(row.rank2)})"
        for row in successful.itertuples()
    ]

    plt.bar(
        labels,
        successful["mean_relative_error_percent"],
    )

    plt.xlabel("(rank1, rank2)")
    plt.ylabel("Mean validation relative error (%)")
    plt.title("TT-DMD Temporal Validation")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plot_path = (
        RESULTS_DIR / "tt_dmd_temporal_validation.png"
    )

    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("\nSaved files:")
    print(summary_path)
    print(frame_metrics_path)
    print(plot_path)
    print(
        RESULTS_DIR
        / "tt_dmd_temporal_validation_best.npy"
    )

    print(
        f"\nTotal runtime: "
        f"{time.perf_counter() - start_time:.4f} seconds"
    )

    print("\nTemporal validation completed.")


if __name__ == "__main__":
    main()
