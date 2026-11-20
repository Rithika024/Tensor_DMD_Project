from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Paths and settings
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = ROOT / "outputs" / "synthetic_video.npy"
RESULTS_DIR = ROOT / "results" / "dmd_rank_investigation"

WINDOW_LENGTH = 18
RANKS = [1, 2, 3, 5, 8, 10, 12, 14, 16, 18]

LAST_TRAIN_FRAME = 18  # Frame 19 (zero-based index)


def predict_next_frame(video, rank):
    start = LAST_TRAIN_FRAME - WINDOW_LENGTH

    X = video[:, :, start:LAST_TRAIN_FRAME].reshape(
        -1, WINDOW_LENGTH
    )
    Y = video[:, :, start + 1:LAST_TRAIN_FRAME + 1].reshape(
        -1, WINDOW_LENGTH
    )

    x_last = video[:, :, LAST_TRAIN_FRAME].reshape(-1)
    actual = video[:, :, LAST_TRAIN_FRAME + 1]

    U, singular_values, Vh = np.linalg.svd(
        X, full_matrices=False
    )

    U_r = U[:, :rank]
    s_r = singular_values[:rank]
    V_r = Vh[:rank, :].T

    A_tilde = (U_r.T @ Y @ V_r) / s_r[np.newaxis, :]

    eigenvalues, W = np.linalg.eig(A_tilde)

    Phi = (Y @ V_r) @ np.diag(1.0 / s_r) @ W

    amplitudes = np.linalg.pinv(Phi) @ x_last

    prediction = Phi @ (eigenvalues * amplitudes)
    prediction = np.real(prediction).reshape(video.shape[:2])

    mae = np.mean(np.abs(actual - prediction))
    rmse = np.sqrt(np.mean((actual - prediction) ** 2))
    relative_error = (
        np.linalg.norm(actual - prediction)
        / np.linalg.norm(actual)
    ) * 100

    return prediction, {
        "rank": rank,
        "MAE": mae,
        "RMSE": rmse,
        "relative_error_percent": relative_error,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    video = np.load(VIDEO_PATH).astype(float)

    rows = []

    for rank in RANKS:
        prediction, metrics = predict_next_frame(video, rank)

        np.save(
            RESULTS_DIR / f"prediction_rank_{rank}.npy",
            prediction,
        )

        rows.append(metrics)

        print(
            f"Rank: {rank:2d} | "
            f"MAE: {metrics['MAE']:.8f} | "
            f"RMSE: {metrics['RMSE']:.8f} | "
            f"Relative error: "
            f"{metrics['relative_error_percent']:.6f}%"
        )

    results = pd.DataFrame(rows)

    results.to_csv(
        RESULTS_DIR / "rank_comparison.csv",
        index=False,
    )

    print("\nResults saved to:")
    print(RESULTS_DIR / "rank_comparison.csv")

    print("\nRank comparison:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()