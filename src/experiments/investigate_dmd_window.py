from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Paths and settings
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = ROOT / "outputs" / "synthetic_video.npy"
RESULTS_DIR = ROOT / "results" / "dmd_window_investigation"

WINDOW_LENGTHS = [10, 12, 14, 16, 18]

# Zero-based index 18 corresponds to frame 19.
# Frame 20 (index 19) is reserved as the prediction target.
LAST_TRAIN_FRAME = 18


def predict_next_frame(video, window_length):
    """Fit exact DMD on a trailing window and predict frame 20."""

    start = LAST_TRAIN_FRAME - window_length

    # Training snapshot pairs:
    # X: frames before frame 19
    # Y: corresponding next frames, ending at frame 19
    X = video[:, :, start:LAST_TRAIN_FRAME].reshape(
        -1, window_length
    )
    Y = video[:, :, start + 1:LAST_TRAIN_FRAME + 1].reshape(
        -1, window_length
    )

    # Last observed frame: frame 19
    x_last = video[:, :, LAST_TRAIN_FRAME].reshape(-1)

    # SVD
    U, singular_values, Vh = np.linalg.svd(
        X, full_matrices=False
    )

    tolerance = (
        np.finfo(float).eps
        * max(X.shape)
        * singular_values[0]
    )
    rank = int(np.sum(singular_values > tolerance))

    U_r = U[:, :rank]
    s_r = singular_values[:rank]
    V_r = Vh[:rank, :].T

    # Reduced DMD operator
    A_tilde = (
        U_r.T @ Y @ V_r
    ) / s_r[np.newaxis, :]

    eigenvalues, W = np.linalg.eig(A_tilde)

    # Exact DMD modes
    Phi = (
        Y @ V_r
    ) @ np.diag(1.0 / s_r) @ W

    # Fit amplitudes using the last observed frame
    amplitudes = np.linalg.pinv(Phi) @ x_last

    # One-step prediction: frame 19 -> frame 20
    prediction = Phi @ (eigenvalues * amplitudes)
    prediction = np.real(prediction).reshape(video.shape[:2])

    actual = video[:, :, LAST_TRAIN_FRAME + 1]

    mae = np.mean(np.abs(actual - prediction))
    rmse = np.sqrt(np.mean((actual - prediction) ** 2))

    relative_error = (
        np.linalg.norm(actual - prediction)
        / np.linalg.norm(actual)
    )

    return prediction, {
        "window_pairs": window_length,
        "numerical_rank": rank,
        "MAE": mae,
        "RMSE": rmse,
        "relative_error_percent": relative_error * 100,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    video = np.load(VIDEO_PATH).astype(float)

    if video.ndim != 3 or video.shape[2] != 20:
        raise ValueError(
            f"Expected a 3D video with 20 frames; got {video.shape}"
        )

    rows = []

    # Persistence baseline: predict frame 20 as frame 19
    persistence = video[:, :, 18]
    actual = video[:, :, 19]

    rows.append({
        "window_pairs": 0,
        "numerical_rank": 0,
        "MAE": np.mean(np.abs(actual - persistence)),
        "RMSE": np.sqrt(np.mean((actual - persistence) ** 2)),
        "relative_error_percent": (
            np.linalg.norm(actual - persistence)
            / np.linalg.norm(actual)
        ) * 100,
    })

    for length in WINDOW_LENGTHS:
        prediction, metrics = predict_next_frame(video, length)

        np.save(
            RESULTS_DIR / f"prediction_window_{length}.npy",
            prediction,
        )

        rows.append(metrics)

        print(
            f"Window: {length:2d} pairs | "
            f"Rank: {metrics['numerical_rank']:2d} | "
            f"MAE: {metrics['MAE']:.8f} | "
            f"RMSE: {metrics['RMSE']:.8f} | "
            f"Relative error: "
            f"{metrics['relative_error_percent']:.6f}%"
        )

    results = pd.DataFrame(rows)
    results.to_csv(
        RESULTS_DIR / "window_comparison.csv",
        index=False,
    )

    print("\nResults saved to:")
    print(RESULTS_DIR / "window_comparison.csv")
    print("\nComparison:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
    