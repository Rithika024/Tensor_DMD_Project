from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = ROOT / "outputs" / "synthetic_video.npy"
RESULTS_DIR = ROOT / "results" / "dmd_projection_investigation"

WINDOW_LENGTH = 18
RANK = 18
LAST_TRAIN_FRAME = 18  # Frame 19


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    video = np.load(VIDEO_PATH).astype(float)

    start = LAST_TRAIN_FRAME - WINDOW_LENGTH

    X = video[:, :, start:LAST_TRAIN_FRAME].reshape(
        -1, WINDOW_LENGTH
    )
    Y = video[:, :, start + 1:LAST_TRAIN_FRAME + 1].reshape(
        -1, WINDOW_LENGTH
    )

    actual = video[:, :, LAST_TRAIN_FRAME + 1].reshape(-1)
    x_last = video[:, :, LAST_TRAIN_FRAME].reshape(-1)

    # SVD of training snapshots
    U, singular_values, Vh = np.linalg.svd(
        X, full_matrices=False
    )

    U_r = U[:, :RANK]
    s_r = singular_values[:RANK]
    V_r = Vh[:RANK, :].T

    # Project actual frame 20 onto the learned spatial subspace
    spatial_projection = U_r @ (U_r.T @ actual)

    # Build reduced DMD model
    A_tilde = (U_r.T @ Y @ V_r) / s_r[np.newaxis, :]

    eigenvalues, W = np.linalg.eig(A_tilde)

    Phi = (Y @ V_r) @ np.diag(1.0 / s_r) @ W

    amplitudes = np.linalg.pinv(Phi) @ x_last

    # One-step DMD prediction of frame 20
    dmd_prediction = np.real(
        Phi @ (eigenvalues * amplitudes)
    )

    # Calculate error components
    spatial_error = actual - spatial_projection
    temporal_error = actual - dmd_prediction
    prediction_vs_projection = dmd_prediction - spatial_projection

    def metrics(error):
        return {
            "MAE": np.mean(np.abs(error)),
            "RMSE": np.sqrt(np.mean(error ** 2)),
            "relative_error_percent": (
                np.linalg.norm(error) / np.linalg.norm(actual)
            ) * 100,
        }

    results = {
        "Spatial projection error": metrics(spatial_error),
        "DMD prediction error": metrics(temporal_error),
        "DMD vs spatial projection": metrics(
            prediction_vs_projection
        ),
    }

    for name, values in results.items():
        print(f"\n{name}")
        print(f"MAE: {values['MAE']:.8f}")
        print(f"RMSE: {values['RMSE']:.8f}")
        print(
            f"Relative error: "
            f"{values['relative_error_percent']:.6f}%"
        )

    np.save(
        RESULTS_DIR / "actual_frame_20.npy",
        actual.reshape(video.shape[:2]),
    )
    np.save(
        RESULTS_DIR / "spatial_projection_frame_20.npy",
        spatial_projection.reshape(video.shape[:2]),
    )
    np.save(
        RESULTS_DIR / "dmd_prediction_frame_20.npy",
        dmd_prediction.reshape(video.shape[:2]),
    )

    pd.DataFrame(results).T.to_csv(
        RESULTS_DIR / "projection_diagnostics.csv"
    )

    print("\nSaved diagnostics to:")
    print(RESULTS_DIR)


if __name__ == "__main__":
    main()
    