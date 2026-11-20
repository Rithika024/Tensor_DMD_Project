from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.fft import dct, idct

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "outputs" / "benchmark_v1"
RESULTS_DIR = ROOT / "results" / "foreground_background_variants"

sys.path.insert(0, str(ROOT / "src" / "climate"))
from haar_transform import haar_blocks, inverse_haar_blocks


# ------------------------------------------------------------
# Shared evaluation settings
# ------------------------------------------------------------

DT = 1.0
TAU_RE = 2e-3
TAU_IM = 0.02
FOREGROUND_THRESHOLD = 0.10

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# DMD implementation shared by the three transforms
# ------------------------------------------------------------

def estimate_dmd_slice(Xi, Yi):
    U, singular_values, Vh = np.linalg.svd(
        Xi, full_matrices=False
    )

    if len(singular_values) == 0 or singular_values[0] == 0:
        return None

    tolerance = (
        np.finfo(float).eps
        * max(Xi.shape)
        * singular_values[0]
    )

    rank = max(1, int(np.sum(singular_values > tolerance)))

    U_r = U[:, :rank]
    s_r = singular_values[:rank]
    V_r = Vh[:rank, :].conj().T

    A_tilde = (
        U_r.conj().T @ Yi @ V_r
    ) @ np.diag(1.0 / s_r)

    eigenvalues, W = np.linalg.eig(A_tilde)
    modes = U_r @ W
    amplitudes = np.linalg.pinv(modes) @ Xi[:, 0]

    return eigenvalues, modes, amplitudes


def select_background_modes(eigenvalues):
    valid = (
        np.isfinite(eigenvalues)
        & (np.abs(eigenvalues) > 1e-12)
    )

    omega = np.full(
        eigenvalues.shape,
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )

    omega[valid] = np.log(eigenvalues[valid].astype(np.complex128)) / DT

    selected = (
        valid
        & (np.abs(omega.real) < TAU_RE)
        & (np.abs(omega.imag) < TAU_IM)
    )

    if not np.any(selected) and np.any(valid):
        indices = np.flatnonzero(valid)
        closest = indices[np.argmin(np.abs(omega[indices]))]
        selected[closest] = True

    return selected


# ------------------------------------------------------------
# Background estimators
# ------------------------------------------------------------

def estimate_temporal_median(video):
    background = np.median(video, axis=2)
    return np.repeat(
        background[:, :, np.newaxis],
        video.shape[2],
        axis=2,
    )


def estimate_tensor_background(video, transform):
    n, m, T = video.shape

    # Same row-mean detrending for FFT, Cosine and Haar.
    row_mean = np.mean(video, axis=1, keepdims=True)
    video_detrended = video - row_mean

    row_mean_bg = np.median(
        row_mean, axis=2, keepdims=True
    )

    background_rows = np.broadcast_to(
        row_mean_bg, (n, m, T)
    ).copy()

    X = np.transpose(
        video_detrended[:, :, :-1], (0, 2, 1)
    )
    Y = np.transpose(
        video_detrended[:, :, 1:], (0, 2, 1)
    )

    if transform == "fft":
        X_coeff = np.fft.fft(X, axis=2)
        Y_coeff = np.fft.fft(Y, axis=2)

        inverse_transform = lambda z: np.fft.ifft(
            z, axis=2
        ).real

    elif transform == "cosine":
        width = 2 ** int(np.ceil(np.log2(2 * m)))

        X_pad = np.pad(
            X, ((0, 0), (0, 0), (0, width - m))
        )
        Y_pad = np.pad(
            Y, ((0, 0), (0, 0), (0, width - m))
        )

        X_coeff = dct(
            X_pad, type=2, axis=2, norm="ortho"
        )
        Y_coeff = dct(
            Y_pad, type=2, axis=2, norm="ortho"
        )

        inverse_transform = lambda z: idct(
            z, type=2, axis=2, norm="ortho"
        )

    elif transform == "haar":
        # The transformed dimension is spatial width (60),
        # not the number of temporal snapshots (59).
        X_coeff = haar_blocks(X)
        Y_coeff = haar_blocks(Y)

        inverse_transform = inverse_haar_blocks

    else:
        raise ValueError(f"Unknown transform: {transform}")

    width = X_coeff.shape[2]

    Z_background = np.zeros(
        (n, T, width), dtype=np.complex128
    )

    for k in range(width):
        fitted = estimate_dmd_slice(
            X_coeff[:, :, k],
            Y_coeff[:, :, k],
        )

        if fitted is None:
            continue

        eigenvalues, modes, amplitudes = fitted
        selected = select_background_modes(eigenvalues)

        if not np.any(selected):
            continue

        powers = np.arange(T)

        Z_background[:, :, k] = (
            modes[:, selected]
            @ (
                amplitudes[selected, None]
                * eigenvalues[selected, None]
                ** powers[None, :]
            )
        )

    reconstructed = inverse_transform(Z_background)

    background_detrended = np.transpose(
        reconstructed[:, :, :m], (0, 2, 1)
    ).real

    return background_detrended + background_rows


# ------------------------------------------------------------
# Evaluation metrics
# ------------------------------------------------------------

def background_metrics(estimate, true_background):
    difference = estimate - true_background[:, :, None]

    return {
        "background_MAE": float(
            np.mean(np.abs(difference))
        ),
        "background_RMSE": float(
            np.sqrt(np.mean(difference ** 2))
        ),
    }


def foreground_metrics(video, estimated_background, true_mask):
    residual = np.maximum(
        video - estimated_background, 0.0
    )

    predicted_mask = residual > FOREGROUND_THRESHOLD

    tp = int(np.count_nonzero(predicted_mask & true_mask))
    fp = int(np.count_nonzero(predicted_mask & ~true_mask))
    fn = int(np.count_nonzero(~predicted_mask & true_mask))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }


# ------------------------------------------------------------
# Main benchmark
# ------------------------------------------------------------

def main():
    video_files = sorted(DATA_DIR.glob("*_video.npy"))

    if len(video_files) != 15:
        raise RuntimeError(
            f"Expected 15 videos, found {len(video_files)}"
        )

    methods = {
        "TemporalMedian": lambda video: estimate_temporal_median(video),
        "FFT_TDMD": lambda video: estimate_tensor_background(video, "fft"),
        "Cosine_TDMD": lambda video: estimate_tensor_background(video, "cosine"),
        "Haar_TDMD": lambda video: estimate_tensor_background(video, "haar"),
    }

    rows = []

    for video_path in video_files:
        name = video_path.name.replace("_video.npy", "")

        background_path = DATA_DIR / f"{name}_background.npy"
        mask_path = DATA_DIR / f"{name}_mask.npy"

        video = np.load(video_path)
        true_background = np.load(background_path)
        true_mask = np.load(mask_path)

        print(f"\nEvaluating {name} | shape={video.shape}")

        for method_name, estimator in methods.items():
            start = time.perf_counter()
            estimated_background = estimator(video)
            elapsed = time.perf_counter() - start

            if estimated_background.shape != video.shape:
                raise ValueError(
                    f"{method_name} returned "
                    f"{estimated_background.shape}, expected {video.shape}"
                )

            bg = background_metrics(
                estimated_background, true_background
            )

            fg = foreground_metrics(
                video, estimated_background, true_mask
            )

            row = {
                "scenario": name.rsplit("_noise_", 1)[0],
                "noise_sigma": float(name.rsplit("_noise_", 1)[1]),
                "method": method_name,
                **bg,
                **fg,
                "runtime_seconds": elapsed,
                "foreground_threshold": FOREGROUND_THRESHOLD,
            }

            rows.append(row)

            print(
                f"  {method_name:15s} "
                f"BG_MAE={bg['background_MAE']:.6f} "
                f"BG_RMSE={bg['background_RMSE']:.6f} "
                f"F1={fg['F1']:.4f} "
                f"time={elapsed:.2f}s"
            )

    results = pd.DataFrame(rows)

    detailed_path = RESULTS_DIR / "foreground_background_metrics.csv"
    results.to_csv(detailed_path, index=False)

    summary = (
        results.groupby("method")[
            [
                "background_MAE",
                "background_RMSE",
                "precision",
                "recall",
                "F1",
                "runtime_seconds",
            ]
        ]
        .mean()
        .sort_values("background_MAE")
    )

    summary_path = RESULTS_DIR / "summary_by_method.csv"
    summary.to_csv(summary_path)

    noise_summary = (
        results.groupby(["noise_sigma", "method"])[
            [
                "background_MAE",
                "background_RMSE",
                "precision",
                "recall",
                "F1",
            ]
        ].mean()
    )

    noise_path = RESULTS_DIR / "summary_by_noise.csv"
    noise_summary.to_csv(noise_path)

    scenario_summary = (
        results.groupby(["scenario", "method"])[
            [
                "background_MAE",
                "background_RMSE",
                "precision",
                "recall",
                "F1",
            ]
        ].mean()
    )

    scenario_path = RESULTS_DIR / "summary_by_scenario.csv"
    scenario_summary.to_csv(scenario_path)

    print("\n================ OVERALL METHOD SUMMARY ================")
    print(summary.to_string(float_format=lambda x: f"{x:.6f}"))

    print("\nSaved results:")
    print(detailed_path)
    print(summary_path)
    print(noise_path)
    print(scenario_path)


if __name__ == "__main__":
    main()

