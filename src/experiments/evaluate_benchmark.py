
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.fft import dct, idct


# ============================================================
# Independent benchmark evaluator
# Existing project scripts are not modified.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "outputs" / "benchmark_v1"
RESULTS_DIR = ROOT / "results" / "benchmark_v1"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DT = 1.0
TAU_RE = 2e-3
TAU_IM = 0.02
HYBRID_MEDIAN_WEIGHT = 0.50


def estimate_median(video):
    """Temporal-median background baseline."""
    return np.median(video, axis=2)


def estimate_dmd_slice(Xi, Yi):
    """Fit a DMD model to one spatial/frequency slice."""
    U, singular_values, Vh = np.linalg.svd(
        Xi, full_matrices=False
    )

    if len(singular_values) == 0:
        return None

    # Retain numerically meaningful singular directions.
    tol = (
        np.finfo(float).eps
        * max(Xi.shape)
        * singular_values[0]
    )
    rank = max(1, int(np.sum(singular_values > tol)))

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
    """Select approximately stationary DMD modes."""
    omega = np.full(
        eigenvalues.shape,
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )

    valid = (
        np.isfinite(eigenvalues)
        & (np.abs(eigenvalues) > 1e-12)
    )

    omega[valid] = np.log(eigenvalues[valid]) / DT

    stationary = (
        valid
        & (np.abs(omega.real) < TAU_RE)
        & (np.abs(omega.imag) < TAU_IM)
    )

    # Match the existing implementations' fallback.
    if not np.any(stationary) and np.any(valid):
        indices = np.flatnonzero(valid)
        closest = indices[np.argmin(np.abs(omega[indices]))]
        stationary[closest] = True

    return stationary


def estimate_tensor_background(video, transform):
    """
    Estimate background with frequency-slice DMD.

    transform='fft'    : FFT-TDMD-style decomposition.
    transform='cosine' : Cosine-TDMD-style decomposition.
    """
    n, m, T = video.shape

    # Match the row-mean detrending used in the existing scripts.
    row_mean = np.mean(video, axis=1, keepdims=True)
    video_detrended = video - row_mean

    row_mean_bg = np.median(
        row_mean, axis=2, keepdims=True
    )
    background_rows = np.broadcast_to(
        row_mean_bg, (n, m, T)
    ).copy()

    X = np.transpose(video_detrended[:, :, :-1], (0, 2, 1))
    Y = np.transpose(video_detrended[:, :, 1:], (0, 2, 1))

    if transform == "fft":
        X_coeff = np.fft.fft(X, axis=2)
        Y_coeff = np.fft.fft(Y, axis=2)
        inverse_transform = lambda z: np.fft.ifft(
            z, axis=2
        ).real
        width = m

    elif transform == "cosine":
        width = 2 ** int(np.ceil(np.log2(2 * m)))

        X_coeff = np.pad(
            X, ((0, 0), (0, 0), (0, width - m))
        )
        Y_coeff = np.pad(
            Y, ((0, 0), (0, 0), (0, width - m))
        )

        X_coeff = dct(
            X_coeff, type=2, axis=2, norm="ortho"
        )
        Y_coeff = dct(
            Y_coeff, type=2, axis=2, norm="ortho"
        )

        inverse_transform = lambda z: idct(
            z, type=2, axis=2, norm="ortho"
        )

    else:
        raise ValueError(f"Unknown transform: {transform}")

    Z_background = np.zeros(
        (n, T, width), dtype=np.complex128
    )

    for k in range(width):
        fitted = estimate_dmd_slice(
            X_coeff[:, :, k], Y_coeff[:, :, k]
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
                * eigenvalues[selected, None] ** powers[None, :]
            )
        )

    background_padded = inverse_transform(Z_background)

    background_detrended = np.transpose(
        background_padded[:, :, :m], (0, 2, 1)
    ).real

    return background_detrended + background_rows


def estimate_standard_dmd_background(video):
    """
    Standard pixel-space DMD background estimate.

    This uses stationary-mode selection as an explicit
    background-extraction rule; it is not the same as
    evaluating the full Standard-DMD reconstruction.
    """
    n, m, T = video.shape

    X = video[:, :, :-1].reshape(n * m, T - 1)
    Y = video[:, :, 1:].reshape(n * m, T - 1)

    U, s, Vh = np.linalg.svd(X, full_matrices=False)

    tol = np.finfo(float).eps * max(X.shape) * s[0]
    rank = max(1, int(np.sum(s > tol)))

    U_r = U[:, :rank]
    s_r = s[:rank]
    V_r = Vh[:rank, :].T

    A_tilde = (U_r.T @ Y @ V_r) @ np.diag(1.0 / s_r)

    eigenvalues, W = np.linalg.eig(A_tilde)
    modes = U_r @ W
    amplitudes = np.linalg.pinv(modes) @ X[:, 0]

    selected = select_background_modes(eigenvalues)

    if not np.any(selected):
        # Defensive fallback: retain the mode nearest lambda=1.
        selected = np.zeros(len(eigenvalues), dtype=bool)
        selected[np.argmin(np.abs(eigenvalues - 1))] = True

    powers = np.arange(T)
    background = modes[:, selected] @ (
        amplitudes[selected, None]
        * eigenvalues[selected, None] ** powers[None, :]
    )

    return background.real.reshape(n, m, T)


def calculate_metrics(estimate, truth):
    """Calculate metrics against the noise-free true background."""
    if estimate.ndim == 3:
        estimate = np.mean(estimate, axis=2)

    difference = estimate - truth

    return {
        "MAE": float(np.mean(np.abs(difference))),
        "RMSE": float(np.sqrt(np.mean(difference ** 2))),
    }


def main():
    video_files = sorted(DATA_DIR.glob("*_video.npy"))

    if not video_files:
        raise FileNotFoundError(
            f"No benchmark videos found in {DATA_DIR}"
        )

    rows = []

    for video_path in video_files:
        name = video_path.name.replace("_video.npy", "")
        truth_path = DATA_DIR / f"{name}_background.npy"

        video = np.load(video_path)
        truth = np.load(truth_path)

        print(f"\nEvaluating: {name} {video.shape}")

        # Temporal median
        methods = {
            "TemporalMedian": lambda: estimate_median(video),
            "StandardDMD": lambda: estimate_standard_dmd_background(video),
            "FFT_TDMD": lambda: estimate_tensor_background(
                video, "fft"
            ),
            "Cosine_TDMD": lambda: estimate_tensor_background(
                video, "cosine"
            ),
        }

        estimates = {}

        for method_name, method in methods.items():
            start = time.perf_counter()
            estimate = method()
            elapsed = time.perf_counter() - start

            estimates[method_name] = estimate
            metrics = calculate_metrics(estimate, truth)

            rows.append({
                "scenario": name,
                "method": method_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "runtime_seconds": elapsed,
            })

            print(
                f"  {method_name:15s} "
                f"MAE={metrics['MAE']:.6f} "
                f"RMSE={metrics['RMSE']:.6f} "
                f"time={elapsed:.2f}s"
            )

        # Fixed hybrid weight chosen before inspecting results.
        alpha = HYBRID_MEDIAN_WEIGHT
        hybrid = (
            alpha * estimates["TemporalMedian"]
            + (1 - alpha) * estimates["Cosine_TDMD"]
        )

        metrics = calculate_metrics(hybrid, truth)

        rows.append({
            "scenario": name,
            "method": f"Hybrid_median_{alpha:.2f}",
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "runtime_seconds": np.nan,
        })

        print(
            f"  {'Hybrid':15s} "
            f"MAE={metrics['MAE']:.6f} "
            f"RMSE={metrics['RMSE']:.6f}"
        )

    results = pd.DataFrame(rows)

    output_csv = RESULTS_DIR / "benchmark_metrics.csv"
    results.to_csv(output_csv, index=False)

    summary = (
        results.groupby("method")[["MAE", "RMSE"]]
        .mean()
        .sort_values("MAE")
    )

    summary_path = RESULTS_DIR / "benchmark_summary.csv"
    summary.to_csv(summary_path)

    print("\n================ OVERALL SUMMARY ================")
    print(summary.to_string(float_format=lambda x: f"{x:.6f}"))
    print(f"\nDetailed results saved to: {output_csv}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
