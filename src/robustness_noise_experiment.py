
from pathlib import Path
import csv
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Experiment settings
NOISE_LEVELS = [0.005, 0.010, 0.015, 0.020]
RANDOM_SEEDS = [42, 123, 2026]
ALPHA = 0.75
THRESHOLD = 0.030
TAU_RE = 2e-3
TAU_IM = 0.02

# Load the original static background.
true_background = np.load(
    OUTPUTS / "synthetic_background.npy"
).astype(np.float64)

n, m = true_background.shape
T = 30
SQUARE_SIZE = 8
ROW0 = 19
START_COL = 0
COL_SPEED = 2
FOREGROUND_VALUE = 0.04


def generate_video(noise_std, seed):
    """Generate a noisy video and its ground-truth foreground mask."""
    rng = np.random.default_rng(seed)

    video = np.zeros((n, m, T), dtype=np.float64)
    mask = np.zeros((n, m, T), dtype=bool)

    for t in range(T):
        frame = true_background.copy()

        c1 = min(m - SQUARE_SIZE, START_COL + t * COL_SPEED)
        c2 = c1 + SQUARE_SIZE
        r1 = ROW0
        r2 = r1 + SQUARE_SIZE

        frame[r1:r2, c1:c2] += FOREGROUND_VALUE
        mask[r1:r2, c1:c2, t] = True

        noise = rng.normal(0, noise_std, size=(n, m))
        video[:, :, t] = np.clip(frame + noise, 0, 1)

    return video, mask


def run_fft_tdmd(video):
    """Run FFT-TDMD using the same core settings as tdmd_fft_noisy.py."""
    height, width, frames = video.shape

    row_mean = np.mean(video, axis=1, keepdims=True)
    video_centered = video - row_mean

    video_fft = np.fft.fft(video_centered, axis=1)
    background_fft = np.zeros_like(video_fft, dtype=np.complex128)

    for k in range(width):
        X = video_fft[:, k, :-1]
        Y = video_fft[:, k, 1:]

        U, s, Vh = np.linalg.svd(X, full_matrices=False)

        if len(s) == 0:
            continue

        tolerance = np.finfo(float).eps * max(X.shape) * s[0]
        rank = int(np.sum(s > tolerance))

        if rank == 0:
            continue

        U_r = U[:, :rank]
        S_r = np.diag(s[:rank])
        V_r = Vh.conj().T[:, :rank]

        A_tilde = U_r.conj().T @ Y @ V_r @ np.linalg.inv(S_r)
        eigenvalues, W = np.linalg.eig(A_tilde)
        modes = Y @ V_r @ np.linalg.inv(S_r) @ W

        b = np.linalg.lstsq(modes, X[:, 0], rcond=None)[0]
        time_indices = np.arange(frames - 1)

        dynamics = np.zeros(
            (rank, frames - 1), dtype=np.complex128
        )

        for j in range(rank):
            dynamics[j, :] = b[j] * (
                eigenvalues[j] ** time_indices
            )

        omega = np.zeros(rank, dtype=np.complex128)
        nonzero = np.abs(eigenvalues) > 1e-12
        omega[nonzero] = np.log(eigenvalues[nonzero])

        background_modes = (
            (np.abs(omega.real) < TAU_RE)
            & (np.abs(omega.imag) < TAU_IM)
        )

        if not np.any(background_modes):
            background_modes[np.argmin(np.abs(omega))] = True

        background_fft[:, k, :-1] = (
            modes[:, background_modes]
            @ dynamics[background_modes, :]
        )

    background_centered = np.fft.ifft(
        background_fft, axis=1
    ).real

    # Restore the estimated row mean, matching the existing implementation.
    background_mean = np.median(
        row_mean, axis=2, keepdims=True
    )
    background = background_centered + background_mean

    return background[:, :, :frames - 1]


def calculate_metrics(prediction, truth):
    tp = np.logical_and(prediction, truth).sum()
    fp = np.logical_and(prediction, ~truth).sum()
    fn = np.logical_and(~prediction, truth).sum()

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return precision, recall, f1


def evaluate_method(name, background, video, truth_mask):
    """Evaluate background error and foreground detection."""
    video_29 = video[:, :, :29]
    truth_29 = truth_mask[:, :, :29]

    if background.ndim == 2:
        background_29 = background[:, :, None]
    else:
        background_29 = background[:, :, :29]

    foreground = video_29 - background_29
    prediction = np.abs(foreground) > THRESHOLD

    precision, recall, f1 = calculate_metrics(
        prediction, truth_29
    )

    background_mae = np.mean(
        np.abs(background_29 - true_background[:, :, None])
    )

    return {
        "method": name,
        "background_mae": float(background_mae),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


results = []
total = len(NOISE_LEVELS) * len(RANDOM_SEEDS)
completed = 0

for noise_std in NOISE_LEVELS:
    for seed in RANDOM_SEEDS:
        completed += 1
        print(
            f"\nExperiment {completed}/{total}: "
            f"noise_std={noise_std:.3f}, seed={seed}"
        )

        video, mask = generate_video(noise_std, seed)

        # Run the existing FFT-TDMD method on this exact video.
        fft_background = run_fft_tdmd(video)

        # Temporal-median background baseline.
        median_background = np.median(video, axis=2)

        # Hybrid: 75% temporal median, 25% FFT-TDMD.
        hybrid_background = (
            ALPHA * median_background[:, :, None]
            + (1 - ALPHA) * fft_background
        )

        methods = [
            ("FFT-TDMD", fft_background),
            ("Temporal Median", median_background),
            ("Hybrid alpha=0.75", hybrid_background),
        ]

        for name, background in methods:
            metrics = evaluate_method(
                name, background, video, mask
            )

            row = {
                "noise_std": noise_std,
                "seed": seed,
                "threshold": THRESHOLD,
                **metrics,
            }
            results.append(row)

            print(
                f"  {name:18s} "
                f"MAE={metrics['background_mae']:.6f} "
                f"Precision={metrics['precision']:.4f} "
                f"Recall={metrics['recall']:.4f} "
                f"F1={metrics['f1']:.4f}"
            )

# Save detailed results without changing existing result files.
output_csv = RESULTS / "noise_robustness_metrics.csv"

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

# Print average metrics across all three seeds for each noise level.
print("\nAVERAGE RESULTS ACROSS RANDOM SEEDS")
print("Noise | Method             | Background MAE | Mean F1")

for noise_std in NOISE_LEVELS:
    for name, _ in [
        ("FFT-TDMD", None),
        ("Temporal Median", None),
        ("Hybrid alpha=0.75", None),
    ]:
        selected = [
            row for row in results
            if row["noise_std"] == noise_std
            and row["method"] == name
        ]

        mean_mae = np.mean(
            [row["background_mae"] for row in selected]
        )
        mean_f1 = np.mean(
            [row["f1"] for row in selected]
        )

        print(
            f"{noise_std:.3f} | {name:18s} | "
            f"{mean_mae:.6f}       | {mean_f1:.4f}"
        )

print("\nRobustness experiment completed.")
print("Results saved to:", output_csv)
