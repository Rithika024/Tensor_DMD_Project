from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"

video = np.load(OUTPUT_DIR / "challenge_video.npy")[:, :, :-1]
ground_truth = np.load(
    OUTPUT_DIR / "challenge_foreground_mask.npy"
)[:, :, :-1].astype(bool)

median_background = np.load(
    RESULTS_DIR / "challenge_median_background.npy"
)[:, :, :-1]

fft_background = np.load(
    RESULTS_DIR / "challenge_fft_background.npy"
)

# Estimate foreground using each method's background estimate.
median_foreground = np.maximum(
    video - median_background, 0
)
fft_foreground = np.maximum(
    video - fft_background, 0
)

def calculate_metrics(predicted, actual):
    tp = np.count_nonzero(predicted & actual)
    fp = np.count_nonzero(predicted & ~actual)
    fn = np.count_nonzero(~predicted & actual)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return precision, recall, f1, tp, fp, fn

def background_mae(estimated):
    true_background = np.load(
        OUTPUT_DIR / "synthetic_background.npy"
    )
    true_background = np.broadcast_to(
        true_background[:, :, None], estimated.shape
    )
    return np.mean(np.abs(estimated - true_background))

print("Challenge comparison: first 29 frames")
print("Foreground threshold | Method          | Precision | Recall | F1-score")

thresholds = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08]

for threshold in thresholds:
    for name, foreground in [
        ("Temporal Median", median_foreground),
        ("FFT-TDMD", fft_foreground),
    ]:
        predicted = foreground > threshold
        precision, recall, f1, tp, fp, fn = calculate_metrics(
            predicted, ground_truth
        )

        print(
            f"{threshold:19.3f} | {name:15s} | "
            f"{precision:9.4f} | {recall:6.4f} | {f1:8.4f}"
        )

print("\nBackground MAE (lower is better)")
print(f"Temporal Median: {background_mae(median_background):.8f}")
print(f"FFT-TDMD:        {background_mae(fft_background):.8f}")

print("\nComparison completed.")
