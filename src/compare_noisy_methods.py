
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results"

# Load ground-truth foreground mask
mask = np.load(OUTPUTS / "noisy_challenge_foreground_mask.npy")
mask = mask[:, :, :29].astype(bool)

# Load foreground residuals from both methods
median_foreground = np.load(
    RESULTS / "noisy_median_foreground.npy"
)[:, :, :29]

fft_foreground = np.load(
    RESULTS / "noisy_fft_foreground.npy"
)[:, :, :29]

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

thresholds = [0.010, 0.015, 0.020, 0.025, 0.030, 0.040]

print("\nNoisy Challenge: Temporal Median vs FFT-TDMD")
print("Threshold | Method          | Precision | Recall | F1-score")

for threshold in thresholds:
    median_prediction = np.abs(median_foreground) > threshold
    fft_prediction = np.abs(fft_foreground) > threshold

    for name, prediction in [
        ("Temporal Median", median_prediction),
        ("FFT-TDMD", fft_prediction),
    ]:
        precision, recall, f1 = calculate_metrics(
            prediction, mask
        )

        print(
            f"{threshold:9.3f} | {name:15s} | "
            f"{precision:9.4f} | {recall:6.4f} | {f1:8.4f}"
        )


# Compare background estimates with the known static background
true_background = np.load(
    OUTPUTS / "synthetic_background.npy"
)

median_background = np.load(
    RESULTS / "noisy_median_background.npy"
)

fft_background = np.load(
    RESULTS / "noisy_fft_background.npy"
)

# Temporal median background may be 2D or 3D.
# If 3D, compare every frame with the true static background.
median_mae = np.mean(
    np.abs(median_background - true_background[..., None])
) if median_background.ndim == 3 else np.mean(
    np.abs(median_background - true_background)
)

# Compare only the first 29 FFT-TDMD frames
fft_background_29 = fft_background[:, :, :29]

fft_mae = np.mean(
    np.abs(fft_background_29 - true_background[:, :, None])
)

print("\nBackground MAE (lower is better)")
print(f"Temporal Median: {median_mae:.8f}")
print(f"FFT-TDMD:        {fft_mae:.8f}")
