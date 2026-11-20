
from pathlib import Path
import csv
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Evaluate the same 29 frames for every method.
video = np.load(OUTPUTS / "challenge_video.npy")[:, :, :29]
truth = np.load(
    OUTPUTS / "challenge_foreground_mask.npy"
)[:, :, :29].astype(bool)

true_background = np.load(OUTPUTS / "synthetic_background.npy")

median_bg = np.load(
    RESULTS / "challenge_median_background.npy"
)[:, :, :29]

fft_bg = np.load(RESULTS / "challenge_fft_background.npy")

if not (
    video.shape == truth.shape == median_bg.shape == fft_bg.shape
):
    raise ValueError(
        f"Shape mismatch: video={video.shape}, "
        f"truth={truth.shape}, median={median_bg.shape}, "
        f"FFT={fft_bg.shape}"
    )

# Ground-truth background repeated over the evaluated frames.
true_bg_video = np.broadcast_to(true_background[:, :, None], video.shape)

def metrics(foreground, threshold):
    prediction = foreground > threshold

    tp = np.count_nonzero(prediction & truth)
    fp = np.count_nonzero(prediction & ~truth)
    fn = np.count_nonzero(~prediction & truth)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return precision, recall, f1

# Test the baseline methods and several hybrid weights.
weights = [0.0, 0.25, 0.5, 0.75, 1.0]
thresholds = [0.005, 0.01, 0.02, 0.04, 0.06]

rows = []

print("HYBRID CHALLENGE EXPERIMENT")
print("Evaluated frames: 1-29")
print("alpha | threshold | background MAE | precision | recall | F1")

for alpha in weights:
    background = alpha * median_bg + (1 - alpha) * fft_bg
    foreground = np.maximum(video - background, 0)
    mae = float(np.mean(np.abs(background - true_bg_video)))

    for threshold in thresholds:
        precision, recall, f1 = metrics(foreground, threshold)
        rows.append([
            alpha, threshold, mae, precision, recall, f1
        ])
        print(
            f"{alpha:5.2f} | {threshold:9.3f} | {mae:14.8f} | "
            f"{precision:9.4f} | {recall:6.4f} | {f1:6.4f}"
        )

csv_path = RESULTS / "hybrid_challenge_metrics.csv"
with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "alpha", "threshold", "background_mae",
        "precision", "recall", "f1_score"
    ])
    writer.writerows(rows)

print(f"\nResults saved to: {csv_path}")
print("alpha=0 means FFT-TDMD; alpha=1 means temporal median.")
