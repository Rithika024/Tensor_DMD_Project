from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

video = np.load(OUTPUT_DIR / "noisy_challenge_video.npy")
ground_truth = np.load(
    OUTPUT_DIR / "noisy_challenge_foreground_mask.npy"
).astype(bool)

# Estimate the background using the temporal median.
background = np.median(video, axis=2)
background_video = np.repeat(
    background[:, :, None], video.shape[2], axis=2
)
foreground = np.maximum(video - background_video, 0)

np.save(
    RESULTS_DIR / "noisy_median_background.npy", background
)
np.save(
    RESULTS_DIR / "noisy_median_foreground.npy", foreground
)

# Compare estimated background with the known clean background.
true_background = np.load(
    OUTPUT_DIR / "synthetic_background.npy"
)
background_mae = np.mean(np.abs(background - true_background))

print("Noisy Challenge: Temporal Median")
print("Threshold | Precision | Recall | F1-score")

for threshold in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04]:
    predicted = foreground > threshold

    tp = np.count_nonzero(predicted & ground_truth)
    fp = np.count_nonzero(predicted & ~ground_truth)
    fn = np.count_nonzero(~predicted & ground_truth)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    print(
        f"{threshold:9.3f} | {precision:9.4f} | "
        f"{recall:6.4f} | {f1:8.4f}"
    )

print(f"\nBackground MAE: {background_mae:.8f}")
print("Saved noisy temporal-median results.")
