
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"
RESULTS = ROOT / "results"

video = np.load(OUTPUT / "challenge_video.npy")
ground_truth = np.load(
    OUTPUT / "challenge_foreground_mask.npy"
)

# Estimate background and foreground using temporal median.
median_background = np.median(video, axis=2)
median_background_video = np.repeat(
    median_background[:, :, None],
    video.shape[2],
    axis=2,
)
median_foreground = np.maximum(
    video - median_background_video, 0
)

# Evaluate against the known moving-object mask.
thresholds = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08]


def evaluate(name, foreground):
    print(f"\n{name}")
    print("Threshold | Precision | Recall | F1-score")

    for threshold in thresholds:
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


# Evaluate the temporal-median method.
evaluate("Temporal Median", median_foreground)

# Background quality against the known static background.
true_background = np.load(
    OUTPUT / "synthetic_background.npy"
)
median_mae = np.mean(
    np.abs(median_background - true_background)
)
print(f"\nTemporal-median background MAE: {median_mae:.8f}")

# Save this method's outputs separately.
RESULTS.mkdir(parents=True, exist_ok=True)
np.save(RESULTS / "challenge_median_background.npy",
        median_background_video)
np.save(RESULTS / "challenge_median_foreground.npy",
        median_foreground)

print("\nChallenge evaluation completed.")
