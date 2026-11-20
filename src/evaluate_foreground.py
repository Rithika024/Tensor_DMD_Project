
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load the video and foreground estimates.
video = np.load(OUTPUT_DIR / "synthetic_video.npy")
n, m, T = video.shape

median_foreground = np.load(
    RESULTS_DIR / "median_foreground.npy"
)
tdmd_foreground = np.load(
    RESULTS_DIR / "tdmd_fft_foreground.npy"
)

# Re-create the exact moving-square ground truth.
square_size = 8
row0, col0 = 19, 0
col_speed = 1

ground_truth = np.zeros((n, m, T), dtype=bool)

for t in range(T):
    c1 = max(0, min(m - square_size, col0 + t * col_speed))
    c2 = c1 + square_size
    r1 = max(0, min(n - square_size, row0))
    r2 = r1 + square_size

    ground_truth[r1:r2, c1:c2, t] = True

np.save(
    RESULTS_DIR / "ground_truth_foreground_mask.npy",
    ground_truth
)

# Use one common threshold for this initial comparison.
threshold = 0.10


def evaluate_foreground(name, estimated_foreground):
    predicted = estimated_foreground > threshold

    tp = np.count_nonzero(predicted & ground_truth)
    fp = np.count_nonzero(predicted & ~ground_truth)
    fn = np.count_nonzero(~predicted & ground_truth)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    print(f"\n{name}")
    print(f"  True positives:  {tp}")
    print(f"  False positives: {fp}")
    print(f"  False negatives: {fn}")
    print(f"  Precision:       {precision:.6f}")
    print(f"  Recall:          {recall:.6f}")
    print(f"  F1-score:        {f1:.6f}")


print("Ground-truth foreground pixels:", np.count_nonzero(ground_truth))
print("Threshold:", threshold)

evaluate_foreground("Temporal Median", median_foreground)
evaluate_foreground("FFT-TDMD Baseline", tdmd_foreground)

print("\nGround-truth mask saved to results.")
