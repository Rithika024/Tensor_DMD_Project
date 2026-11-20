
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

ground_truth = np.load(
    RESULTS / "ground_truth_foreground_mask.npy"
)

methods = {
    "Temporal Median": np.load(
        RESULTS / "median_foreground.npy"
    ),
    "FFT-TDMD": np.load(
        RESULTS / "tdmd_fft_foreground.npy"
    ),
}

thresholds = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]

for name, foreground in methods.items():
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
            f"{threshold:9.2f} | {precision:9.4f} | "
            f"{recall:6.4f} | {f1:8.4f}"
        )
