
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

video = np.load(ROOT / "outputs" / "synthetic_video.npy")
true_background = np.load(ROOT / "outputs" / "synthetic_background.npy")

n, m, T = video.shape

# Original row-mean detrending estimate.
row_mean = np.mean(video, axis=1)
row_mean_bg = np.median(row_mean, axis=1)

# Ground-truth row means are used only to evaluate the estimate.
true_row_mean = np.mean(true_background, axis=1)

# Known synthetic square coordinates (Python's zero-based indexing).
row0 = 19
square_size = 8

# Diagnostic: compare affected rows with unaffected rows.
affected = np.zeros(n, dtype=bool)
affected[row0:row0 + square_size] = True

print("ROW-BACKGROUND ERROR BY REGION")
print("--------------------------------")
print(
    "Affected rows MAE:",
    f"{np.mean(np.abs(row_mean_bg[affected] - true_row_mean[affected])):.6f}"
)
print(
    "Unaffected rows MAE:",
    f"{np.mean(np.abs(row_mean_bg[~affected] - true_row_mean[~affected])):.6f}"
)

print("\nThis test uses the known square location only for diagnosis.")
print("It does not modify the FFT-TDMD algorithm.")
