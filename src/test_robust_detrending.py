
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

video = np.load(ROOT / "outputs" / "synthetic_video.npy")
true_background = np.load(ROOT / "outputs" / "synthetic_background.npy")

# Original method: mean across columns, then median across time.
original_row_mean = np.mean(video, axis=1)
original_estimate = np.median(original_row_mean, axis=1)

# Experimental method: use the 20th percentile across columns
# in each frame, then take the median across time.
# The bright square occupies only part of each affected row.
robust_row_estimate = np.percentile(video, 20, axis=1)
robust_estimate = np.median(robust_row_estimate, axis=1)

# Ground truth is used ONLY to evaluate the experiment.
true_estimate = np.mean(true_background, axis=1)

original_mae = np.mean(np.abs(original_estimate - true_estimate))
robust_mae = np.mean(np.abs(robust_estimate - true_estimate))

print("ROW-BACKGROUND ESTIMATOR COMPARISON")
print("-----------------------------------")
print(f"Original method MAE: {original_mae:.6f}")
print(f"Robust method MAE:   {robust_mae:.6f}")

print("\nRows 18–29")
print("Row | True mean | Original | Robust")
for i in range(17, 29):
    print(
        f"{i+1:3d} | "
        f"{true_estimate[i]:9.6f} | "
        f"{original_estimate[i]:8.6f} | "
        f"{robust_estimate[i]:8.6f}"
    )
