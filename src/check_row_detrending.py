
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

video = np.load(ROOT / "outputs" / "synthetic_video.npy")
true_background = np.load(ROOT / "outputs" / "synthetic_background.npy")

# Original row means used by the current implementation.
row_mean = np.mean(video, axis=1, keepdims=True)
row_mean_bg = np.median(row_mean, axis=2, keepdims=True)

# Ground-truth row means: diagnostic only.
true_row_mean = np.mean(true_background, axis=1, keepdims=True)
true_row_mean = true_row_mean[:, :, None]

print("Original restored row mean vs ground truth")
print("Mean absolute difference:",
      np.mean(np.abs(row_mean_bg - true_row_mean)))

print("\nRows 18–29")
for i in range(17, 29):
    print(
        f"Row {i+1:2d}: "
        f"estimated={row_mean_bg[i, 0, 0]:.6f}, "
        f"true={true_row_mean[i, 0, 0]:.6f}, "
        f"difference={row_mean_bg[i, 0, 0] - true_row_mean[i, 0, 0]:.6f}"
    )
