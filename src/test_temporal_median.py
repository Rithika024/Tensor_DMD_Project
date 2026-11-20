
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

video = np.load(ROOT / "outputs" / "synthetic_video.npy")
true_background = np.load(ROOT / "outputs" / "synthetic_background.npy")

# Estimate each pixel's background from its values over time.
estimated_background = np.median(video, axis=2)

# Evaluate against ground truth; ground truth is not used for estimation.
mae = np.mean(np.abs(estimated_background - true_background))
relative_error = (
    np.linalg.norm(estimated_background - true_background)
    / np.linalg.norm(true_background)
)

print("TEMPORAL-MEDIAN BACKGROUND BASELINE")
print("-----------------------------------")
print(f"Estimated background shape: {estimated_background.shape}")
print(f"Mean absolute error:         {mae:.6f}")
print(f"Relative error:              {relative_error:.6f}")

# Save this experimental result separately.
output_path = ROOT / "results" / "temporal_median_background.npy"
np.save(output_path, estimated_background)
print(f"Saved to: {output_path}")
