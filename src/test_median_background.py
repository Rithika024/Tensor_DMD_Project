
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

video = np.load(ROOT / "outputs" / "synthetic_video.npy")
true_background = np.load(
    ROOT / "outputs" / "synthetic_background.npy"
)

# Estimate the static background independently at each pixel.
estimated_background = np.median(video, axis=2)

# Repeat the background estimate across all frames.
estimated_background_video = np.repeat(
    estimated_background[:, :, None],
    video.shape[2],
    axis=2,
)

# Extract positive foreground residuals.
estimated_foreground = np.maximum(
    video - estimated_background_video, 0
)

# Evaluate against the known synthetic background.
background_mae = np.mean(
    np.abs(estimated_background - true_background)
)

print("Video shape:", video.shape)
print("Estimated background shape:", estimated_background.shape)
print(f"Background MAE: {background_mae:.10f}")
print(
    "Estimated background finite:",
    np.isfinite(estimated_background).all(),
)
print(
    "Estimated foreground finite:",
    np.isfinite(estimated_foreground).all(),
)

# Save separately; do not overwrite FFT-TDMD results.
results_dir = ROOT / "results"
results_dir.mkdir(parents=True, exist_ok=True)

np.save(
    results_dir / "median_background.npy",
    estimated_background_video,
)
np.save(
    results_dir / "median_foreground.npy",
    estimated_foreground,
)

print("Median-background results saved.")
