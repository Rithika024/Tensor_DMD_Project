
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
video_path = ROOT / "outputs" / "synthetic_video.npy"

video = np.load(video_path)

n, m, T = video.shape
print("Video shape:", video.shape)

# Use the same row detrending as tdmd_fft.py
row_mean = np.mean(video, axis=1, keepdims=True)
video_detrended = video - row_mean

X = video_detrended[:, :, :-1]
X = np.transpose(X, (0, 2, 1))
X_fft = np.fft.fft(X, axis=2)

threshold_differences = []
smallest_kept_values = []
condition_numbers = []

for i in range(X_fft.shape[2]):
    Xi = X_fft[:, :, i]

    singular_values = np.linalg.svd(
        Xi, compute_uv=False
    )

    if len(singular_values) == 0:
        continue

    custom_threshold = (
        np.finfo(float).eps
        * max(Xi.shape)
        * singular_values[0]
    )

    numpy_threshold = (
        max(Xi.shape)
        * np.finfo(float).eps
        * singular_values[0]
    )

    threshold_differences.append(
        abs(custom_threshold - numpy_threshold)
    )

    kept = singular_values[singular_values > custom_threshold]

    if len(kept) > 0:
        smallest_kept_values.append(kept[-1])
        condition_numbers.append(
            singular_values[0] / kept[-1]
        )

    print(
        f"Frequency bin {i:2d}: "
        f"largest singular value = {singular_values[0]:.6e}, "
        f"threshold = {custom_threshold:.6e}, "
        f"retained = {len(kept)}/{len(singular_values)}"
    )

print("\n--- Summary ---")
print("Frequency slices checked:", X_fft.shape[2])
print(
    "Maximum custom-vs-NumPy threshold difference:",
    max(threshold_differences)
)
print(
    "Minimum retained singular value:",
    min(smallest_kept_values)
)
print(
    "Maximum retained-subspace condition number:",
    max(condition_numbers)
)
