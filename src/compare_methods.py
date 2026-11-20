
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
video = np.load(ROOT / "outputs" / "synthetic_video.npy")
standard = np.load(ROOT / "results" / "standard_dmd_reconstruction.npy")
fft_tdmd = np.load(ROOT / "results" / "tdmd_fft_reconstruction.npy")

def frame_errors(reference, prediction):
    errors = []
    for i in range(reference.shape[2]):
        actual = reference[:, :, i]
        estimated = prediction[:, :, i]
        denominator = max(np.linalg.norm(actual), 1e-15)
        errors.append(np.linalg.norm(actual - estimated) / denominator)
    return np.asarray(errors)

dmd_errors = frame_errors(video, standard)
tdmd_errors = frame_errors(video, fft_tdmd)

print("\nRelative reconstruction errors")
print("-" * 57)
print(f"{'Metric':30s} {'Standard DMD':>13s} {'FFT-TDMD':>11s}")
print("-" * 57)

for label, selection in [
    ("Frames 1-19 mean (%)", slice(0, 19)),
    ("Frame 20 (%)", slice(19, 20)),
    ("All 20 frames mean (%)", slice(None)),
]:
    dmd_value = 100 * np.mean(dmd_errors[selection])
    tdmd_value = 100 * np.mean(tdmd_errors[selection])
    print(f"{label:30s} {dmd_value:13.6f} {tdmd_value:11.6f}")

print("-" * 57)

np.savetxt(
    ROOT / "results" / "method_comparison.csv",
    np.column_stack([
        np.arange(1, video.shape[2] + 1),
        100 * dmd_errors,
        100 * tdmd_errors,
    ]),
    delimiter=",",
    header="frame,standard_dmd_error_percent,fft_tdmd_error_percent",
    comments="",
)

plt.figure(figsize=(10, 5))
plt.plot(
    np.arange(1, video.shape[2] + 1),
    100 * dmd_errors,
    marker="o",
    label="Standard DMD",
)
plt.plot(
    np.arange(1, video.shape[2] + 1),
    100 * tdmd_errors,
    marker="s",
    label="FFT-TDMD",
)
plt.axvline(19, linestyle="--", alpha=0.6, label="Last training snapshot")
plt.xlabel("Frame number")
plt.ylabel("Relative reconstruction error (%)")
plt.title("Standard DMD vs FFT-TDMD: All 20 Frames")
plt.xticks(np.arange(1, video.shape[2] + 1))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    ROOT / "results" / "method_comparison.png",
    dpi=200,
    bbox_inches="tight",
)
plt.show()

print("\nUpdated comparison files:")
print(ROOT / "results" / "method_comparison.csv")
print(ROOT / "results" / "method_comparison.png")
