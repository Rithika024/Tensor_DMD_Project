
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Load the same input and ground truth for every evaluation.
video = np.load(ROOT / "outputs" / "synthetic_video.npy")
true_background = np.load(ROOT / "outputs" / "synthetic_background.npy")

standard_dmd = np.load(RESULTS / "standard_dmd_reconstruction.npy")
tdmd_reconstruction = np.load(RESULTS / "tdmd_fft_reconstruction.npy")
tdmd_background = np.load(RESULTS / "tdmd_fft_background.npy")
median_background = np.load(RESULTS / "temporal_median_background.npy")

def relative_error(estimate, reference):
    denominator = np.linalg.norm(reference)
    if denominator == 0:
        return 0.0 if np.linalg.norm(estimate) == 0 else np.inf
    return np.linalg.norm(estimate - reference) / denominator

# Reconstruction metrics: how well each method reconstructs the video.
reconstruction_methods = {
    "Standard DMD": standard_dmd,
    "FFT-TDMD": tdmd_reconstruction,
}

reconstruction_errors = {}
for name, estimate in reconstruction_methods.items():
    if estimate.shape != video.shape:
        raise ValueError(
            f"{name} has shape {estimate.shape}, expected {video.shape}"
        )

    reconstruction_errors[name] = [
        100 * relative_error(estimate[:, :, t], video[:, :, t])
        for t in range(video.shape[2])
    ]

# Background metrics: compare background estimates with the true background.
background_methods = {
    "FFT-TDMD": tdmd_background,
    "Temporal median": median_background,
}

background_mae = {}
background_relative_errors = {}

for name, estimate in background_methods.items():
    if estimate.shape == video.shape:
        # A video-sized estimate: evaluate every frame against the static
        # ground-truth background.
        reference = np.broadcast_to(true_background[:, :, None], video.shape)
    elif estimate.shape == true_background.shape:
        reference = true_background
    else:
        raise ValueError(
            f"{name} has unexpected shape {estimate.shape}"
        )

    background_mae[name] = float(np.mean(np.abs(estimate - reference)))
    background_relative_errors[name] = (
        100 * relative_error(estimate, reference)
    )

# Print results.
print("\nVIDEO RECONSTRUCTION ERROR")
print("--------------------------")
print(f"{'Method':<18} {'Frames 1-19 mean (%)':>22} {'Frame 20 (%)':>15}")
for name, errors in reconstruction_errors.items():
    print(
        f"{name:<18} "
        f"{np.mean(errors[:-1]):>22.6f} "
        f"{errors[-1]:>15.6f}"
    )

print("\nBACKGROUND ESTIMATION ERROR")
print("---------------------------")
print(f"{'Method':<18} {'MAE':>12} {'Relative error (%)':>21}")
for name in background_methods:
    print(
        f"{name:<18} "
        f"{background_mae[name]:>12.6f} "
        f"{background_relative_errors[name]:>21.6f}"
    )

# Save a CSV with separate, clearly labelled metrics.
csv_path = RESULTS / "evaluation_metrics.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Evaluation type", "Method", "Metric", "Value"])

    for name, errors in reconstruction_errors.items():
        writer.writerow([
            "Video reconstruction", name,
            "Mean relative error frames 1-19 (%)",
            f"{np.mean(errors[:-1]):.8f}"
        ])
        writer.writerow([
            "Video reconstruction", name,
            "Relative error frame 20 (%)",
            f"{errors[-1]:.8f}"
        ])

    for name in background_methods:
        writer.writerow([
            "Background estimation", name, "MAE",
            f"{background_mae[name]:.10f}"
        ])
        writer.writerow([
            "Background estimation", name, "Relative error (%)",
            f"{background_relative_errors[name]:.8f}"
        ])

# Plot reconstruction errors across frames.
plt.figure(figsize=(9, 5))
frames = np.arange(1, video.shape[2] + 1)

for name, errors in reconstruction_errors.items():
    plt.plot(frames, errors, marker="o", label=name)

plt.xlabel("Frame number")
plt.ylabel("Relative reconstruction error (%)")
plt.title("Video Reconstruction Error by Frame")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plot_path = RESULTS / "evaluation_reconstruction_errors.png"
plt.savefig(plot_path, dpi=200)
plt.close()

print(f"\nCSV saved: {csv_path}")
print(f"Plot saved: {plot_path}")
