from pathlib import Path
import sys
import time
from io import BytesIO

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "outputs" / "benchmark_v1"
OUTPUT_DIR = ROOT / "results" / "live_demo"

sys.path.insert(0, str(ROOT / "src" / "experiments"))

from evaluate_foreground_background_variants import (
    estimate_temporal_median,
    estimate_tensor_background,
    foreground_metrics,
)

VIDEO_NAME = "slow_noise_0.02"

video = np.load(DATA_DIR / f"{VIDEO_NAME}_video.npy")
true_background = np.load(DATA_DIR / f"{VIDEO_NAME}_background.npy")
true_mask = np.load(DATA_DIR / f"{VIDEO_NAME}_mask.npy")

print("Loaded video:", video.shape)
print("Loaded true background:", true_background.shape)
print("Loaded foreground mask:", true_mask.shape)

methods = {
    "Temporal Median": lambda x: estimate_temporal_median(x),
    "FFT-TDMD": lambda x: estimate_tensor_background(x, "fft"),
    "Haar-TDMD": lambda x: estimate_tensor_background(x, "haar"),
    "Cosine-TDMD": lambda x: estimate_tensor_background(x, "cosine"),
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

estimates = {}
scores = {}

for name, estimator in methods.items():
    print(f"\nEstimating background using {name}...")
    start = time.perf_counter()
    estimated = estimator(video)
    elapsed = time.perf_counter() - start

    if estimated.shape != video.shape:
        raise ValueError(f"{name}: unexpected output shape {estimated.shape}")

    predicted_mask = np.maximum(video - estimated, 0.0) > 0.10
    metrics = foreground_metrics(video, estimated, true_mask)

    estimates[name] = (estimated, predicted_mask)
    scores[name] = metrics["F1"]

    print(
        f"F1={metrics['F1']:.4f} | "
        f"Precision={metrics['precision']:.4f} | "
        f"Recall={metrics['recall']:.4f} | "
        f"Runtime={elapsed:.2f}s"
    )

gif_path = OUTPUT_DIR / "foreground_background_live_demo.gif"
representative_path = OUTPUT_DIR / "foreground_background_comparison.png"

frames = []

print("\nCreating comparison animation...")

for t in range(video.shape[2]):
    fig, axes = plt.subplots(5, 3, figsize=(11, 15))

    rows = [
        ("Ground Truth", video[:, :, t], true_background,
         true_mask[:, :, t]),
        ("Temporal Median", video[:, :, t],
         estimates["Temporal Median"][0][:, :, t],
         estimates["Temporal Median"][1][:, :, t]),
        ("FFT-TDMD", video[:, :, t],
         estimates["FFT-TDMD"][0][:, :, t],
         estimates["FFT-TDMD"][1][:, :, t]),
        ("Haar-TDMD", video[:, :, t],
         estimates["Haar-TDMD"][0][:, :, t],
         estimates["Haar-TDMD"][1][:, :, t]),
        ("Cosine-TDMD", video[:, :, t],
         estimates["Cosine-TDMD"][0][:, :, t],
         estimates["Cosine-TDMD"][1][:, :, t]),
    ]

    for row_index, (name, input_frame, background, mask) in enumerate(rows):
        axes[row_index, 0].imshow(input_frame, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 1].imshow(background, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 2].imshow(mask, cmap="gray", vmin=0, vmax=1)

        axes[row_index, 0].set_ylabel(name, fontsize=10)

        if row_index == 0:
            axes[row_index, 0].set_title("Input video")
            axes[row_index, 1].set_title("Background")
            axes[row_index, 2].set_title("Foreground mask")

        for ax in axes[row_index]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"Foreground / Background Comparison | "
        f"{VIDEO_NAME} | Frame {t + 1}/{video.shape[2]}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if t == 29:
        fig.savefig(representative_path, dpi=150, bbox_inches="tight")

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    frames.append(frame)
    plt.close(fig)

imageio.mimsave(gif_path, frames, fps=6, loop=0)

print("\n" + "=" * 60)
print("LIVE DEMO COMPLETE")
print("=" * 60)

print("\nForeground F1 scores:")
for name, score in scores.items():
    print(f"{name:20s}: {score:.4f}")

print("\nSaved files:")
print("GIF:", gif_path)
print("Comparison image:", representative_path)
