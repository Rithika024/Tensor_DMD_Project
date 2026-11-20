
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

video = np.load(ROOT / "outputs" / "synthetic_video.npy")
true_background = np.load(ROOT / "outputs" / "synthetic_background.npy")
estimated_background = np.load(
    ROOT / "results" / "temporal_median_background.npy"
)

frame_index = 9  # Frame 10

input_frame = video[:, :, frame_index]
absolute_error = np.abs(estimated_background - true_background)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

images = [
    (input_frame, "Input video — Frame 10"),
    (true_background, "True background"),
    (estimated_background, "Temporal-median estimate"),
    (absolute_error, "Absolute error"),
]

for ax, (image, title) in zip(axes, images):
    im = ax.imshow(image, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.tight_layout()

output_path = ROOT / "results" / "temporal_median_comparison.png"
fig.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Comparison figure saved to: {output_path}")
