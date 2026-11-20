
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from PIL import Image

# 1. Set paths and experiment parameters
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(1)

n, m, T = 60, 60, 20
square_size = 8
row0, col0 = 19, 0
col_speed = 1
foreground_value = 0.95

# 2. Generate a static, noisy background
background_base = 0.25

illumination = gaussian_filter(np.random.randn(n, m), sigma=7)
illumination = (
    illumination - illumination.mean()
) / (illumination.std() + np.finfo(float).eps)

xx, yy = np.meshgrid(np.linspace(-1, 1, m), np.linspace(-1, 1, n))
radius = np.sqrt(xx**2 + yy**2)
vignette = 1 - 0.25 * radius**2
gradient = 0.10 * xx + 0.05 * yy

texture = gaussian_filter(np.random.randn(n, m), sigma=1.5)
texture = (
    texture - texture.mean()
) / (texture.std() + np.finfo(float).eps)

background = (
    background_base
    + 0.08 * illumination
    + 0.04 * texture
    + 0.08 * (vignette - vignette.mean())
    + 0.05 * gradient
)

background += 0.01 * np.random.randn(n, m)
background = np.clip(background, 0, 1)

# 3. Generate the 20-frame video
video = np.zeros((n, m, T), dtype=np.float64)

for t in range(T):
    foreground = np.zeros((n, m))

    c1 = max(0, min(m - square_size, col0 + t * col_speed))
    c2 = c1 + square_size

    r1 = max(0, min(n - square_size, row0))
    r2 = r1 + square_size

    foreground[r1:r2, c1:c2] = foreground_value
    video[:, :, t] = background + foreground

video = np.clip(video, 0, 1)

# 4. Save the generated data
np.save(OUTPUT_DIR / "synthetic_video.npy", video)
np.save(OUTPUT_DIR / "synthetic_background.npy", background)

print("Synthetic video generated successfully.")
print(f"Video dimensions: {video.shape}")
print(f"Number of frames: {T}")
print(f"Frame dimensions: {n} x {m}")
print(f"Minimum pixel value: {video.min():.4f}")
print(f"Maximum pixel value: {video.max():.4f}")

# 5. Display and save representative frames
selected_frames = [0, 1, 9, 19]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))

for ax, frame_index in zip(axes, selected_frames):
    ax.imshow(video[:, :, frame_index], cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Frame {frame_index + 1}")
    ax.axis("off")

fig.suptitle("Synthetic Video: Moving Square on a Noisy Background")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "synthetic_video_frames.png", dpi=200,
            bbox_inches="tight")
plt.show()

# 6. Save an animated GIF
gif_frames = [
    Image.fromarray(np.uint8(video[:, :, t] * 255)).convert("L")
    for t in range(T)
]

gif_path = OUTPUT_DIR / "synthetic_video.gif"
gif_frames[0].save(
    gif_path,
    save_all=True,
    append_images=gif_frames[1:],
    duration=150,
    loop=0
)

print(f"Frame visualization saved to: {OUTPUT_DIR / 'synthetic_video_frames.png'}")
print(f"Animated video saved to: {gif_path}")
