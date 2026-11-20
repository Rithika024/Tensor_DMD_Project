from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

background = np.load(
    OUTPUT_DIR / "synthetic_background.npy"
).astype(float)

n, m = background.shape
T = 30
square_size = 8
row0 = 19
start_col = 0
col_speed = 2
foreground_value = 0.04
noise_std = 0.01

video = np.zeros((n, m, T), dtype=float)
mask = np.zeros((n, m, T), dtype=bool)

for t in range(T):
    frame = background.copy()

    c1 = min(m - square_size, start_col + t * col_speed)
    c2 = c1 + square_size
    r1 = row0
    r2 = r1 + square_size

    frame[r1:r2, c1:c2] += foreground_value
    mask[r1:r2, c1:c2, t] = True

    noise = rng.normal(0, noise_std, size=(n, m))
    video[:, :, t] = np.clip(frame + noise, 0, 1)

np.save(OUTPUT_DIR / "noisy_challenge_video.npy", video)
np.save(OUTPUT_DIR / "noisy_challenge_foreground_mask.npy", mask)

print("Noisy challenge video saved.")
print("Video shape:", video.shape)
print("Foreground intensity increment:", foreground_value)
print("Background noise standard deviation:", noise_std)
print("Foreground pixels per frame:", int(mask[:, :, 0].sum()))
