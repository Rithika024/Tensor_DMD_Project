
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Use the existing background to isolate the effect of
# changing foreground contrast and motion.
background = np.load(OUTPUT / "synthetic_background.npy")
n, m = background.shape

T = 30
square_size = 8
row0 = 19
start_col = 0
col_speed = 2
foreground_value = 0.08

video = np.zeros((n, m, T), dtype=float)
mask = np.zeros((n, m, T), dtype=bool)

for t in range(T):
    frame = background.copy()

    c1 = min(m - square_size, start_col + t * col_speed)
    c2 = c1 + square_size
    r1 = row0
    r2 = r1 + square_size

    # Add a lower-contrast moving square.
    frame[r1:r2, c1:c2] += foreground_value
    video[:, :, t] = np.clip(frame, 0, 1)
    mask[r1:r2, c1:c2, t] = True

np.save(OUTPUT / "challenge_video.npy", video)
np.save(OUTPUT / "challenge_foreground_mask.npy", mask)

print("Challenge video saved.")
print("Video shape:", video.shape)
print("Ground-truth mask shape:", mask.shape)
print("Foreground intensity increment:", foreground_value)
print("Foreground pixels per frame:", np.count_nonzero(mask[:, :, 0]))
