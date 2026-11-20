
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Set paths and load the synthetic video
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

video_path = OUTPUT_DIR / "synthetic_video.npy"

if not video_path.exists():
    raise FileNotFoundError(
        f"Cannot find {video_path}. Run src/generate_video.py first."
    )

video = np.load(video_path)
n, m, T = video.shape

print("Video loaded:", video.shape)


# --------------------------------------------------
# 2. Convert frames into snapshot vectors
# --------------------------------------------------
# Each column of X contains one flattened video frame.
X = video[:, :, :-1].reshape(n * m, T - 1)
Y = video[:, :, 1:].reshape(n * m, T - 1)

print("X dimensions:", X.shape)
print("Y dimensions:", Y.shape)


# --------------------------------------------------
# 3. Compute the reduced-order DMD model
# --------------------------------------------------
# X = U Sigma V^T
U, singular_values, Vh = np.linalg.svd(X, full_matrices=False)

# Numerical rank threshold
tolerance = (
    np.finfo(float).eps
    * max(X.shape)
    * singular_values[0]
)
rank = int(np.sum(singular_values > tolerance))

# Use a truncated SVD
U_r = U[:, :rank]
s_r = singular_values[:rank]
V_r = Vh[:rank, :].T

print("Numerical rank:", rank)

# Reduced dynamics matrix:
# A_tilde = U_r^T Y V_r Sigma_r^(-1)
A_tilde = (
    U_r.T @ Y @ V_r
) @ np.diag(1.0 / s_r)

# Eigenvalues and eigenvectors
eigenvalues, W = np.linalg.eig(A_tilde)

# DMD modes
Phi = U_r @ W

# Initial modal amplitudes
x0 = video[:, :, 0].reshape(-1)
b = np.linalg.lstsq(Phi, x0, rcond=None)[0]


# --------------------------------------------------
# 4. Reconstruct the video using DMD
# --------------------------------------------------
# Reconstruct all 20 frames using powers of eigenvalues.
reconstructed_frames = []

for t in range(T):
    time_dynamics = (eigenvalues ** t) * b
    x_reconstructed = Phi @ time_dynamics

    frame = x_reconstructed.real.reshape(n, m)
    reconstructed_frames.append(frame)

reconstruction = np.stack(reconstructed_frames, axis=2)

np.save(
    RESULTS_DIR / "standard_dmd_reconstruction.npy",
    reconstruction
)


# --------------------------------------------------
# 5. Calculate reconstruction error
# --------------------------------------------------
difference = video - reconstruction

relative_error = (
    np.linalg.norm(difference.ravel())
    / max(np.linalg.norm(video.ravel()), np.finfo(float).eps)
)

frame_errors = []

for t in range(T):
    numerator = np.linalg.norm(
        video[:, :, t] - reconstruction[:, :, t]
    )
    denominator = max(
        np.linalg.norm(video[:, :, t]),
        np.finfo(float).eps
    )
    frame_errors.append(numerator / denominator)

print(f"Relative reconstruction error: {relative_error:.6f}")
print(f"Mean per-frame relative error: {np.mean(frame_errors):.6f}")


# --------------------------------------------------
# 6. Plot original and reconstructed frames
# --------------------------------------------------
selected_frames = [0, 9, 19]

fig, axes = plt.subplots(2, len(selected_frames), figsize=(10, 6))

for col, t in enumerate(selected_frames):
    axes[0, col].imshow(
        video[:, :, t], cmap="gray", vmin=0, vmax=1
    )
    axes[0, col].set_title(f"Original: frame {t + 1}")
    axes[0, col].axis("off")

    axes[1, col].imshow(
        reconstruction[:, :, t], cmap="gray", vmin=0, vmax=1
    )
    axes[1, col].set_title(f"DMD: frame {t + 1}")
    axes[1, col].axis("off")

fig.suptitle("Standard DMD Reconstruction")
fig.tight_layout()

fig.savefig(
    RESULTS_DIR / "standard_dmd_comparison.png",
    dpi=200,
    bbox_inches="tight"
)
plt.show()


# --------------------------------------------------
# 7. Plot frame-by-frame reconstruction errors
# --------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(
    np.arange(1, T + 1),
    frame_errors,
    marker="o"
)
plt.xlabel("Frame number")
plt.ylabel("Relative reconstruction error")
plt.title("Standard DMD: Per-frame Reconstruction Error")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "standard_dmd_errors.png",
    dpi=200,
    bbox_inches="tight"
)
plt.show()

print("\nStandard DMD completed.")
print("Saved reconstruction and evaluation plots in:", RESULTS_DIR)
