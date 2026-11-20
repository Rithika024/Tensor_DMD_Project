
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct


# --------------------------------------------------
# 1. Project paths and parameters
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "outputs" / "synthetic_video.npy"
OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dt = 1.0
tau_re = 2e-3
tau_im = 0.02


# --------------------------------------------------
# 2. DMD for one cosine-frequency slice
# --------------------------------------------------

def fit_dmd_modes(Xi, Yi):
    """Match the SVD and DMD equations in the MATLAB reference."""
    U, s, Vh = np.linalg.svd(Xi, full_matrices=False)

    a = min(Xi.shape[0], Xi.shape[1])
    k2 = min(a, len(s))

    if k2 == 0:
        return None

    U2 = U[:, :k2]
    S2 = np.diag(s[:k2])
    V2 = Vh.conj().T[:, :k2]

    # MATLAB: Atil = U2' * Yi * V2 * pinv(S2)
    A_tilde = U2.conj().T @ Yi @ V2 @ np.linalg.pinv(S2)

    eigenvalues, W = np.linalg.eig(A_tilde)

    # MATLAB: Phi = U2 * W
    modes = U2 @ W

    # MATLAB: b = pinv(Phi) * x1
    amplitudes = np.linalg.pinv(modes) @ Xi[:, 0]

    return eigenvalues, modes, amplitudes



# --------------------------------------------------
# 3. Load video and apply reference-style detrending
# --------------------------------------------------

video = np.load(INPUT_PATH).astype(float)
n, m, T = video.shape

print("Loaded video:", video.shape)

row_mean = np.mean(video, axis=1, keepdims=True)
video_detrended = video - row_mean

row_mean_bg = np.median(row_mean, axis=2, keepdims=True)
background_rows = np.broadcast_to(
    row_mean_bg, (n, m, T)
).copy()

# Consecutive snapshot tensors
X = video_detrended[:, :, :-1]
Y = video_detrended[:, :, 1:]

# Match the reference permutation: (rows, time, width)
X_sw = np.transpose(X, (0, 2, 1))
Y_sw = np.transpose(Y, (0, 2, 1))

# MATLAB reference: mp = 2^nextpow2(2*m)
mp = 2 ** int(np.ceil(np.log2(2 * m)))
print("Original width:", m)
print("Padded cosine width:", mp)

# Zero-pad the third dimension
X_pad = np.pad(
    X_sw, ((0, 0), (0, 0), (0, mp - m))
)
Y_pad = np.pad(
    Y_sw, ((0, 0), (0, 0), (0, mp - m))
)

# MATLAB dct(..., [], 3) corresponds to DCT-II along axis 2.
# norm=None matches MATLAB's default unnormalized DCT convention.
fX = dct(X_pad, type=2, axis=2, norm="ortho")
fY = dct(Y_pad, type=2, axis=2, norm="ortho")


# --------------------------------------------------
# 4. DMD per cosine-frequency slice
# --------------------------------------------------

background_coefficients = np.zeros(
    (n, T, mp), dtype=complex
)

selected_mode_counts = []

for k in range(mp):
    Xi = fX[:, :, k]
    Yi = fY[:, :, k]

    fitted = fit_dmd_modes(Xi, Yi)

    if fitted is None:
        selected_mode_counts.append(0)
        continue

    eigenvalues, modes, amplitudes = fitted

    valid = (
        np.isfinite(eigenvalues)
        & (np.abs(eigenvalues) > 1e-12)
    )

    omega = np.full(eigenvalues.shape, np.nan + 1j * np.nan)
    omega[valid] = np.log(eigenvalues[valid]) / dt

    stationary = (
        valid
        & (np.abs(omega.real) < tau_re)
        & (np.abs(omega.imag) < tau_im)
    )

    selected = np.flatnonzero(stationary)

    # Match the reference's fallback: select the valid
    # eigenvalue with frequency closest to zero.
    if selected.size == 0 and np.any(valid):
        valid_indices = np.flatnonzero(valid)
        closest = valid_indices[
            np.argmin(np.abs(omega[valid_indices]))
        ]
        selected = np.array([closest])

    selected_mode_counts.append(len(selected))

    if selected.size == 0:
        continue

    time_powers = np.arange(T)
    selected_eigenvalues = eigenvalues[selected]
    selected_amplitudes = amplitudes[selected]
    selected_modes = modes[:, selected]

    dynamics = (
        selected_amplitudes[:, None]
        * selected_eigenvalues[:, None] ** time_powers[None, :]
    )

    background_coefficients[:, :, k] = (
        selected_modes @ dynamics
    )


# --------------------------------------------------
# 5. Inverse DCT and crop to original dimensions
# --------------------------------------------------

# idct(type=2) is the inverse of dct(type=2) in SciPy
background_padded = idct(
    background_coefficients,
    type=2,
    axis=2,
    norm="ortho",
)

background_detrended = np.real(
    np.transpose(
        background_padded[:, :, :m], (0, 2, 1)
    )
)

background_estimate = background_detrended + background_rows
foreground_estimate = np.maximum(video - background_estimate, 0)

# Save independent outputs
np.save(
    OUTPUT_DIR / "tdmd_cosine_background.npy",
    background_estimate,
)
np.save(
    OUTPUT_DIR / "tdmd_cosine_foreground.npy",
    foreground_estimate,
)

print("Background shape:", background_estimate.shape)
print("Foreground shape:", foreground_estimate.shape)
print(
    "Selected modes per frequency:",
    f"min={min(selected_mode_counts)}, "
    f"max={max(selected_mode_counts)}, "
    f"mean={np.mean(selected_mode_counts):.2f}",
)


# --------------------------------------------------
# 6. Evaluate one-step reconstruction
# --------------------------------------------------

# DMD fitted on frames 1..T-1 predicts frames 2..T.
# Compare predictions against the matching target frames.
reconstruction_error = np.linalg.norm(
    video[:, :, 1:] - (
        background_detrended[:, :, 1:]
        + (video_detrended[:, :, 1:]
           - background_detrended[:, :, 1:])
    )
)

# Evaluate the background estimate against the known synthetic
# background, when that reference file is available.
reference_background_path = OUTPUT_DIR / "synthetic_background.npy"
if not reference_background_path.exists():
    reference_background_path = ROOT / "outputs" / "synthetic_background.npy"

if reference_background_path.exists():
    true_background = np.load(reference_background_path)
    mae = np.mean(np.abs(background_estimate[:, :, T // 2] - true_background))
    print(f"Mid-video background MAE: {mae:.8f}")

# Plot one representative frame
frame_index = min(T // 2, T - 1)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(video[:, :, frame_index], cmap="gray")
axes[0].set_title("Input")
axes[1].imshow(background_estimate[:, :, frame_index], cmap="gray")
axes[1].set_title("Cosine-TDMD Background")
axes[2].imshow(foreground_estimate[:, :, frame_index], cmap="gray")
axes[2].set_title("Estimated Foreground")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "tdmd_cosine_results.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig)

print("Saved Cosine-TDMD results.")
