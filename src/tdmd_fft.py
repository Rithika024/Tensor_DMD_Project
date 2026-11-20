
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Paths and parameters
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

video = np.load(OUTPUT_DIR / "synthetic_video.npy")
n, m, T = video.shape

dt = 1.0
tau_re = 2e-3
tau_im = 0.02

print("Loaded video:", video.shape)
print("Method: FFT-TDMD")


# --------------------------------------------------
# 2. Detrend each frame by subtracting its row means
# --------------------------------------------------
row_mean = np.mean(video, axis=1, keepdims=True)
video_detrended = video - row_mean

# Estimate the persistent row background and retain it
# for adding back after tensor reconstruction.
row_mean_bg = np.median(row_mean, axis=2, keepdims=True)
background_rows = np.broadcast_to(
    row_mean_bg, (n, m, T)
).copy()

# Consecutive snapshots
X = video_detrended[:, :, :-1]
Y = video_detrended[:, :, 1:]
r = T - 1

# Change the dimensions to (rows, time snapshots, columns).
# The third dimension is transformed using the FFT.
X_sw = np.transpose(X, (0, 2, 1))
Y_sw = np.transpose(Y, (0, 2, 1))

fX = np.fft.fft(X_sw, axis=2)
fY = np.fft.fft(Y_sw, axis=2)


# --------------------------------------------------
# 3. Fit a reduced DMD model for each frequency slice
# --------------------------------------------------
def fit_frequency_slice(Xi, Yi):
    """Fit DMD to one frequency slice."""
    U, singular_values, Vh = np.linalg.svd(
        Xi, full_matrices=False
    )

    # The MATLAB reference uses all economical SVD directions.
    rank = min(Xi.shape)

    U_r = U[:, :rank]
    s_r = singular_values[:rank]
    V_r = Vh[:rank, :].conj().T

    # Reduced transition operator
    S_r = np.diag(s_r)
    S_r_pinv = np.linalg.pinv(S_r)

    A_tilde = (
         U_r.conj().T @ Yi @ V_r
    ) @ S_r_pinv

    eigenvalues, W = np.linalg.eig(A_tilde)
    modes = U_r @ W

    # Initial amplitude for this frequency slice
    initial_snapshot = Xi[:, 0]
    amplitudes = np.linalg.pinv(modes) @ initial_snapshot

    return modes, eigenvalues, amplitudes


# --------------------------------------------------
# 4. Reconstruct all modes and persistent background
# --------------------------------------------------
# Arrays use dimensions (rows, time, frequency).
Z_reconstruction = np.zeros(
    (n, T, m), dtype=np.complex128
)
Z_background = np.zeros(
    (n, T, m), dtype=np.complex128
)

background_mode_counts = []

for i in range(m):
    Xi = fX[:, :, i]
    Yi = fY[:, :, i]

    modes, eigenvalues, amplitudes = fit_frequency_slice(
        Xi, Yi
    )

    # Full reconstruction: use every computed mode.
    for t in range(T):
        Z_reconstruction[:, t, i] = modes @ (
            (eigenvalues ** t) * amplitudes
        )

    # Identify approximately stationary modes.
    # omega = log(lambda) / dt
    
    eigenvalues = np.asarray(eigenvalues, dtype=np.complex128)

    omega = np.full(
        eigenvalues.shape,
        np.nan + 1j * np.nan,
        dtype=np.complex128
    )

    valid = np.abs(eigenvalues) > 1e-14

    omega[valid] = np.log(eigenvalues[valid]) / dt


    
    valid_omega = (
        np.isfinite(omega.real)
        & np.isfinite(omega.imag)
    )

    is_background = (
        valid_omega
        & (np.abs(omega.real) < tau_re)
        & (np.abs(omega.imag) < tau_im)
    )

    # If no stationary mode qualifies, select the
    # valid mode whose frequency is closest to zero.
    if not np.any(is_background):
        valid_indices = np.where(valid_omega)[0]

        if len(valid_indices) > 0:
            closest_index = valid_indices[
               np.argmin(np.abs(omega[valid_indices]))
            ]
            is_background[closest_index] = True


    background_mode_counts.append(
        int(np.sum(is_background))
    )

    selected_modes = modes[:, is_background]
    selected_eigenvalues = eigenvalues[is_background]
    selected_amplitudes = amplitudes[is_background]

    for t in range(T):
        Z_background[:, t, i] = selected_modes @ (
            (selected_eigenvalues ** t) * selected_amplitudes
        )


# --------------------------------------------------
# 5. Inverse FFT and restore the row background
# --------------------------------------------------
# Inverse FFT produces dimensions (rows, time, columns).
reconstruction_detrended = np.fft.ifft(
    Z_reconstruction, axis=2
).real

background_detrended = np.fft.ifft(
    Z_background, axis=2
).real

# Convert back to (rows, columns, time).
reconstruction_detrended = np.transpose(
    reconstruction_detrended, (0, 2, 1)
)
background_detrended = np.transpose(
    background_detrended, (0, 2, 1)
)

reconstruction = (
    reconstruction_detrended + background_rows
)

estimated_background = (
    background_detrended + background_rows
)

estimated_foreground = np.maximum(
    video - estimated_background, 0
)


# --------------------------------------------------
# 6. Evaluate full reconstruction on frames 1 to T-1
# --------------------------------------------------
# This matches the frame range used in the reference
# MATLAB experiment.
target = video[:, :, :T - 1]
predicted = reconstruction[:, :, :T - 1]

overall_relative_error = (
    np.linalg.norm((target - predicted).ravel())
    / max(np.linalg.norm(target.ravel()), 1e-15)
)

per_frame_errors = []

for t in range(T - 1):
    numerator = np.linalg.norm(
        target[:, :, t] - predicted[:, :, t]
    )
    denominator = max(
        np.linalg.norm(target[:, :, t]), 1e-15
    )
    per_frame_errors.append(numerator / denominator)

print("\nFFT-TDMD completed.")
print("Frequency slices processed:", m)
print(
    "Background modes selected per slice:",
    background_mode_counts
)
print(f"Overall relative reconstruction error: "
      f"{overall_relative_error:.6f}")
print(f"Mean per-frame relative error: "
      f"{np.mean(per_frame_errors):.6f}")


# --------------------------------------------------
# 7. Save numerical results
# --------------------------------------------------
np.save(
    RESULTS_DIR / "tdmd_fft_reconstruction.npy",
    reconstruction
)
np.save(
    RESULTS_DIR / "tdmd_fft_background.npy",
    estimated_background
)
np.save(
    RESULTS_DIR / "tdmd_fft_foreground.npy",
    estimated_foreground
)
np.save(
    RESULTS_DIR / "tdmd_fft_frame_errors.npy",
    np.asarray(per_frame_errors)
)


# --------------------------------------------------
# 8. Visualize foreground and background separation
# --------------------------------------------------
frame_index = min(9, T - 2)

fig, axes = plt.subplots(1, 3, figsize=(11, 4))

axes[0].imshow(
    video[:, :, frame_index], cmap="gray",
    vmin=0, vmax=1
)
axes[0].set_title(f"Original frame {frame_index + 1}")

axes[1].imshow(
    estimated_background[:, :, frame_index],
    cmap="gray", vmin=0, vmax=1
)
axes[1].set_title("Estimated background")

axes[2].imshow(
    estimated_foreground[:, :, frame_index],
    cmap="gray", vmin=0, vmax=1
)
axes[2].set_title("Estimated foreground")

for ax in axes:
    ax.axis("off")

fig.tight_layout()
fig.savefig(
    RESULTS_DIR / "tdmd_fft_foreground_background.png",
    dpi=200,
    bbox_inches="tight"
)
plt.show()


# --------------------------------------------------
# 9. Plot per-frame reconstruction error
# --------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(
    np.arange(1, T),
    per_frame_errors,
    marker="o"
)
plt.xlabel("Frame number")
plt.ylabel("Relative reconstruction error")
plt.title("FFT-TDMD Reconstruction Error")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "tdmd_fft_errors.png",
    dpi=200,
    bbox_inches="tight"
)
plt.show()

print("\nResults saved in:", RESULTS_DIR)
