

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Load the noisy challenge video
video = np.load(OUTPUTS / "noisy_challenge_video.npy")
print("Loaded noisy video:", video.shape)

# Convert to floating point
video = video.astype(np.float64)
height, width, frames = video.shape

# Remove the spatial row mean before applying FFT
row_mean = np.mean(video, axis=1, keepdims=True)
video_centered = video - row_mean

# FFT across the width dimension
video_fft = np.fft.fft(video_centered, axis=1)
reconstruction_fft = np.zeros_like(video_fft, dtype=np.complex128)
background_fft = np.zeros_like(video_fft, dtype=np.complex128)

# Process each spatial-frequency slice using DMD
for k in range(width):
    X = video_fft[:, k, :-1]
    Y = video_fft[:, k, 1:]

    U, s, Vh = np.linalg.svd(X, full_matrices=False)

    if len(s) == 0:
        continue

    tolerance = np.finfo(float).eps * max(X.shape) * s[0]
    rank = int(np.sum(s > tolerance))

    if rank == 0:
        continue

    U_r = U[:, :rank]
    S_r = np.diag(s[:rank])
    V_r = Vh.conj().T[:, :rank]

    A_tilde = U_r.conj().T @ Y @ V_r @ np.linalg.inv(S_r)
    eigenvalues, W = np.linalg.eig(A_tilde)

    modes = Y @ V_r @ np.linalg.inv(S_r) @ W

    # Initial modal amplitudes
    b = np.linalg.lstsq(modes, X[:, 0], rcond=None)[0]

    # Reconstruct all input frames using discrete-time eigenvalue powers
    time_indices = np.arange(frames - 1)
    dynamics = np.zeros((rank, frames - 1), dtype=np.complex128)

    for j in range(rank):
        dynamics[j, :] = b[j] * (eigenvalues[j] ** time_indices)

    reconstructed_slice = modes @ dynamics
    reconstruction_fft[:, k, :-1] = reconstructed_slice

    # Select slowly varying modes as background
    omega = np.zeros(rank, dtype=np.complex128)
    nonzero = np.abs(eigenvalues) > 1e-12
    omega[nonzero] = np.log(eigenvalues[nonzero])

    tau_re = 2e-3
    tau_im = 0.02

    background_modes = (
        (np.abs(omega.real) < tau_re)
        & (np.abs(omega.imag) < tau_im)
    )

    if not np.any(background_modes):
        background_modes[np.argmin(np.abs(omega))] = True

    background_slice = (
        modes[:, background_modes]
        @ dynamics[background_modes, :]
    )
    background_fft[:, k, :-1] = background_slice

# Inverse FFT
reconstruction = np.fft.ifft(
    reconstruction_fft, axis=1
).real

background_centered = np.fft.ifft(
    background_fft, axis=1
).real

# Restore the removed row mean to the background estimate
background_mean = np.median(row_mean, axis=2, keepdims=True)
background = background_centered + background_mean

# Foreground residual
foreground = video[:, :, :-1] - background[:, :, :-1]

# Save results
np.save(RESULTS / "noisy_fft_reconstruction.npy", reconstruction)
np.save(RESULTS / "noisy_fft_background.npy", background)
np.save(RESULTS / "noisy_fft_foreground.npy", foreground)

print("FFT-TDMD noisy challenge completed.")
print("Reconstruction shape:", reconstruction.shape)
print("Background shape:", background.shape)
print("Foreground shape:", foreground.shape)
print("All finite:", np.isfinite(foreground).all())
print("Saved noisy FFT-TDMD results.")
