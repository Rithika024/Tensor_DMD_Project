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

video = np.load(OUTPUT_DIR / "challenge_video.npy")
n, m, T = video.shape

dt = 1.0
tau_re = 2e-3
tau_im = 0.02

print("Loaded challenge video:", video.shape)
print("Method: FFT-TDMD Challenge")


# --------------------------------------------------
# 2. Detrend each frame by subtracting row means
# --------------------------------------------------
row_mean = np.mean(video, axis=1, keepdims=True)
video_detrended = video - row_mean

row_mean_bg = np.median(row_mean, axis=2, keepdims=True)
background_rows = np.broadcast_to(
    row_mean_bg, (n, m, T)
).copy()

X = video_detrended[:, :, :-1]
Y = video_detrended[:, :, 1:]

X_sw = np.transpose(X, (0, 2, 1))
Y_sw = np.transpose(Y, (0, 2, 1))

fX = np.fft.fft(X_sw, axis=2)
fY = np.fft.fft(Y_sw, axis=2)


# --------------------------------------------------
# 3. Fit DMD to each frequency slice
# --------------------------------------------------
def fit_frequency_slice(Xi, Yi):
    """Fit DMD to one frequency slice."""
    U, singular_values, Vh = np.linalg.svd(
        Xi, full_matrices=False
    )

    if len(singular_values) == 0:
        return np.zeros_like(Xi), np.zeros_like(Xi)

    tolerance = (
        np.finfo(float).eps
        * max(Xi.shape)
        * singular_values[0]
    )
    rank = int(np.sum(singular_values > tolerance))

    if rank == 0:
        return np.zeros_like(Xi), np.zeros_like(Xi)

    U_r = U[:, :rank]
    s_r = singular_values[:rank]
    V_r = Vh.conj().T[:, :rank]

    A_tilde = (
        U_r.conj().T @ Yi @ V_r
    ) @ np.diag(1.0 / s_r)

    eigenvalues, W = np.linalg.eig(A_tilde)
    modes = U_r @ W

    with np.errstate(divide="ignore", invalid="ignore"):
        omega = np.log(eigenvalues.astype(complex)) / dt

    valid = (
        np.isfinite(omega.real)
        & np.isfinite(omega.imag)
    )

    is_background = (
        valid
        & (np.abs(omega.real) < tau_re)
        & (np.abs(omega.imag) < tau_im)
    )

    if not np.any(is_background):
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices):
            best = valid_indices[
                np.argmin(np.abs(omega[valid_indices]))
            ]
            is_background[best] = True

    # Xi contains T-1 snapshots. Reconstruct that same number
    # of time points so the result matches the allocated arrays.
    time_indices = np.arange(Xi.shape[1])

    if not np.any(valid):
        return np.zeros_like(Xi), np.zeros_like(Xi)

    all_amplitudes = np.linalg.lstsq(
        modes, Xi[:, 0], rcond=None
    )[0]

    
    time_dynamics = np.power(
        eigenvalues[:, None],
        time_indices[None, :]
    )

    full_reconstruction = modes @ (
        all_amplitudes[:, None] * time_dynamics
    )


    if not np.any(is_background):
        background_reconstruction = np.zeros_like(
            full_reconstruction
        )
    else:
        background_modes = modes[:, is_background]
        background_eigenvalues = eigenvalues[is_background]

        amplitudes = np.linalg.lstsq(
            background_modes, Xi[:, 0], rcond=None
        )[0]

        
        background_dynamics = np.power(
           background_eigenvalues[:, None],
           time_indices[None, :]
        )

        background_reconstruction = background_modes @ (
            amplitudes[:, None] * background_dynamics
        )


    return full_reconstruction, background_reconstruction


# --------------------------------------------------
# 4. Process every frequency slice
# --------------------------------------------------

# The DMD model reconstructs the first T-1 snapshots.
f_reconstructed = np.zeros(
    (n, T - 1, m), dtype=complex
)
f_background = np.zeros(
    (n, T - 1, m), dtype=complex
)


for j in range(m):
    full_j, bg_j = fit_frequency_slice(
        fX[:, :, j], fY[:, :, j]
    )
    f_reconstructed[:, :, j] = full_j
    f_background[:, :, j] = bg_j

    if (j + 1) % 10 == 0:
        print(f"Processed {j + 1}/{m} frequency slices")


# --------------------------------------------------
# 5. Inverse FFT and restore row background
# --------------------------------------------------
reconstructed_detrended = np.fft.ifft(
    f_reconstructed, axis=2
).real

background_detrended = np.fft.ifft(
    f_background, axis=2
).real

reconstructed = np.transpose(
    reconstructed_detrended, (0, 2, 1)
) + background_rows[:, :, :T - 1]

estimated_background = np.transpose(
    background_detrended, (0, 2, 1)
) + background_rows[:, :, :T - 1]


video_used = video[:, :, :T - 1]

estimated_foreground = np.maximum(
    video_used - estimated_background, 0
)


# Match the first T-1 snapshots used by the original DMD pipeline.
reconstructed = reconstructed[:, :, :T - 1]
estimated_background = estimated_background[:, :, :T - 1]
estimated_foreground = estimated_foreground[:, :, :T - 1]

np.save(
    RESULTS_DIR / "challenge_fft_reconstruction.npy",
    reconstructed
)
np.save(
    RESULTS_DIR / "challenge_fft_background.npy",
    estimated_background
)
np.save(
    RESULTS_DIR / "challenge_fft_foreground.npy",
    estimated_foreground
)

print("\nChallenge FFT-TDMD completed.")
print("Reconstruction shape:", reconstructed.shape)
print("Background shape:", estimated_background.shape)
print("Foreground shape:", estimated_foreground.shape)
print(
    "All outputs finite:",
    np.isfinite(reconstructed).all()
    and np.isfinite(estimated_background).all()
    and np.isfinite(estimated_foreground).all()
)
print(
    "Saved results to:",
    RESULTS_DIR
)
