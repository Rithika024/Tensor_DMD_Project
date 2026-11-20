
import numpy as np
from scipy.fft import dct, idct
from pathlib import Path

from tdmd_cosine import fit_dmd_modes

ROOT = Path(__file__).resolve().parents[1]
video = np.load(ROOT / "outputs" / "synthetic_video.npy")
true_background = np.load(ROOT / "outputs" / "synthetic_background.npy")

n, m, T = video.shape
dt = 1.0
tau_re = 2e-3
tau_im = 0.02

# Match the preprocessing in the MATLAB reference.
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

mp = 2 ** int(np.ceil(np.log2(2 * m)))

X_pad = np.pad(X_sw, ((0, 0), (0, 0), (0, mp - m)))
Y_pad = np.pad(Y_sw, ((0, 0), (0, 0), (0, mp - m)))

for norm in [None, "ortho"]:
    print(f"\nTesting norm={norm}")

    fX = dct(X_pad, type=2, axis=2, norm=norm)
    fY = dct(Y_pad, type=2, axis=2, norm=norm)

    background_coefficients = np.zeros(
        (n, T, mp), dtype=complex
    )

    selected_counts = []
    fallback_count = 0
    invalid_count = 0

    for k in range(mp):
        fitted = fit_dmd_modes(fX[:, :, k], fY[:, :, k])

        if fitted is None:
            continue

        eigenvalues, modes, amplitudes = fitted
        valid = (
            np.isfinite(eigenvalues)
            & (np.abs(eigenvalues) > 1e-12)
        )

        invalid_count += np.count_nonzero(~valid)

        omega = np.full(
            eigenvalues.shape, np.nan + 1j * np.nan
        )
        omega[valid] = np.log(eigenvalues[valid]) / dt

        stationary = (
            valid
            & (np.abs(omega.real) < tau_re)
            & (np.abs(omega.imag) < tau_im)
        )

        selected = np.flatnonzero(stationary)

        if selected.size == 0 and np.any(valid):
            valid_indices = np.flatnonzero(valid)
            closest = valid_indices[
                np.argmin(np.abs(omega[valid_indices]))
            ]
            selected = np.array([closest])
            fallback_count += 1

        selected_counts.append(len(selected))

        if selected.size:
            dynamics = (
                amplitudes[selected, None]
                * eigenvalues[selected, None]
                ** np.arange(T)[None, :]
            )
            background_coefficients[:, :, k] = (
                modes[:, selected] @ dynamics
            )

    bg_padded = idct(
        background_coefficients,
        type=2,
        axis=2,
        norm=norm,
    )

    bg_detrended = np.real(
        np.transpose(bg_padded[:, :, :m], (0, 2, 1))
    )
    estimated_background = bg_detrended + background_rows

    frame = 9
    mae = np.mean(
        np.abs(estimated_background[:, :, frame] - true_background)
    )

    print("Background MAE at frame 10:", f"{mae:.8f}")
    print("Fallback frequency bins:", fallback_count)
    print("Invalid eigenvalues:", invalid_count)
    print(
        "Selected modes per bin:",
        f"min={min(selected_counts)},",
        f"max={max(selected_counts)},",
        f"mean={np.mean(selected_counts):.2f}",
    )
