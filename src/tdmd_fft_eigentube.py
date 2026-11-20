from pathlib import Path
import csv
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results" / "eigentube_norm_experiment"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

video_path = OUTPUT_DIR / "synthetic_video.npy"
if not video_path.exists():
    raise FileNotFoundError(f"Could not find input video: {video_path}")

video = np.load(video_path)
if video.ndim != 3:
    raise ValueError(f"Expected a 3-D video tensor (height, width, frames), got {video.shape}")

n, m, T = video.shape
if T < 3:
    raise ValueError("At least 3 frames are required for DMD.")

dt = 1.0
threshold = 0.995
print("Loaded video:", video.shape)
print("Method: FFT-TDMD with paper-based eigentube Frobenius-norm separation")
print("Threshold:", threshold)

# Preserve the preprocessing used by the existing MATLAB/Python F-B experiment.
row_mean = np.mean(video, axis=1, keepdims=True)
video_detrended = video - row_mean
row_mean_bg = np.median(row_mean, axis=2, keepdims=True)
background_rows = np.broadcast_to(row_mean_bg, (n, m, T)).copy()

# Each frame is represented as an n x 1 x m tensor. Concatenating snapshots
# along mode 2 gives the n x (T-1) x m trajectory tensors used by TDMD.
X = video_detrended[:, :, :-1]
Y = video_detrended[:, :, 1:]
X_sw = np.transpose(X, (0, 2, 1))
Y_sw = np.transpose(Y, (0, 2, 1))
fX = np.fft.fft(X_sw, axis=2)
fY = np.fft.fft(Y_sw, axis=2)

r = min(fX.shape[0], fX.shape[1])
frequency_eigenvalues = np.empty((r, m), dtype=np.complex128)
frequency_modes = []
frequency_amplitudes = []

# Fourier-slice implementation of TSVD -> reduced operator -> slice EVD.
for k in range(m):
    Xi = fX[:, :, k]
    Yi = fY[:, :, k]
    U, singular_values, Vh = np.linalg.svd(Xi, full_matrices=False)
    rank = min(Xi.shape)
    Ur = U[:, :rank]
    sr = singular_values[:rank]
    Vr = Vh[:rank, :].conj().T
    A_tilde = (Ur.conj().T @ Yi @ Vr) @ np.linalg.pinv(np.diag(sr))

    eigenvalues, W = np.linalg.eig(A_tilde)
    Phi = Ur @ W
    amplitudes = np.linalg.pinv(Phi) @ Xi[:, 0]

    frequency_eigenvalues[:, k] = eigenvalues
    frequency_modes.append(Phi)
    frequency_amplitudes.append(amplitudes)

# Track eigenvalue identities on the nonnegative half-spectrum,
# then explicitly match negative frequencies to conjugate partners.

def apply_permutation(k, permutation):
    frequency_eigenvalues[:, k] = frequency_eigenvalues[permutation, k]
    frequency_modes[k] = frequency_modes[k][:, permutation]
    frequency_amplitudes[k] = frequency_amplitudes[k][permutation]

# Anchor track 1 at DC to the eigenvalue closest to one.
dc_values = frequency_eigenvalues[:, 0].copy()
unit_index = int(np.argmin(np.abs(dc_values - 1.0)))
remaining = np.array([j for j in range(r) if j != unit_index])

order = np.lexsort((
    dc_values[remaining].imag,
    dc_values[remaining].real
))
dc_permutation = np.concatenate(([unit_index], remaining[order]))
apply_permutation(0, dc_permutation)

# Track continuously from DC through the Nyquist frequency.
for k in range(1, m // 2 + 1):
    previous = frequency_eigenvalues[:, k - 1].copy()
    current = frequency_eigenvalues[:, k].copy()

    cost = np.abs(previous[:, None] - current[None, :])
    rows, columns = linear_sum_assignment(cost)

    permutation = np.empty(r, dtype=int)
    permutation[rows] = columns
    apply_permutation(k, permutation)

# Match each negative-frequency slice to the conjugate of its
# corresponding positive-frequency slice.
for k in range(1, (m + 1) // 2):
    partner = m - k
    positive_values = frequency_eigenvalues[:, k].copy()
    negative_values = frequency_eigenvalues[:, partner].copy()

    cost = np.abs(
        np.conj(positive_values[:, None]) -
        negative_values[None, :]
    )
    rows, columns = linear_sum_assignment(cost)

    permutation = np.empty(r, dtype=int)
    permutation[rows] = columns
    apply_permutation(partner, permutation)

print("Half-spectrum eigenvalue tracking completed.")
print("Negative frequencies matched to conjugate positive frequencies.")
print("Track 1 is anchored at DC to the eigenvalue closest to one.")

# The diagonal of each Fourier-domain reduced EVD is assembled into the
# F-diagonal tensor D. Inverse FFT along the tensor third mode gives D_jj: tubes.
eigentubes = np.fft.ifft(frequency_eigenvalues, axis=1)
eigentube_norms = np.linalg.norm(eigentubes, axis=1)
persistent_mask = eigentube_norms > threshold
transient_mask = ~persistent_mask

# Save the classification table for auditability.
with (RESULTS_DIR / "eigentube_norms.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["eigentube_index_1_based", "frobenius_norm", "classification", "threshold"])
    for j, norm_value in enumerate(eigentube_norms):
        writer.writerow([j + 1, f"{norm_value:.12g}", "persistent" if persistent_mask[j] else "transient", threshold])

print("Reduced rank:", r)
print("Persistent eigentubes:", int(np.sum(persistent_mask)))
print("Transient eigentubes:", int(np.sum(transient_mask)))
print("Eigentube Frobenius norms:", np.array2string(eigentube_norms, precision=5))

# Reconstruct persistent and transient states independently in the Fourier domain.
Z_persistent = np.zeros((n, T, m), dtype=np.complex128)
Z_transient = np.zeros((n, T, m), dtype=np.complex128)
Z_all = np.zeros((n, T, m), dtype=np.complex128)
for k in range(m):
    Phi = frequency_modes[k]
    lam = frequency_eigenvalues[:, k]
    b = frequency_amplitudes[k]

    for t in range(T):
        temporal_coefficients = (lam ** t) * b
        Z_all[:, t, k] = Phi @ temporal_coefficients
        Z_persistent[:, t, k] = Phi[:, persistent_mask] @ temporal_coefficients[persistent_mask]
        Z_transient[:, t, k] = Phi[:, transient_mask] @ temporal_coefficients[transient_mask]


# Verify that each reconstructed component is real-valued
# before discarding numerical imaginary residuals.
for name, Z in [
    ("Full reconstruction", Z_all),
    ("Persistent component", Z_persistent),
    ("Transient component", Z_transient),
]:
    complex_result = np.fft.ifft(Z, axis=2)
    imaginary_ratio = (
        np.linalg.norm(complex_result.imag) /
        (np.linalg.norm(complex_result.real) + 1e-12)
    )
    print(f"{name} imaginary residual ratio: {imaginary_ratio:.3e}")

# Return from Fourier coordinates to the original tensor coordinates.
def inverse_tensor_transform(Z):
    return np.transpose(np.fft.ifft(Z, axis=2).real, (0, 2, 1))

reconstruction_detrended = inverse_tensor_transform(Z_all)
persistent_detrended = inverse_tensor_transform(Z_persistent)
transient_detrended = inverse_tensor_transform(Z_transient)

# Restore the slowly varying row-mean background used by the reference experiment.
estimated_background = persistent_detrended + background_rows
estimated_foreground = transient_detrended
reconstruction = reconstruction_detrended + background_rows

# Reconstruction diagnostics over all available frames.
per_frame_errors = []
for t in range(T):
    numerator = np.linalg.norm(video[:, :, t] - reconstruction[:, :, t])
    denominator = max(np.linalg.norm(video[:, :, t]), 1e-15)
    per_frame_errors.append(numerator / denominator)
overall_relative_error = np.linalg.norm((video - reconstruction).ravel()) / max(np.linalg.norm(video.ravel()), 1e-15)

print(f"Overall relative reconstruction error (all frames): {overall_relative_error:.8f}")
print(f"Mean per-frame relative reconstruction error: {np.mean(per_frame_errors):.8f}")

np.save(RESULTS_DIR / "tdmd_eigentube_reconstruction.npy", reconstruction)
np.save(RESULTS_DIR / "tdmd_eigentube_background.npy", estimated_background)
np.save(RESULTS_DIR / "tdmd_eigentube_foreground.npy", estimated_foreground)
np.save(RESULTS_DIR / "tdmd_eigentube_frame_errors.npy", np.asarray(per_frame_errors))
np.save(RESULTS_DIR / "tdmd_eigentubes.npy", eigentubes)
np.save(RESULTS_DIR / "tdmd_eigentube_norms.npy", eigentube_norms)
np.save(RESULTS_DIR / "tdmd_eigentube_persistent_mask.npy", persistent_mask)

frame_index = min(9, T - 1)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(video[:, :, frame_index], cmap="gray", vmin=0, vmax=1)
axes[0].set_title(f"Original frame {frame_index + 1}")
axes[1].imshow(estimated_background[:, :, frame_index], cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Persistent component + row background")
axes[2].imshow(np.clip(estimated_foreground[:, :, frame_index], 0, 1), cmap="gray", vmin=0, vmax=1)
axes[2].set_title("Transient component")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "eigentube_foreground_background.png", dpi=200, bbox_inches="tight")
plt.close(fig)

plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, T + 1), per_frame_errors, marker="o")
plt.xlabel("Frame number")
plt.ylabel("Relative reconstruction error")
plt.title("FFT-TDMD Eigentube Reconstruction Error")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "eigentube_reconstruction_errors.png", dpi=200, bbox_inches="tight")
plt.close()

print("\nSaved eigentube experiment outputs to:", RESULTS_DIR)
print("NOTE: Slice-wise eigendecompositions use half-spectrum eigenvalue tracking and conjugate-frequency matching.")
print("Check the eigentube norms and reconstruction plots before interpreting the split.")


