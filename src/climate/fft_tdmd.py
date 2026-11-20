
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


# ---------------------------------------------------------
# 1. Paths and parameters
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "reference"
    / "Tensor_DMD-main"
    / "sst_60x60xT_sub.mat"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "climate"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. Load the SST climate dataset
# ---------------------------------------------------------
print("Loading climate/SST dataset...")

data = loadmat(DATA_PATH)
X = np.asarray(data["X"], dtype=np.float64)

if X.ndim != 3:
    raise ValueError(f"Expected a 3D tensor, received shape {X.shape}")

if not np.isfinite(X).all():
    raise ValueError("The dataset contains NaN or infinite values.")

n, m, T = X.shape
r = T - 1
a = min(n, r)

print(f"Dataset shape: {X.shape}")
print(f"Spatial dimensions: {n} x {m}")
print(f"Number of snapshots: {T}")
print(f"Snapshots used for DMD: {r}")
print(f"Economy SVD rank dimension: {a}")


# ---------------------------------------------------------
# 3. Construct the snapshot tensors
# MATLAB:
# X_t = X(:,:,1:T-1)
# Y_t = X(:,:,2:T)
# X_sw = permute(X_t,[1 3 2])
# ---------------------------------------------------------
X_t = X[:, :, :-1]
Y_t = X[:, :, 1:]

X_sw = np.transpose(X_t, (0, 2, 1))
Y_sw = np.transpose(Y_t, (0, 2, 1))

print("X_sw shape:", X_sw.shape)
print("Y_sw shape:", Y_sw.shape)


# ---------------------------------------------------------
# 4. Apply FFT along the third tensor dimension
# MATLAB: fft(X_sw,[],3)
# Python axis=2
# ---------------------------------------------------------
print("\nApplying FFT along the spatial-width dimension...")

fX = np.fft.fft(X_sw, axis=2)
fY = np.fft.fft(Y_sw, axis=2)

# FFT of the initial snapshot, one value per frequency.
# MATLAB: fft(X_sw(:,1,:),[],3)
X0_fft = np.fft.fft(X_sw[:, 0, :], axis=1)

# Shape: (n, r, m), holding reconstructed values
# in Fourier space for all frequencies.
X_recon_fft = np.zeros(
    (n, r, m),
    dtype=np.complex128
)

frequency_diagnostics = []

# ---------------------------------------------------------
# 5. Perform DMD independently at every FFT frequency
# This follows the operations in Climate_all.m:
#   [u,s,v] = svd(fX(:,:,i),'econ')
#   Atil = u' * fY(:,:,i) * v * pinv(s)
#   [W,D] = eig(Atil)
#   Phi = u*W
#   b = pinv(Phi)*X0_fft
#   reconstruct using Phi*D^t*b
# ---------------------------------------------------------
print("Running frequency-by-frequency DMD...")

for i in range(m):
    Xi = fX[:, :, i]
    Yi = fY[:, :, i]

    # Economy SVD: Xi = U @ S @ Vh
    U, singular_values, Vh = np.linalg.svd(
        Xi,
        full_matrices=False
    )

    # MATLAB's v is V, not Vh.
    V = Vh.conj().T

    # Build the diagonal singular-value matrix.
    S = np.diag(singular_values)

    # Use the pseudoinverse, matching MATLAB pinv(s).
    A_tilde = U.conj().T @ Yi @ V @ np.linalg.pinv(S)

    eigenvalues, W = np.linalg.eig(A_tilde)

    # DMD modes
    Phi = U @ W

    # Initial amplitudes for this frequency
    b = np.linalg.pinv(Phi) @ X0_fft[:, i]

    # MATLAB reconstructs for t = 1,...,r.
    for t in range(1, r + 1):
        time_dynamics = (eigenvalues ** t) * b
        X_recon_fft[:, t - 1, i] = Phi @ time_dynamics

    frequency_diagnostics.append({
        "frequency_index": i,
        "largest_singular_value": float(singular_values[0]),
        "smallest_singular_value": float(singular_values[-1]),
        "max_eigenvalue_magnitude": float(
            np.max(np.abs(eigenvalues))
        ),
    })

    if (i + 1) % 10 == 0 or i == m - 1:
        print(f"  Processed {i + 1}/{m} frequencies")


# ---------------------------------------------------------
# 6. Inverse FFT and restore tensor ordering
# MATLAB: ifft(X_recon_fft,[],3)
# Result becomes n x m x r
# ---------------------------------------------------------
print("\nApplying inverse FFT...")

X_reconstructed_complex = np.fft.ifft(
    X_recon_fft,
    axis=2
)

X_reconstructed = np.transpose(
    X_reconstructed_complex,
    (0, 2, 1)
).real

print("Reconstruction shape:", X_reconstructed.shape)


# ---------------------------------------------------------
# 7. Evaluate against the next snapshots
# MATLAB compares reconstructed frame t against Y_t(:,:,t).
# Thus reconstructed frame 1 corresponds to original frame 2.
# ---------------------------------------------------------
print("\nCalculating reconstruction errors...")

rows = []

for t in range(r):
    target = Y_t[:, :, t]
    estimate = X_reconstructed[:, :, t]

    difference = estimate - target

    target_norm = np.linalg.norm(target)
    relative_error = (
        np.linalg.norm(difference) / target_norm * 100.0
        if target_norm > 0
        else np.nan
    )

    rows.append({
        "prediction_step": t + 1,
        "target_original_frame": t + 2,
        "relative_error_percent": relative_error,
        "MAE": float(np.mean(np.abs(difference))),
        "RMSE": float(np.sqrt(np.mean(difference ** 2))),
    })

metrics = pd.DataFrame(rows)

mean_relative_error = metrics["relative_error_percent"].mean()
mean_mae = metrics["MAE"].mean()
mean_rmse = metrics["RMSE"].mean()

print("\n========== FFT-TDMD CLIMATE RESULTS ==========")
print(f"Mean relative error: {mean_relative_error:.6f}%")
print(f"Mean frame MAE:      {mean_mae:.8f}")
print(f"Mean frame RMSE:     {mean_rmse:.8f}")
print("\nPer-frame results:")
print(metrics.to_string(index=False))


# ---------------------------------------------------------
# 8. Save outputs
# ---------------------------------------------------------
np.save(
    RESULTS_DIR / "fft_tdmd_reconstruction.npy",
    X_reconstructed
)

metrics.to_csv(
    RESULTS_DIR / "fft_tdmd_frame_metrics.csv",
    index=False
)

pd.DataFrame(frequency_diagnostics).to_csv(
    RESULTS_DIR / "fft_tdmd_frequency_diagnostics.csv",
    index=False
)

summary = pd.DataFrame([{
    "method": "FFT-TDMD",
    "dataset_shape": str(X.shape),
    "reconstruction_shape": str(X_reconstructed.shape),
    "mean_relative_error_percent": mean_relative_error,
    "mean_frame_MAE": mean_mae,
    "mean_frame_RMSE": mean_rmse,
}])

summary.to_csv(
    RESULTS_DIR / "fft_tdmd_summary.csv",
    index=False
)


# ---------------------------------------------------------
# 9. Plot per-frame errors
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(
    metrics["target_original_frame"],
    metrics["relative_error_percent"],
    marker="o"
)
plt.xlabel("Target original frame")
plt.ylabel("Relative reconstruction error (%)")
plt.title("FFT-TDMD: Climate/SST Per-Frame Error")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "fft_tdmd_error_plot.png",
    dpi=200
)
plt.close()


# ---------------------------------------------------------
# 10. Plot a target frame and its reconstruction
# ---------------------------------------------------------
frame_index = min(5, r - 1)

target_frame = Y_t[:, :, frame_index]
reconstructed_frame = X_reconstructed[:, :, frame_index]
absolute_error = np.abs(reconstructed_frame - target_frame)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

images = [
    axes[0].imshow(target_frame, cmap="viridis"),
    axes[1].imshow(reconstructed_frame, cmap="viridis"),
    axes[2].imshow(absolute_error, cmap="magma"),
]

axes[0].set_title(
    f"Target: original frame {frame_index + 2}"
)
axes[1].set_title("FFT-TDMD reconstruction")
axes[2].set_title("Absolute error")

for ax in axes:
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

fig.colorbar(images[0], ax=axes[0], fraction=0.046)
fig.colorbar(images[1], ax=axes[1], fraction=0.046)
fig.colorbar(images[2], ax=axes[2], fraction=0.046)

fig.tight_layout()
fig.savefig(
    RESULTS_DIR / "fft_tdmd_frame_comparison.png",
    dpi=200
)
plt.close(fig)


print("\nSaved results in:", RESULTS_DIR)
print("  fft_tdmd_reconstruction.npy")
print("  fft_tdmd_frame_metrics.csv")
print("  fft_tdmd_frequency_diagnostics.csv")
print("  fft_tdmd_summary.csv")
print("  fft_tdmd_error_plot.png")
print("  fft_tdmd_frame_comparison.png")
print("\nFFT-TDMD experiment finished.")
