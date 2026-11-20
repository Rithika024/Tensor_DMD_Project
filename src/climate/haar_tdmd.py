from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

from haar_transform import haar_blocks, inverse_haar_blocks


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
# 2. Load climate dataset
# ---------------------------------------------------------
print("Loading climate/SST dataset...")

data = loadmat(DATA_PATH)
X = np.asarray(data["X"], dtype=np.float64)

if X.ndim != 3:
    raise ValueError(f"Expected a 3D tensor; received {X.shape}")

if not np.isfinite(X).all():
    raise ValueError("Dataset contains NaN or infinite values.")

n, m, T = X.shape
r = T - 1
a = min(n, r)

print("Original shape:", X.shape)
print("Number of snapshots:", T)
print("DMD snapshot pairs:", r)
print("Economy SVD rank dimension:", a)


# ---------------------------------------------------------
# 3. Construct snapshot tensors
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
# 4. Apply Haar block transforms
# MATLAB:
# Xh = fast_bdwt_haar_blocks(X_sw)
# Yh = fast_bdwt_haar_blocks(Y_sw)
# ---------------------------------------------------------
print("\nApplying Haar block transform...")

Xh = haar_blocks(X_sw)
Yh = haar_blocks(Y_sw)

print("Xh shape:", Xh.shape)
print("Yh shape:", Yh.shape)


# ---------------------------------------------------------
# 5. Frequency/slice-wise DMD in Haar domain
# MATLAB:
# [u,s,v] = svd(Xh(:,:,i),'econ')
# Atil = u' * Yh(:,:,i) * v * pinv(s)
# [W,D] = eig(Atil)
# Phi = u*W
# ---------------------------------------------------------
print("\nRunning Haar-TDMD...")

Phi_h = []
Lam_h = []
b_h = []

# MATLAB X0_h = fast_bdwt_haar_blocks(X_sw(:,1,:))
X0_h = haar_blocks(X_sw[:, 0:1, :])

diagnostics = []

for i in range(m):
    Xi = Xh[:, :, i]
    Yi = Yh[:, :, i]

    U, singular_values, Vh = np.linalg.svd(
        Xi,
        full_matrices=False
    )

    V = Vh.conj().T
    S = np.diag(singular_values)

    A_tilde = (
        U.conj().T
        @ Yi
        @ V
        @ np.linalg.pinv(S)
    )

    eigenvalues, W = np.linalg.eig(A_tilde)

    Phi = U @ W
    b = np.linalg.pinv(Phi) @ X0_h[:, 0, i]

    Phi_h.append(Phi)
    Lam_h.append(eigenvalues)
    b_h.append(b)

    diagnostics.append({
        "slice_index": i,
        "largest_singular_value": float(singular_values[0]),
        "smallest_singular_value": float(singular_values[-1]),
        "max_eigenvalue_magnitude": float(
            np.max(np.abs(eigenvalues))
        ),
    })

    if (i + 1) % 10 == 0 or i == m - 1:
        print(f"  Processed {i + 1}/{m} slices")


# ---------------------------------------------------------
# 6. Reconstruct in the Haar domain
# MATLAB:
# Xt_h(:,t,i) = Phi_h(:,:,i) * (Lam_h(:,:,i)^t) * b_h(:,:,i)
# ---------------------------------------------------------
print("\nReconstructing in the Haar domain...")

Xt_h = np.zeros((n, r, m), dtype=np.complex128)

for t in range(1, r + 1):
    for i in range(m):
        Xt_h[:, t - 1, i] = (
            Phi_h[i]
            @ ((Lam_h[i] ** t) * b_h[i])
        )


# ---------------------------------------------------------
# 7. Inverse Haar transform
# MATLAB:
# Xrec_haar_all = fast_bdwt_haar_blocks_inv(Xt_h)
# ---------------------------------------------------------
print("Applying inverse Haar transform...")

X_reconstructed_complex = inverse_haar_blocks(Xt_h)

# Real-valued input is expected, as in the MATLAB reference.
imaginary_max = float(
    np.max(np.abs(X_reconstructed_complex.imag))
)

X_reconstructed = X_reconstructed_complex.real
X_reconstructed = np.transpose(
    X_reconstructed,
    (0, 2, 1)
)

print("Reconstruction shape:", X_reconstructed.shape)
print("Maximum imaginary component:", imaginary_max)


# ---------------------------------------------------------
# 8. Evaluate against next snapshots
# ---------------------------------------------------------
print("\nCalculating frame-wise errors...")

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
        "max_absolute_error": float(np.max(np.abs(difference))),
    })

metrics = pd.DataFrame(rows)

print("\n========== HAAR-TDMD CLIMATE RESULTS ==========")
print(
    "Mean relative error: "
    f"{metrics['relative_error_percent'].mean():.8f}%"
)
print(f"Mean frame MAE: {metrics['MAE'].mean():.10e}")
print(f"Mean frame RMSE: {metrics['RMSE'].mean():.10e}")
print("\nPer-frame results:")
print(metrics.to_string(index=False))


# ---------------------------------------------------------
# 9. Save outputs
# ---------------------------------------------------------
np.save(
    RESULTS_DIR / "haar_tdmd_reconstruction.npy",
    X_reconstructed
)

metrics.to_csv(
    RESULTS_DIR / "haar_tdmd_frame_metrics.csv",
    index=False
)

pd.DataFrame(diagnostics).to_csv(
    RESULTS_DIR / "haar_tdmd_slice_diagnostics.csv",
    index=False
)

summary = pd.DataFrame([{
    "method": "Haar-TDMD",
    "dataset_shape": str(X.shape),
    "reconstruction_shape": str(X_reconstructed.shape),
    "mean_relative_error_percent": (
        metrics["relative_error_percent"].mean()
    ),
    "mean_frame_MAE": metrics["MAE"].mean(),
    "mean_frame_RMSE": metrics["RMSE"].mean(),
    "max_imaginary_component": imaginary_max,
}])

summary.to_csv(
    RESULTS_DIR / "haar_tdmd_summary.csv",
    index=False
)


# ---------------------------------------------------------
# 10. Plot per-frame error
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(
    metrics["target_original_frame"],
    metrics["relative_error_percent"],
    marker="o"
)
plt.xlabel("Target original frame")
plt.ylabel("Relative error (%)")
plt.title("Haar-TDMD: Climate/SST Per-Frame Error")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "haar_tdmd_error_plot.png",
    dpi=200
)
plt.close()


# ---------------------------------------------------------
# 11. Plot target, reconstruction and absolute error
# ---------------------------------------------------------
frame_index = min(5, r - 1)

target_frame = Y_t[:, :, frame_index]
reconstructed_frame = X_reconstructed[:, :, frame_index]
absolute_error = np.abs(reconstructed_frame - target_frame)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

im0 = axes[0].imshow(target_frame, cmap="viridis")
im1 = axes[1].imshow(reconstructed_frame, cmap="viridis")
im2 = axes[2].imshow(absolute_error, cmap="magma")

axes[0].set_title(f"Target: original frame {frame_index + 2}")
axes[1].set_title("Haar-TDMD reconstruction")
axes[2].set_title("Absolute error")

for ax in axes:
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

fig.colorbar(im0, ax=axes[0], fraction=0.046)
fig.colorbar(im1, ax=axes[1], fraction=0.046)
fig.colorbar(im2, ax=axes[2], fraction=0.046)

fig.tight_layout()
fig.savefig(
    RESULTS_DIR / "haar_tdmd_frame_comparison.png",
    dpi=200
)
plt.close(fig)


print("\nSaved Haar-TDMD results in:", RESULTS_DIR)
print("Haar-TDMD experiment finished.")