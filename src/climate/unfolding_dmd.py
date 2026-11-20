
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "reference" / "Tensor_DMD-main"
    / "sst_60x60xT_sub.mat"
)

OUT = ROOT / "results" / "climate"
OUT.mkdir(parents=True, exist_ok=True)

PINV_TOL = 1e-12


# --------------------------------------------------
# 2. MATLAB-compatible block-circulant construction
# --------------------------------------------------

def bcirc(A):
    """
    Construct the block-circulant matrix of a 3D tensor.

    A has shape (n1, n2, n3).
    Output has shape (n1*n3, n2*n3).

    Block layout:
        A1  A3  A2 ...
        A2  A1  A3 ...
        A3  A2  A1 ...
    """
    A = np.asarray(A)

    if A.ndim != 3:
        raise ValueError(
            f"bcirc expects a 3D tensor; received {A.shape}"
        )

    n1, n2, n3 = A.shape
    result = np.zeros(
        (n1 * n3, n2 * n3),
        dtype=A.dtype
    )

    for block_row in range(n3):
        for block_col in range(n3):
            # Standard block-circulant indexing.
            slice_index = (block_row - block_col) % n3

            result[
                block_row * n1:(block_row + 1) * n1,
                block_col * n2:(block_col + 1) * n2
            ] = A[:, :, slice_index]

    return result


# --------------------------------------------------
# 3. Load data
# --------------------------------------------------

data = loadmat(DATA_PATH)
X = np.asarray(data["X"], dtype=np.float64)

if X.ndim != 3 or not np.all(np.isfinite(X)):
    raise ValueError("X must be a finite 3D tensor.")

n, m, T = X.shape

X_t = X[:, :, :-1]
Y_t = X[:, :, 1:]

r = T - 1

# MATLAB: permute(X_t,[1 3 2])
X_sw = np.transpose(X_t, (0, 2, 1))
Y_sw = np.transpose(Y_t, (0, 2, 1))

print("=" * 60)
print("UNFOLDING DMD: WHOLE BLOCK-CIRCULANT MATRIX")
print("=" * 60)
print(f"Original data shape: {X.shape}")
print(f"X_sw shape: {X_sw.shape}")
print(f"Y_sw shape: {Y_sw.shape}")


# --------------------------------------------------
# 4. Build whole block-circulant matrices
# --------------------------------------------------

Xb = bcirc(X_sw)
Yb = bcirc(Y_sw)

print(f"Xb shape: {Xb.shape}")
print(f"Yb shape: {Yb.shape}")

expected_shape = (n * m, r * m)

if Xb.shape != expected_shape or Yb.shape != expected_shape:
    raise ValueError(
        f"Expected both matrices to have shape {expected_shape}."
    )


# --------------------------------------------------
# 5. DMD on the complete block-circulant matrices
# --------------------------------------------------
# MATLAB:
# [Uu,Su,Vu] = svd(Xb,'econ');
# Atil_unf = Uu' * Yb * Vu * pinv(Su);
# [W_unf,D_unf] = eig(Atil_unf);
# Phi_unf = Uu * W_unf;

print("\nComputing economy SVD of Xb...")
Uu, singular_values, Vh = np.linalg.svd(
    Xb, full_matrices=False
)

Vu = Vh.conj().T
Su = np.diag(singular_values)

Su_pinv = np.linalg.pinv(Su, rcond=PINV_TOL)

print(f"Uu shape: {Uu.shape}")
print(f"Su shape: {Su.shape}")
print(f"Vu shape: {Vu.shape}")

print("Computing reduced DMD operator...")
Atil = Uu.conj().T @ Yb @ Vu @ Su_pinv

print(f"Reduced operator shape: {Atil.shape}")
print("Computing eigendecomposition...")

eigenvalues, W_unf = np.linalg.eig(Atil)

Phi_unf = Uu @ W_unf

print(f"Eigenvalue count: {len(eigenvalues)}")
print(f"Phi_unf shape: {Phi_unf.shape}")


# --------------------------------------------------
# 6. Initial amplitudes
# --------------------------------------------------
# MATLAB: b_unf = pinv(Phi_unf) * Xb(:,1)

print("Computing initial amplitudes...")

b_unf = (
    np.linalg.pinv(Phi_unf, rcond=PINV_TOL)
    @ Xb[:, 0]
)


# --------------------------------------------------
# 7. Reconstruct frames
# --------------------------------------------------
# MATLAB uses x_t = Phi_unf * D_unf^t * b_unf.
# Store each vector as an n-by-m frame using
# MATLAB's column-major reshape convention.

Xrec = np.zeros((n, m, r), dtype=np.float64)

for t in range(1, r + 1):
    x_t = Phi_unf @ (
        (eigenvalues ** t) * b_unf
    )

    Xrec[:, :, t - 1] = np.real(
        np.reshape(x_t, (n, m), order="F")
    )

    if not np.all(np.isfinite(Xrec[:, :, t - 1])):
        raise FloatingPointError(
            f"Non-finite values in reconstructed frame {t}."
        )

    print(f"Reconstructed frame {t}/{r}")

print(f"\nReconstruction shape: {Xrec.shape}")


# --------------------------------------------------
# 8. Evaluate against Y_t
# --------------------------------------------------

target = Y_t
metrics = []

for j in range(r):
    actual = target[:, :, j]
    predicted = Xrec[:, :, j]
    error = predicted - actual

    actual_norm = np.linalg.norm(actual)
    error_norm = np.linalg.norm(error)

    relative_error = (
        100.0 * error_norm / actual_norm
        if actual_norm > 0 else np.nan
    )

    metrics.append({
        "original_frame": j + 2,
        "mae": np.mean(np.abs(error)),
        "rmse": np.sqrt(np.mean(error ** 2)),
        "max_absolute_error": np.max(np.abs(error)),
        "relative_error_percent": relative_error,
    })

metrics_df = pd.DataFrame(metrics)

difference = Xrec - target

overall_mae = float(np.mean(np.abs(difference)))
overall_rmse = float(np.sqrt(np.mean(difference ** 2)))
overall_relative_error = float(
    100.0 * np.linalg.norm(difference)
    / max(np.linalg.norm(target), np.finfo(float).eps)
)

summary_df = pd.DataFrame([{
    "method": "Unfolding DMD",
    "input_frames": T,
    "reconstructed_frames": r,
    "overall_mae": overall_mae,
    "overall_rmse": overall_rmse,
    "overall_relative_error_percent": overall_relative_error,
    "mean_frame_relative_error_percent":
        metrics_df["relative_error_percent"].mean(),
    "max_frame_relative_error_percent":
        metrics_df["relative_error_percent"].max(),
}])

print("\n" + "=" * 60)
print("UNFOLDING DMD RESULTS")
print("=" * 60)
print(f"Overall MAE: {overall_mae:.10f}")
print(f"Overall RMSE: {overall_rmse:.10f}")
print(f"Overall relative error: {overall_relative_error:.8f}%")
print("\nPer-frame metrics:")
print(metrics_df.to_string(index=False))


# --------------------------------------------------
# 9. Save numerical outputs
# --------------------------------------------------

np.save(
    OUT / "unfolding_dmd_reconstruction.npy",
    Xrec
)

metrics_df.to_csv(
    OUT / "unfolding_dmd_frame_metrics.csv",
    index=False
)

summary_df.to_csv(
    OUT / "unfolding_dmd_summary.csv",
    index=False
)

pd.DataFrame({
    "eigenvalue_real": eigenvalues.real,
    "eigenvalue_imag": eigenvalues.imag,
    "eigenvalue_magnitude": np.abs(eigenvalues),
}).to_csv(
    OUT / "unfolding_dmd_eigenvalues.csv",
    index=False
)


# --------------------------------------------------
# 10. Plot per-frame relative error
# --------------------------------------------------

plt.figure(figsize=(9, 5))
plt.plot(
    metrics_df["original_frame"],
    metrics_df["relative_error_percent"],
    marker="o"
)
plt.xlabel("Original target frame")
plt.ylabel("Relative error (%)")
plt.title("Unfolding DMD: Per-Frame Relative Error")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUT / "unfolding_dmd_error_plot.png",
    dpi=200
)
plt.close()


# --------------------------------------------------
# 11. Plot final target frame
# --------------------------------------------------

actual = target[:, :, -1]
predicted = Xrec[:, :, -1]
absolute_error = np.abs(predicted - actual)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

im0 = axes[0].imshow(actual, cmap="viridis")
axes[0].set_title(f"Actual frame {T}")
fig.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(predicted, cmap="viridis")
axes[1].set_title("Unfolding DMD")
fig.colorbar(im1, ax=axes[1], fraction=0.046)

im2 = axes[2].imshow(absolute_error, cmap="magma")
axes[2].set_title("Absolute error")
fig.colorbar(im2, ax=axes[2], fraction=0.046)

for ax in axes:
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

fig.suptitle("Unfolding DMD: Final Target Frame")
fig.tight_layout()
fig.savefig(
    OUT / "unfolding_dmd_frame_comparison.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close(fig)

print(f"\nAll outputs saved to: {OUT}")
print("Unfolding DMD run completed.")
