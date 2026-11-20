
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cosine_transform import (
    cosine_transform,
    inverse_cosine_transform,
)


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "reference"
    / "Tensor_DMD-main"
    / "sst_60x60xT_sub.mat"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "climate"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MATLAB Climate_all.m uses full economy-SVD rank.
# Do not truncate the rank for this reproduction.
PINV_TOL = 1e-12


# --------------------------------------------------
# 2. Load climate dataset
# --------------------------------------------------

from scipy.io import loadmat

mat_data = loadmat(DATA_PATH)

if "X" in mat_data:
    X = np.asarray(mat_data["X"], dtype=np.float64)
else:
    candidates = [
        (key, value)
        for key, value in mat_data.items()
        if not key.startswith("__")
        and isinstance(value, np.ndarray)
        and value.ndim == 3
    ]

    if len(candidates) != 1:
        raise ValueError(
            "Could not identify the climate tensor. "
            f"3D variables found: {[key for key, _ in candidates]}"
        )

    variable_name, X = candidates[0]
    X = np.asarray(X, dtype=np.float64)
    print(f"Using MAT variable: {variable_name}")

if X.ndim != 3:
    raise ValueError(f"Expected a 3D tensor; received {X.shape}")

if not np.all(np.isfinite(X)):
    raise ValueError("Input dataset contains NaN or infinite values.")

n, m, T = X.shape

print("=" * 60)
print("COSINE-TDMD: CLIMATE DATASET")
print("=" * 60)
print(f"Input shape: {X.shape}")
print(f"Minimum: {X.min():.6f}")
print(f"Maximum: {X.max():.6f}")
print(f"Mean: {X.mean():.6f}")


# --------------------------------------------------
# 3. Form consecutive snapshot pairs
# --------------------------------------------------
# Match Climate_all.m:
# X_t = X(:,:,1:T-1)
# Y_t = X(:,:,2:T)
# X_sw = permute(X_t,[1 3 2])
# Y_sw = permute(Y_t,[1 3 2])

X_t = X[:, :, :-1]
Y_t = X[:, :, 1:]

r = T - 1
a = min(n, r)

X_sw = np.transpose(X_t, (0, 2, 1))
Y_sw = np.transpose(Y_t, (0, 2, 1))

print(f"Snapshot-pair shape: {X_t.shape}")
print(f"Transformed tensor shape: {X_sw.shape}")
print(f"Economy-SVD rank limit: {a}")


# --------------------------------------------------
# 4. Apply cosine transform along dimension 3
# --------------------------------------------------

Xc = cosine_transform(X_sw)
Yc = cosine_transform(Y_sw)

print(f"Cosine-transformed X shape: {Xc.shape}")
print(f"Cosine-transformed Y shape: {Yc.shape}")


# --------------------------------------------------
# 5. Fit DMD independently for each cosine slice
# --------------------------------------------------

Phi_c = np.zeros((n, a, m), dtype=np.complex128)
Lam_c = np.zeros((a, a, m), dtype=np.complex128)

slice_diagnostics = []

for i in range(m):
    Xi = Xc[:, :, i]
    Yi = Yc[:, :, i]

    U, singular_values, Vh = np.linalg.svd(
        Xi, full_matrices=False
    )

    # MATLAB: V from [u,s,v] = svd(...)
    V = Vh.conj().T

    # MATLAB: Atil = u' * Yi * v * pinv(s)
    S = np.diag(singular_values)
    S_pinv = np.linalg.pinv(S, rcond=PINV_TOL)

    A_tilde = U.conj().T @ Yi @ V @ S_pinv

    eigenvalues, W = np.linalg.eig(A_tilde)

    Phi = U @ W

    Phi_c[:, :, i] = Phi
    Lam_c[:, :, i] = np.diag(eigenvalues)

    numerical_rank = int(
        np.sum(
            singular_values
            > PINV_TOL * singular_values[0]
        )
    ) if singular_values.size and singular_values[0] > 0 else 0

    slice_diagnostics.append({
        "cosine_slice": i + 1,
        "numerical_rank": numerical_rank,
        "largest_singular_value": (
            singular_values[0]
            if singular_values.size else 0.0
        ),
        "smallest_singular_value": (
            singular_values[-1]
            if singular_values.size else 0.0
        ),
        "maximum_eigenvalue_magnitude": (
            np.max(np.abs(eigenvalues))
            if eigenvalues.size else 0.0
        ),
    })

print(f"Processed cosine slices: {m}")


# --------------------------------------------------
# 6. Calculate initial amplitudes
# --------------------------------------------------
# MATLAB uses X0_c = cosine_transform(X_sw(:,1,:)).
# Keep the singleton middle dimension so the input
# remains a 3D tensor.

X0_c = cosine_transform(X_sw[:, 0:1, :])

b_c = np.zeros((a, m), dtype=np.complex128)

for i in range(m):
    b_c[:, i] = (
        np.linalg.pinv(Phi_c[:, :, i], rcond=PINV_TOL)
        @ X0_c[:, 0, i]
    )


# --------------------------------------------------
# 7. Reconstruct transformed snapshots
# --------------------------------------------------
# Match MATLAB's Lam^t, for t = 1,...,r.
# The resulting reconstruction corresponds to
# original frames 2 through T.

Xt_c = np.zeros((n, r, m), dtype=np.complex128)

for t in range(1, r + 1):
    for i in range(m):
        eigenvalues = np.diag(Lam_c[:, :, i])

        Xt_c[:, t - 1, i] = (
            Phi_c[:, :, i]
            @ (eigenvalues ** t * b_c[:, i])
        )


# --------------------------------------------------
# 8. Inverse cosine transform
# --------------------------------------------------

Xrec_complex = inverse_cosine_transform(Xt_c)

# Undo the permutation to obtain (height, width, time).
Xrec_complex = np.transpose(Xrec_complex, (0, 2, 1))

max_imaginary = float(np.max(np.abs(Xrec_complex.imag)))
Xrec = np.real_if_close(Xrec_complex, tol=1000).real

if not np.all(np.isfinite(Xrec)):
    raise ValueError("Reconstruction contains NaN or infinite values.")

print(f"Reconstruction shape: {Xrec.shape}")
print(f"Maximum imaginary component: {max_imaginary:.6e}")


# --------------------------------------------------
# 9. Compare with target frames 2 through T
# --------------------------------------------------

target = X[:, :, 1:]

frame_metrics = []

for j in range(r):
    actual = target[:, :, j]
    predicted = Xrec[:, :, j]

    difference = predicted - actual

    mae = float(np.mean(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    max_abs_error = float(np.max(np.abs(difference)))

    actual_norm = np.linalg.norm(actual)
    error_norm = np.linalg.norm(difference)

    relative_error = (
        100.0 * error_norm / actual_norm
        if actual_norm > 0 else np.nan
    )

    frame_metrics.append({
        "original_frame": j + 2,
        "mae": mae,
        "rmse": rmse,
        "max_absolute_error": max_abs_error,
        "relative_error_percent": relative_error,
    })

metrics_df = pd.DataFrame(frame_metrics)

overall_difference = Xrec - target

overall_mae = float(np.mean(np.abs(overall_difference)))
overall_rmse = float(
    np.sqrt(np.mean(overall_difference ** 2))
)
overall_relative_error = float(
    100.0
    * np.linalg.norm(overall_difference)
    / np.linalg.norm(target)
)

summary_df = pd.DataFrame([{
    "method": "Cosine-TDMD",
    "input_frames": T,
    "reconstructed_frames": r,
    "mean_frame_mae": metrics_df["mae"].mean(),
    "mean_frame_rmse": metrics_df["rmse"].mean(),
    "overall_mae": overall_mae,
    "overall_rmse": overall_rmse,
    "overall_relative_error_percent": overall_relative_error,
    "maximum_imaginary_component": max_imaginary,
}])

print("\n" + "=" * 60)
print("COSINE-TDMD RESULTS")
print("=" * 60)
print(f"Overall MAE: {overall_mae:.10f}")
print(f"Overall RMSE: {overall_rmse:.10f}")
print(
    f"Overall relative error: "
    f"{overall_relative_error:.8f}%"
)
print(f"Mean frame MAE: {metrics_df['mae'].mean():.10f}")
print(f"Mean frame RMSE: {metrics_df['rmse'].mean():.10f}")

print("\nPer-frame metrics:")
print(metrics_df.to_string(index=False))


# --------------------------------------------------
# 10. Save reconstruction, metrics and diagnostics
# --------------------------------------------------

np.save(
    OUTPUT_DIR / "cosine_tdmd_reconstruction.npy",
    Xrec
)

metrics_df.to_csv(
    OUTPUT_DIR / "cosine_tdmd_frame_metrics.csv",
    index=False
)

pd.DataFrame(slice_diagnostics).to_csv(
    OUTPUT_DIR / "cosine_tdmd_slice_diagnostics.csv",
    index=False
)

summary_df.to_csv(
    OUTPUT_DIR / "cosine_tdmd_summary.csv",
    index=False
)


# --------------------------------------------------
# 11. Plot frame-wise error
# --------------------------------------------------

plt.figure(figsize=(10, 5))
plt.plot(
    metrics_df["original_frame"],
    metrics_df["relative_error_percent"],
    marker="o"
)
plt.xlabel("Original target frame")
plt.ylabel("Relative reconstruction error (%)")
plt.title("Cosine-TDMD: Frame-wise Relative Error")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "cosine_tdmd_error_plot.png",
    dpi=200
)
plt.close()


# --------------------------------------------------
# 12. Plot one actual/predicted/error comparison
# --------------------------------------------------

# Show the final reconstructed target frame.
frame_index = r - 1
actual = target[:, :, frame_index]
predicted = Xrec[:, :, frame_index]
absolute_error = np.abs(predicted - actual)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

images = [
    axes[0].imshow(actual, cmap="viridis"),
    axes[1].imshow(predicted, cmap="viridis"),
    axes[2].imshow(absolute_error, cmap="magma"),
]

axes[0].set_title(f"Actual frame {T}")
axes[1].set_title(f"Cosine-TDMD frame {T}")
axes[2].set_title("Absolute error")

for ax in axes:
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

fig.colorbar(images[0], ax=axes[:2], shrink=0.8)
fig.colorbar(images[2], ax=axes[2], shrink=0.8)

fig.suptitle("Cosine-TDMD: Final Target Frame Comparison")
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "cosine_tdmd_frame_comparison.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close(fig)


print("\nSaved outputs to:")
print(OUTPUT_DIR)
print("\nCosine-TDMD run completed.")
