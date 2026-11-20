
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


# ------------------------------------------------------------
# 1. Paths and settings
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = (
    PROJECT_ROOT
    / "reference"
    / "Tensor_DMD-main"
    / "sst_60x60xT_sub.mat"
)
RESULTS_DIR = PROJECT_ROOT / "results" / "climate"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Load the climate tensor
# ------------------------------------------------------------
def load_climate_tensor(path):
    mat_data = loadmat(path)

    candidates = {
        key: value
        for key, value in mat_data.items()
        if not key.startswith("__")
        and isinstance(value, np.ndarray)
        and value.ndim == 3
        and np.issubdtype(value.dtype, np.number)
    }

    if not candidates:
        raise ValueError(
            "No numeric 3D tensor found in the MAT file. "
            f"Available variables: {list(mat_data.keys())}"
        )

    # Prefer the expected 60 x 60 x 12 climate tensor.
    for name, value in candidates.items():
        if value.shape == (60, 60, 12):
            print(f"Loaded MATLAB variable: {name}")
            return np.asarray(value, dtype=np.float64)

    # If the dimensions differ, display candidates rather than
    # silently choosing an unexpected tensor.
    print("3D variables found:")
    for name, value in candidates.items():
        print(f"  {name}: {value.shape}")

    raise ValueError(
        "Expected a tensor with shape (60, 60, 12)."
    )


# ------------------------------------------------------------
# 3. Block-circulant matrix, following bcirc.m
# ------------------------------------------------------------
def block_circulant(tensor):
    """
    tensor has shape (n, r, m).
    Returns a block-circulant matrix with shape (n*m, r*m).

    The block ordering follows the convention used by the
    existing Python unfolding-DMD implementation. Verify its
    equivalence to MATLAB gallery('circul', ...) if exact
    element-by-element MATLAB reproduction is required.
    """
    n, r, m = tensor.shape
    result = np.zeros((n * m, r * m), dtype=tensor.dtype)

    for block_row in range(m):
        for block_col in range(m):
            slice_index = (block_row - block_col) % m
            result[
                block_row * n:(block_row + 1) * n,
                block_col * r:(block_col + 1) * r,
            ] = tensor[:, :, slice_index]

    return result


# ------------------------------------------------------------
# 4. Three-core TT-SVD
# ------------------------------------------------------------
def tt_svd_3d(tensor):
    """
    Factor a 3D tensor into TT cores:
      G1: (1, n, rank1)
      G2: (rank1, m, rank2)
      G3: (rank2, T)

    Uses full numerical ranks (no deliberate truncation).
    MATLAB's eps_tt=1e-31 is below normal double-precision
    resolution, so this is treated as an essentially exact
    decomposition, subject to floating-point arithmetic.
    """
    n, m, T = tensor.shape

    # First unfolding, using MATLAB-compatible column-major order.
    first_matrix = tensor.reshape((n, m * T), order="F")
    U1, s1, Vh1 = np.linalg.svd(
        first_matrix, full_matrices=False
    )
    rank1 = len(s1)

    G1 = U1.reshape((1, n, rank1), order="F")
    remainder = (
        s1[:, None] * Vh1
    ).reshape((rank1, m, T), order="F")

    # Second unfolding splits the remaining m and T modes.
    second_matrix = remainder.reshape(
        (rank1 * m, T), order="F"
    )
    U2, s2, Vh2 = np.linalg.svd(
        second_matrix, full_matrices=False
    )
    rank2 = len(s2)

    G2 = U2.reshape((rank1, m, rank2), order="F")
    G3 = s2[:, None] * Vh2

    return G1, G2, G3


def contract_spatial_cores(G1, G2):
    """Contract the first two TT cores into an (n*m) x rank2 basis."""
    _, n, rank1 = G1.shape
    _, m, rank2 = G2.shape

    first = G1[0, :, :]  # n x rank1
    second = G2.reshape((rank1, m * rank2), order="F")

    combined = first @ second
    return combined.reshape((n * m, rank2), order="F")


def reconstruct_tt_tensor(G1, G2, G3):
    """Reconstruct the original 3D tensor from the three TT cores."""
    spatial = contract_spatial_cores(G1, G2)
    n = G1.shape[1]
    m = G2.shape[1]
    T = G3.shape[1]

    flat = spatial @ G3
    return flat.reshape((n, m, T), order="F")


# ------------------------------------------------------------
# 5. TT-DMD method following Climate_all.m
# ------------------------------------------------------------
def run_tt_dmd(video):
    n, m, T = video.shape
    r = T - 1

    # MATLAB:
    # X_t = X(:,:,1:end-1)
    # Y_t = X(:,:,2:end)
    X_t = video[:, :, :-1]
    Y_t = video[:, :, 1:]

    # MATLAB:
    # X_sw = permute(X_t,[1,3,2])
    # Y_sw = permute(Y_t,[1,3,2])
    X_sw = np.transpose(X_t, (0, 2, 1))
    Y_sw = np.transpose(Y_t, (0, 2, 1))

    # Unfolding-DMD matrices used by the reference's TT-DMD block.
    Xb = block_circulant(X_sw)
    Yb = block_circulant(Y_sw)

    Xmat = Xb[:, :r]
    Ymat = Yb[:, :r]

    # Tensor-train decomposition of X_t.
    G1, G2, G3 = tt_svd_3d(X_t)

        # Diagnostic: verify TT representation against DMD input.
    U_check = contract_spatial_cores(G1, G2)
    X_from_tt = U_check @ G3

    tt_vs_Xmat_error = (
        np.linalg.norm(X_from_tt - Xmat)
        / max(np.linalg.norm(Xmat), np.finfo(float).eps)
    )

    expected_Xmat = X_t.reshape((n * m, r), order="F")
    Xmat_order_error = (
        np.linalg.norm(Xmat - expected_Xmat)
        / max(np.linalg.norm(expected_Xmat), np.finfo(float).eps)
    )

    spatial_orthogonality_error = np.linalg.norm(
        U_check.T @ U_check - np.eye(U_check.shape[1])
    )

    print("\nTT-DMD consistency diagnostics")
    print(f"TT representation vs Xmat: {tt_vs_Xmat_error:.6e}")
    print(f"Xmat ordering vs original snapshots: {Xmat_order_error:.6e}")
    print(f"Spatial basis orthogonality error: {spatial_orthogonality_error:.6e}")

    # Verify the TT factorization before using it.
    X_tt = reconstruct_tt_tensor(G1, G2, G3)
    tt_roundtrip_error = np.linalg.norm(X_t - X_tt) / max(
        np.linalg.norm(X_t), np.finfo(float).eps
    )

    # The first two cores form the spatial basis U.
    U = contract_spatial_cores(G1, G2)

    # MATLAB: Xtime = squeeze(G3)
    Xtime = G3

    # MATLAB: [Us,S_tt,Vs] = svd(Xtime,'econ')
    Us, singular_values, Vh = np.linalg.svd(
        Xtime, full_matrices=False
    )
    Vs = Vh.T

    S_tt = np.diag(singular_values)

    Ueff = U @ Us
    Veff = Vs

    # MATLAB:
    # Atil_ttd = (Ueff') * (Ymat * Veff) / S_tt
    S_pinv = np.linalg.pinv(S_tt)
    Atil = Ueff.T @ (Ymat @ Veff) @ S_pinv

    eigenvalues, W = np.linalg.eig(Atil)

    # MATLAB:
    # Phi_ttd = (Ymat * Veff) / S_tt * W
    Phi = (Ymat @ Veff) @ S_pinv @ W

    # MATLAB:
    # b_ttd = pinv(Phi_ttd) * Xmat(:,1)
    b = np.linalg.pinv(Phi) @ Xmat[:, 0]

    # MATLAB reconstructs frames using powers D^t for t = 1...r.
    reconstruction = np.zeros(
        (n, m, r), dtype=np.float64
    )

    for t in range(1, r + 1):
        dynamics = np.power(eigenvalues, t) * b
        x_t = Phi @ dynamics

        # MATLAB reshape(...,[n,m]) uses column-major order.
        reconstruction[:, :, t - 1] = np.real(
            x_t.reshape((n, m), order="F")
        )

    return (
        reconstruction,
        eigenvalues,
        tt_roundtrip_error,
        (Xmat, Ymat, G1, G2, G3),
    )


# ------------------------------------------------------------
# 6. Metrics and output files
# ------------------------------------------------------------
def calculate_metrics(reconstruction, target):
    rows = []

    for frame in range(target.shape[2]):
        truth = target[:, :, frame]
        estimate = reconstruction[:, :, frame]
        difference = estimate - truth

        mae = float(np.mean(np.abs(difference)))
        rmse = float(np.sqrt(np.mean(difference ** 2)))
        denominator = np.linalg.norm(truth)
        relative_error = (
            100.0 * np.linalg.norm(difference) / denominator
            if denominator > 0
            else np.nan
        )

        rows.append({
            "frame": frame + 2,
            "MAE": mae,
            "RMSE": rmse,
            "relative_error_percent": relative_error,
            "max_absolute_error": float(
                np.max(np.abs(difference))
            ),
        })

    return pd.DataFrame(rows)


def main():
    start = time.perf_counter()

    print("=" * 60)
    print("TT-DMD — climate dataset")
    print("=" * 60)

    video = load_climate_tensor(DATA_PATH)

    print(f"Input shape: {video.shape}")
    print(
        f"Input range: {video.min():.6f} to "
        f"{video.max():.6f}"
    )

    reconstruction, eigenvalues, tt_error, details = (
        run_tt_dmd(video)
    )

    Xmat, Ymat, G1, G2, G3 = details
    target = video[:, :, 1:]

    print("\nTT decomposition checks")
    print(f"G1 shape: {G1.shape}")
    print(f"G2 shape: {G2.shape}")
    print(f"G3 shape: {G3.shape}")
    print(f"Xmat shape: {Xmat.shape}")
    print(f"Ymat shape: {Ymat.shape}")
    print(f"TT relative round-trip error: {tt_error:.6e}")

    print("\nTT-DMD reconstruction")
    print(f"Reconstruction shape: {reconstruction.shape}")

    metrics = calculate_metrics(reconstruction, target)
    metrics_path = RESULTS_DIR / "tt_dmd_frame_metrics.csv"
    reconstruction_path = (
        RESULTS_DIR / "tt_dmd_reconstruction.npy"
    )
    eigenvalues_path = (
        RESULTS_DIR / "tt_dmd_eigenvalues.csv"
    )
    summary_path = RESULTS_DIR / "tt_dmd_summary.csv"
    plot_path = RESULTS_DIR / "tt_dmd_error_plot.png"
    comparison_path = (
        RESULTS_DIR / "tt_dmd_frame_comparison.png"
    )

    np.save(reconstruction_path, reconstruction)
    metrics.to_csv(metrics_path, index=False)

    pd.DataFrame({
        "real": np.real(eigenvalues),
        "imag": np.imag(eigenvalues),
        "magnitude": np.abs(eigenvalues),
    }).to_csv(eigenvalues_path, index=False)

    summary = pd.DataFrame([{
        "method": "TT-DMD (Python TT-SVD)",
        "MAE": metrics["MAE"].mean(),
        "RMSE": metrics["RMSE"].mean(),
        "mean_relative_error_percent":
            metrics["relative_error_percent"].mean(),
        "max_frame_relative_error_percent":
            metrics["relative_error_percent"].max(),
        "TT_roundtrip_relative_error": tt_error,
        "runtime_seconds": time.perf_counter() - start,
    }])
    summary.to_csv(summary_path, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(
        metrics["frame"],
        metrics["relative_error_percent"],
        marker="o",
    )
    plt.xlabel("Target frame")
    plt.ylabel("Relative error (%)")
    plt.title("TT-DMD: frame-wise relative reconstruction error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Compare one representative target frame.
    frame_index = min(9, target.shape[2] - 1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    images = [
        (target[:, :, frame_index], "Reference target"),
        (reconstruction[:, :, frame_index], "TT-DMD reconstruction"),
        (
            np.abs(
                target[:, :, frame_index]
                - reconstruction[:, :, frame_index]
            ),
            "Absolute error",
        ),
    ]

    for ax, (image, title) in zip(axes, images):
        im = ax.imshow(image, aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(comparison_path, dpi=150)
    plt.close(fig)

    print("\nSummary metrics")
    print(summary.to_string(index=False))

    print("\nFrame-wise metrics")
    print(metrics.to_string(index=False))

    print("\nSaved outputs:")
    for path in [
        reconstruction_path,
        metrics_path,
        eigenvalues_path,
        summary_path,
        plot_path,
        comparison_path,
    ]:
        print(f"  {path}")

    print("\nTT-DMD script completed.")


if __name__ == "__main__":
    main()
