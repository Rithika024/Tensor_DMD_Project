
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import functions from the original, verified TT-DMD implementation.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tt_dmd import load_climate_tensor, DATA_PATH


RESULTS_DIR = PROJECT_ROOT / "results" / "climate"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def truncated_tt_svd(tensor, rank1, rank2):
    """
    TT-SVD with manually selected truncation ranks.

    G1: (1, n, rank1)
    G2: (rank1, m, rank2)
    G3: (rank2, T)
    """
    n, m, T = tensor.shape

    # First unfolding
    first_matrix = tensor.reshape((n, m * T), order="F")

    U1, s1, Vh1 = np.linalg.svd(
        first_matrix, full_matrices=False
    )

    rank1 = min(rank1, len(s1))

    U1 = U1[:, :rank1]
    s1 = s1[:rank1]
    Vh1 = Vh1[:rank1, :]

    G1 = U1.reshape((1, n, rank1), order="F")

    remainder = (
        s1[:, None] * Vh1
    ).reshape((rank1, m, T), order="F")

    # Second unfolding
    second_matrix = remainder.reshape(
        (rank1 * m, T), order="F"
    )

    U2, s2, Vh2 = np.linalg.svd(
        second_matrix, full_matrices=False
    )

    rank2 = min(rank2, len(s2))

    U2 = U2[:, :rank2]
    s2 = s2[:rank2]
    Vh2 = Vh2[:rank2, :]

    G2 = U2.reshape((rank1, m, rank2), order="F")
    G3 = s2[:, None] * Vh2

    return G1, G2, G3


def contract_spatial_cores(G1, G2):
    _, n, rank1 = G1.shape
    _, m, rank2 = G2.shape

    first = G1[0, :, :]
    second = G2.reshape(
        (rank1, m * rank2), order="F"
    )

    combined = first @ second

    return combined.reshape(
        (n * m, rank2), order="F"
    )


def run_rank_tt_dmd(video, rank1, rank2):
    n, m, T = video.shape
    r = T - 1

    X_t = video[:, :, :-1]
    Y_t = video[:, :, 1:]

    X_sw = np.transpose(X_t, (0, 2, 1))
    Y_sw = np.transpose(Y_t, (0, 2, 1))

    # Use the same block-circulant convention as the baseline.
    from tt_dmd import block_circulant

    Xb = block_circulant(X_sw)
    Yb = block_circulant(Y_sw)

    Xmat = Xb[:, :r]
    Ymat = Yb[:, :r]

    G1, G2, G3 = truncated_tt_svd(
        X_t, rank1, rank2
    )

    U = contract_spatial_cores(G1, G2)

    # Reduced SVD of the temporal core
    Us, singular_values, Vh = np.linalg.svd(
        G3, full_matrices=False
    )

    Vs = Vh.T
    S_tt = np.diag(singular_values)

    Ueff = U @ Us
    Veff = Vs

    S_pinv = np.linalg.pinv(S_tt)

    Atil = Ueff.T @ (Ymat @ Veff) @ S_pinv

    eigenvalues, W = np.linalg.eig(Atil)

    Phi = (Ymat @ Veff) @ S_pinv @ W

    b = np.linalg.pinv(Phi) @ Xmat[:, 0]

    reconstruction = np.zeros(
        (n, m, r), dtype=np.float64
    )

    for t in range(1, r + 1):
        dynamics = np.power(eigenvalues, t) * b
        x_t = Phi @ dynamics

        reconstruction[:, :, t - 1] = np.real(
            x_t.reshape((n, m), order="F")
        )

    return reconstruction


def calculate_metrics(reconstruction, target):
    difference = reconstruction - target

    mae = float(np.mean(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference ** 2)))

    relative_error = (
        100 * np.linalg.norm(difference)
        / np.linalg.norm(target)
    )

    return mae, rmse, relative_error


def main():
    print("=" * 65)
    print("TT-DMD RANK SELECTION EXPERIMENT")
    print("=" * 65)

    video = load_climate_tensor(DATA_PATH)

    n, m, T = video.shape
    target = video[:, :, 1:]

    # Candidate ranks for this dataset.
    rank_pairs = [
        (5, 5),
        (10, 5),
        (10, 10),
        (20, 5),
        (20, 10),
        (20, 11),
        (30, 5),
        (30, 10),
        (30, 11),
        (60, 11),
    ]

    rows = []

    for rank1, rank2 in rank_pairs:
        start = time.perf_counter()

        print(
            f"\nTesting rank1={rank1}, rank2={rank2}"
        )

        try:
            reconstruction = run_rank_tt_dmd(
                video, rank1, rank2
            )

            mae, rmse, relative_error = calculate_metrics(
                reconstruction, target
            )

            runtime = time.perf_counter() - start

            rows.append({
                "rank1": rank1,
                "rank2": rank2,
                "MAE": mae,
                "RMSE": rmse,
                "relative_error_percent": relative_error,
                "runtime_seconds": runtime,
                "status": "success",
            })

            print(f"MAE: {mae:.8f}")
            print(f"RMSE: {rmse:.8f}")
            print(f"Relative error: {relative_error:.6f}%")

        except np.linalg.LinAlgError as error:
            print(f"Numerical failure: {error}")

            rows.append({
                "rank1": rank1,
                "rank2": rank2,
                "MAE": np.nan,
                "RMSE": np.nan,
                "relative_error_percent": np.nan,
                "runtime_seconds": time.perf_counter() - start,
                "status": str(error),
            })

    results = pd.DataFrame(rows)

    output_csv = (
        RESULTS_DIR / "tt_dmd_rank_selection.csv"
    )

    results.to_csv(output_csv, index=False)

    successful = results[
        results["status"] == "success"
    ]

    if successful.empty:
        raise RuntimeError(
            "No rank configuration completed successfully."
        )

    best = successful.loc[
        successful["relative_error_percent"].idxmin()
    ]

    print("\n" + "=" * 65)
    print("RANK SELECTION RESULTS")
    print("=" * 65)

    print(results.to_string(index=False))

    print("\nBest rank configuration on this evaluation dataset:")
    print(f"rank1 = {int(best['rank1'])}")
    print(f"rank2 = {int(best['rank2'])}")
    print(
        f"Relative error = "
        f"{best['relative_error_percent']:.6f}%"
    )

    # Plot the relative errors.
    plt.figure(figsize=(10, 6))

    labels = [
        f"({int(row.rank1)}, {int(row.rank2)})"
        for row in successful.itertuples()
    ]

    plt.bar(
        labels,
        successful["relative_error_percent"]
    )

    plt.xlabel("(rank1, rank2)")
    plt.ylabel("Relative reconstruction error (%)")
    plt.title("TT-DMD Rank Selection Experiment")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plot_path = (
        RESULTS_DIR / "tt_dmd_rank_selection.png"
    )

    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("\nSaved:")
    print(output_csv)
    print(plot_path)


if __name__ == "__main__":
    main()
