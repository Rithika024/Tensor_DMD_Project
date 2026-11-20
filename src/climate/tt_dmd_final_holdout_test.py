from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIMATE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(CLIMATE_DIR))

from tt_dmd import load_climate_tensor, DATA_PATH
from tt_dmd_temporal_validation import forecast_tt_dmd

RESULTS_DIR = PROJECT_ROOT / "results" / "climate"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    video = load_climate_tensor(DATA_PATH)

    if video.shape != (60, 60, 12):
        raise ValueError(f"Unexpected climate tensor shape: {video.shape}")

    training_video = video[:, :, :10]
    test_video = video[:, :, 10:]

    print("=" * 65)
    print("TT-DMD FINAL UNTOUCHED HOLDOUT TEST")
    print("=" * 65)
    print(f"Training frames: 1-10 {training_video.shape}")
    print(f"Test frames:     11-12 {test_video.shape}")

    configurations = {
        "Full": (60, 9),
        "Medium": (20, 7),
    }

    all_metrics = []

    for name, (rank1, rank2) in configurations.items():
        print(f"\nTesting {name}: rank1={rank1}, rank2={rank2}")

        start = time.perf_counter()

        prediction = forecast_tt_dmd(
            training_video,
            rank1,
            rank2,
            total_frames=video.shape[2],
        )

        runtime = time.perf_counter() - start

        if prediction.shape != test_video.shape:
            raise ValueError(
                f"{name}: prediction shape {prediction.shape} "
                f"does not match test shape {test_video.shape}"
            )

        np.save(
            RESULTS_DIR / f"tt_dmd_final_test_{name.lower()}.npy",
            prediction,
        )

        for i in range(test_video.shape[2]):
            actual = test_video[:, :, i]
            estimated = prediction[:, :, i]
            difference = estimated - actual

            mae = np.mean(np.abs(difference))
            rmse = np.sqrt(np.mean(difference ** 2))
            relative_error = (
                100 * np.linalg.norm(difference)
                / max(np.linalg.norm(actual), np.finfo(float).eps)
            )

            all_metrics.append({
                "configuration": name,
                "rank1": rank1,
                "rank2": rank2,
                "frame": 11 + i,
                "MAE": float(mae),
                "RMSE": float(rmse),
                "relative_error_percent": float(relative_error),
                "runtime_seconds": runtime,
            })

            print(
                f"Frame {11+i}: MAE={mae:.8f}, "
                f"RMSE={rmse:.8f}, "
                f"Relative error={relative_error:.6f}%"
            )

        print(f"Runtime: {runtime:.4f} seconds")

    metrics_df = pd.DataFrame(all_metrics)

    output_path = RESULTS_DIR / "tt_dmd_final_holdout_metrics.csv"
    metrics_df.to_csv(output_path, index=False)

    print("\n" + "=" * 65)
    print("FINAL HOLDOUT RESULTS")
    print("=" * 65)

    print(
        metrics_df[
            ["configuration", "frame", "MAE", "RMSE",
             "relative_error_percent"]
        ].to_string(index=False)
    )

    print(f"\nSaved metrics: {output_path}")
    print("\nFinal holdout test completed.")


if __name__ == "__main__":
    main()
