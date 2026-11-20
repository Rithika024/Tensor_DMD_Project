from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = ROOT / "results" / "dmd_projection_investigation"
OUTPUT_DIR = ROOT / "results" / "dmd_prediction_visualization"

ACTUAL_PATH = RESULTS_DIR / "actual_frame_20.npy"
PROJECTION_PATH = RESULTS_DIR / "spatial_projection_frame_20.npy"
PREDICTION_PATH = RESULTS_DIR / "dmd_prediction_frame_20.npy"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    actual = np.load(ACTUAL_PATH)
    projection = np.load(PROJECTION_PATH)
    prediction = np.load(PREDICTION_PATH)

    prediction_error = np.abs(actual - prediction)
    projection_error = np.abs(actual - projection)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    images = [
        (actual, "Actual Frame 20"),
        (prediction, "DMD Prediction"),
        (prediction_error, "Absolute DMD Error"),
        (projection, "Spatial Projection"),
        (projection_error, "Absolute Projection Error"),
        (np.abs(prediction - projection), "DMD vs Projection"),
    ]

    for ax, (data, title) in zip(axes.flat, images):
        im = ax.imshow(data, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Standard DMD: Frame 20 Prediction Error Analysis",
        fontsize=15,
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "frame20_error_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("\nVisualization saved to:")
    print(output_path)


if __name__ == "__main__":
    main()