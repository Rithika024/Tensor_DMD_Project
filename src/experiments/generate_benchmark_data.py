
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------
# Benchmark dataset generator
# Does not modify the original synthetic video or background.
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "benchmark_v1"

HEIGHT = 60
WIDTH = 60
FRAMES = 60
SQUARE_SIZE = 8

BACKGROUND_SEED = 2026
NOISE_SEED = 100


def create_background():
    """Create a smooth, spatially varying static background."""
    rng = np.random.default_rng(BACKGROUND_SEED)

    raw = rng.random((HEIGHT, WIDTH))
    background = gaussian_filter(raw, sigma=7)

    # Normalize to a consistent intensity range.
    background -= background.min()
    background /= background.max()

    return 0.15 + 0.25 * background


def object_position(frame, scenario):
    """Return the top-left position of the moving square."""

    max_row = HEIGHT - SQUARE_SIZE
    max_col = WIDTH - SQUARE_SIZE

    if scenario == "slow":
        # Horizontal movement, one pixel per frame.
        row = 20
        col = (frame // 2) % (max_col + 1)

    elif scenario == "fast":
        # Faster horizontal movement.
        row = 20
        col = (2 * frame) % (max_col + 1)

    elif scenario == "diagonal":
        # Diagonal movement with boundary reflection.
        row = min(frame, max_row)
        col = min(frame, max_col)

    elif scenario == "reversing":
        # Move right, then reverse direction.
        period = 2 * max_col
        position = frame % period
        col = position if position <= max_col else period - position
        row = 25

    elif scenario == "long_occupancy":
        # Object stays over the same region for most frames.
        row = 20
        col = 20 if frame < 40 else (frame - 40) % (max_col + 1)

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return int(np.clip(row, 0, max_row)), int(np.clip(col, 0, max_col))


def generate_video(scenario, noise_sigma=0.0):
    """Generate video, true background, and true foreground mask."""

    background = create_background()

    video = np.repeat(
        background[:, :, np.newaxis],
        FRAMES,
        axis=2,
    )

    foreground_mask = np.zeros(
        (HEIGHT, WIDTH, FRAMES),
        dtype=bool,
    )

    for t in range(FRAMES):
        row, col = object_position(t, scenario)

        foreground_mask[
            row:row + SQUARE_SIZE,
            col:col + SQUARE_SIZE,
            t,
        ] = True

        video[
            row:row + SQUARE_SIZE,
            col:col + SQUARE_SIZE,
            t,
        ] = 0.95

    if noise_sigma > 0:
        rng = np.random.default_rng(NOISE_SEED)
        noise = rng.normal(
            loc=0.0,
            scale=noise_sigma,
            size=video.shape,
        )
        video = video + noise

    video = np.clip(video, 0.0, 1.0)

    return video, background, foreground_mask


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = [
        "slow",
        "fast",
        "diagonal",
        "reversing",
        "long_occupancy",
    ]

    noise_levels = [0.0, 0.02, 0.05]

    for scenario in scenarios:
        for sigma in noise_levels:
            video, background, mask = generate_video(
                scenario,
                noise_sigma=sigma,
            )

            filename = f"{scenario}_noise_{sigma:.2f}"

            np.save(OUTPUT_DIR / f"{filename}_video.npy", video)
            np.save(OUTPUT_DIR / f"{filename}_background.npy", background)
            np.save(OUTPUT_DIR / f"{filename}_mask.npy", mask)

            median_background = np.median(video, axis=2)
            median_mae = np.mean(
                np.abs(median_background - background)
            )

            occupancy = np.mean(mask, axis=2)

            print(
                f"{filename:30s} "
                f"shape={video.shape}, "
                f"median_MAE={median_mae:.6f}, "
                f"max_occupancy={occupancy.max():.3f}"
            )

    print("\nBenchmark dataset generation complete.")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
