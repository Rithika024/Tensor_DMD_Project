
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def save_video(array_path, video_path, fps=5):
    video = np.load(array_path)

    # Expected shape: height x width x frames
    if video.ndim != 3:
        raise ValueError(f"Expected a 3D video array, got {video.shape}")

    print(f"Loading: {array_path}")
    print("Array shape:", video.shape)

    frames = []

    for i in range(video.shape[2]):
        frame = np.nan_to_num(video[:, :, i])
        frame = np.clip(frame, 0, 1)
        frame = (frame * 255).astype(np.uint8)
        frames.append(frame)

    imageio.mimsave(
        video_path,
        frames,
        fps=fps,
        macro_block_size=1
    )

    print(f"Saved video: {video_path}")


save_video(
    ROOT / "outputs" / "synthetic_video.npy",
    OUTPUT_DIR / "original_video.mp4"
)

save_video(
    OUTPUT_DIR / "standard_dmd_reconstruction.npy",
    OUTPUT_DIR / "standard_dmd_result.mp4"
)

save_video(
    OUTPUT_DIR / "tdmd_fft_reconstruction.npy",
    OUTPUT_DIR / "fft_dmd_result.mp4"
)
