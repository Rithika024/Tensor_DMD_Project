
from pathlib import Path
import numpy as np
from scipy.io import loadmat

root = Path(__file__).resolve().parents[2]

original = np.asarray(
    loadmat(root / "reference" / "Tensor_DMD-main"
            / "sst_60x60xT_sub.mat")["X"],
    dtype=np.float64,
)

reconstructed = np.load(
    root / "results" / "climate"
    / "fft_tdmd_reconstruction.npy"
)

print("Original shape:", original.shape)
print("Reconstructed shape:", reconstructed.shape)
print("Reconstruction finite:", np.isfinite(reconstructed).all())

# Compare against the next snapshots, as in Climate_all.m.
target = original[:, :, 1:]
difference = reconstructed - target

print("\nNEXT-SNAPSHOT VERIFICATION")
print("Maximum absolute difference:", np.max(np.abs(difference)))
print("Overall MAE:", np.mean(np.abs(difference)))
print("Overall RMSE:", np.sqrt(np.mean(difference ** 2)))

print("\nDETAILED PER-FRAME ERRORS")

for i in range(reconstructed.shape[2]):
    frame_difference = reconstructed[:, :, i] - target[:, :, i]

    relative_error = (
        np.linalg.norm(frame_difference)
        / np.linalg.norm(target[:, :, i])
        * 100
    )

    print(
        f"Target frame {i + 2:2d}: "
        f"MAE={np.mean(np.abs(frame_difference)):.12e}, "
        f"RMSE={np.sqrt(np.mean(frame_difference**2)):.12e}, "
        f"MaxAbs={np.max(np.abs(frame_difference)):.12e}, "
        f"RelativeError={relative_error:.12e}%"
    )

# Check that the reference snapshots themselves are not identical.
print("\nORIGINAL SNAPSHOT CHANGES")
for i in range(original.shape[2] - 1):
    change = np.linalg.norm(
        original[:, :, i + 1] - original[:, :, i]
    )
    print(f"Frame {i + 1} -> {i + 2}: {change:.8f}")
