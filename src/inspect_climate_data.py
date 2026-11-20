

from pathlib import Path
import numpy as np
from scipy.io import loadmat

root = Path(__file__).resolve().parent.parent
mat_path = (
    root / "reference" / "Tensor_DMD-main"
    / "sst_60x60xT_sub.mat"
)

data = loadmat(mat_path)
X = np.asarray(data["X"], dtype=np.float64)

print("Shape:", X.shape)
print("Minimum:", np.min(X))
print("Maximum:", np.max(X))
print("Mean:", np.mean(X))
print("Standard deviation:", np.std(X))
print("NaN values:", np.isnan(X).sum())
print("Infinite values:", np.isinf(X).sum())

for t in range(X.shape[2]):
    print(
        f"Snapshot {t + 1:02d}: "
        f"min={X[:, :, t].min():.6f}, "
        f"max={X[:, :, t].max():.6f}, "
        f"mean={X[:, :, t].mean():.6f}"
    )
