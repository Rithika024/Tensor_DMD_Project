

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
video = np.load(ROOT / "outputs" / "synthetic_video.npy")

row_mean = video.mean(axis=1, keepdims=True)
vd = video - row_mean

X = np.fft.fft(vd[:, :, :-1].transpose(0, 2, 1), axis=2)
Y = np.fft.fft(vd[:, :, 1:].transpose(0, 2, 1), axis=2)

eps = np.finfo(float).eps
results = []

for i in range(X.shape[2]):
    Xi, Yi = X[:, :, i], Y[:, :, i]
    U, s, Vh = np.linalg.svd(Xi, full_matrices=False)
    V = Vh.conj().T

    # Current implementation
    inv_custom = 1.0 / np.maximum(s, 1e-14)
    A_custom = (U.conj().T @ Yi @ V) @ np.diag(inv_custom)

    # Pseudoinverse implementation
    A_pinv = (U.conj().T @ Yi @ V) @ np.linalg.pinv(np.diag(s))

    rel = np.linalg.norm(A_custom - A_pinv) / max(
        np.linalg.norm(A_pinv), 1e-15
    )

    results.append({
        "slice": i,
        "largest_sv": s[0],
        "smallest_sv": s[-1],
        "custom_rank": int(np.sum(s > 1e-14)),
        "pinv_rank": int(np.linalg.matrix_rank(np.diag(s))),
        "custom_norm": np.linalg.norm(A_custom),
        "pinv_norm": np.linalg.norm(A_pinv),
        "relative_difference": rel,
    })

print("Top 10 slices by relative operator difference:")
for item in sorted(
    results, key=lambda x: x["relative_difference"], reverse=True
)[:10]:
    print(item)
