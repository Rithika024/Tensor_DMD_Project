import numpy as np
from tt_dmd import load_climate_tensor, run_tt_dmd, DATA_PATH

video = load_climate_tensor(DATA_PATH)
tt_rec, _, _, details = run_tt_dmd(video)

# The first two tuple items are Xmat and Ymat.
Xmat = details[0]
Ymat = details[1]

print("\nSnapshot matrix shapes:", Xmat.shape, Ymat.shape)

# Ordinary DMD using the same snapshot matrices.
U, s, Vh = np.linalg.svd(Xmat, full_matrices=False)
V = Vh.T
S_inv = np.linalg.pinv(np.diag(s))

A_tilde = U.T @ Ymat @ V @ S_inv
eigenvalues, W = np.linalg.eig(A_tilde)

Phi = Ymat @ V @ S_inv @ W
b = np.linalg.pinv(Phi) @ Xmat[:, 0]

n, m = video.shape[:2]
r = Xmat.shape[1]
direct_rec = np.zeros((n, m, r))

for t in range(1, r + 1):
    snapshot = Phi @ ((eigenvalues ** t) * b)
    direct_rec[:, :, t - 1] = np.real(
        snapshot.reshape((n, m), order="F")
    )

difference = tt_rec - direct_rec

print("\n=== TT-DMD vs Direct DMD ===")
print("TT reconstruction shape:", tt_rec.shape)
print("Direct reconstruction shape:", direct_rec.shape)
print("Maximum absolute difference:", np.max(np.abs(difference)))
print("Mean absolute difference:", np.mean(np.abs(difference)))
print(
    "Relative difference:",
    np.linalg.norm(difference) / np.linalg.norm(direct_rec)
)
