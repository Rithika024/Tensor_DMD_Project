
import numpy as np

# Representative ill-conditioned matrix for a numerical diagnostic.
Xi = np.array([
    [1.0, 1.0, 1.0],
    [1.0, 1.0 + 1e-12, 1.0],
    [1.0, 1.0, 1.0 + 1e-14],
], dtype=float)

U, s, Vh = np.linalg.svd(Xi, full_matrices=False)

# Current implementation's threshold.
threshold = np.finfo(float).eps * max(Xi.shape) * s[0]
custom_inverse = np.diag(
    np.where(s > threshold, 1.0 / np.maximum(s, 1e-300), 0.0)
)

# NumPy pseudoinverse for comparison.
numpy_inverse = np.linalg.pinv(np.diag(s))

print("Singular values:")
print(s)
print(f"\nCustom threshold: {threshold:.6e}")
print("Custom inverse diagonal:")
print(np.diag(custom_inverse))
print("NumPy pinv diagonal:")
print(np.diag(numpy_inverse))
print(
    "\nRelative difference:",
    np.linalg.norm(custom_inverse - numpy_inverse)
    / max(np.linalg.norm(numpy_inverse), 1e-15)
)
