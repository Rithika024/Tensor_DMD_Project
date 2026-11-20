import numpy as np

from cosine_transform import (
    cosine_transform,
    inverse_cosine_transform,
)


rng = np.random.default_rng(42)
A = rng.standard_normal((5, 11, 60))

C = cosine_transform(A)
A_recovered = inverse_cosine_transform(C)

print("Input shape:", A.shape)
print("Transformed shape:", C.shape)
print("Recovered shape:", A_recovered.shape)

max_error = np.max(np.abs(A - A_recovered))
print("Maximum round-trip error:", max_error)

assert C.shape == A.shape
assert A_recovered.shape == A.shape
assert np.allclose(A, A_recovered, atol=1e-12)

print("Cosine transform tests passed.")