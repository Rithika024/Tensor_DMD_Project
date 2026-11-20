import numpy as np

from haar_transform import haar_blocks, inverse_haar_blocks


rng = np.random.default_rng(42)
A = rng.standard_normal((5, 3, 60))

D = haar_blocks(A)
A_recovered = inverse_haar_blocks(D)

print("Input shape:", A.shape)
print("Transformed shape:", D.shape)
print("Recovered shape:", A_recovered.shape)
print("Maximum round-trip error:",
      np.max(np.abs(A - A_recovered)))

assert D.shape == A.shape
assert A_recovered.shape == A.shape
assert np.allclose(A, A_recovered, atol=1e-12)

print("Haar transform tests passed.")