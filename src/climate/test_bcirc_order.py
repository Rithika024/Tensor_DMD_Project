import numpy as np
from scipy.linalg import circulant

def python_bcirc(A):
    n1, n2, n3 = A.shape
    B = np.zeros((n1*n3, n2*n3))
    for i in range(n3):
        for j in range(n3):
            B[i*n1:(i+1)*n1, j*n2:(j+1)*n2] = A[:, :, (i-j) % n3]
    return B

A = np.zeros((1, 1, 3))
A[0, 0, :] = [10, 20, 30]

print("Input slices:", A.ravel())
print("\nPython block-circulant matrix:")
print(python_bcirc(A))

print("\nCandidate MATLAB gallery conventions:")
v = np.eye(3)[0, :]
print("circulant(v):")
print(circulant(v))
print("circulant(v).T:")
print(circulant(v).T)
