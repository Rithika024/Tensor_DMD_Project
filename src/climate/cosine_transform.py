import numpy as np
from scipy.fft import dct, idct


def cosine_transform(A):
    """
    Apply a DCT along the third tensor dimension.

    This uses an orthonormal DCT-II, matching MATLAB's standard
    dct convention for real-valued input.
    """
    A = np.asarray(A)

    if A.ndim != 3:
        raise ValueError(f"Expected a 3D tensor; received {A.shape}")

    return dct(A, type=2, axis=2, norm="ortho")


def inverse_cosine_transform(A):
    """
    Invert cosine_transform() along the third tensor dimension.
    """
    A = np.asarray(A)

    if A.ndim != 3:
        raise ValueError(f"Expected a 3D tensor; received {A.shape}")

    return idct(A, type=2, axis=2, norm="ortho")