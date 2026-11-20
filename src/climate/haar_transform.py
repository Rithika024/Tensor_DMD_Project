from pathlib import Path

import numpy as np


def haar_blocks(A):
    """
    Python equivalent of fast_bdwt_haar_blocks.m.

    Input:
        A: 3D tensor (n, m, r), with even r.
    Output:
        D: transformed tensor with the MATLAB reference's block ordering.

    Note:
        The MATLAB source sets isqrt2 = 1, so this implementation
        deliberately matches that source rather than normalizing by sqrt(2).
    """
    A = np.asarray(A)

    if A.ndim != 3:
        raise ValueError(f"Expected a 3D tensor; received {A.shape}")

    n, m, r = A.shape

    if r % 2 != 0:
        raise ValueError("The third dimension must be even.")

    if r % 4 != 0 and r != 2:
        raise ValueError(
            "The reference block reordering requires r divisible by 4, "
            "except when r = 2."
        )

    tmp = np.zeros_like(A)

    # Match MATLAB: isqrt2 = 1
    for p in range(r // 2):
        X = A[:, :, 2 * p]
        Y = A[:, :, 2 * p + 1]

        tmp[:, :, 2 * p] = X + Y
        tmp[:, :, 2 * p + 1] = X - Y

    if r == 2:
        return tmp

    D = np.zeros_like(tmp)

    # Match the block reordering in the MATLAB reference.
    for g in range(0, r, 4):
        D[:, :, g] = tmp[:, :, g]
        D[:, :, g + 1] = tmp[:, :, g + 2]
        D[:, :, g + 2] = tmp[:, :, g + 1]
        D[:, :, g + 3] = tmp[:, :, g + 3]

    return D


def inverse_haar_blocks(D):
    """
    Inverse of haar_blocks(), matching its block ordering
    and the unnormalized sum/difference transform.
    """
    D = np.asarray(D)

    if D.ndim != 3:
        raise ValueError(f"Expected a 3D tensor; received {D.shape}")

    n, m, r = D.shape

    if r % 2 != 0:
        raise ValueError("The third dimension must be even.")

    if r % 4 != 0 and r != 2:
        raise ValueError("The third dimension must be divisible by 4.")

    # Undo the forward block reordering.
    if r == 2:
        tmp = D.copy()
    else:
        tmp = np.zeros_like(D)

        for g in range(0, r, 4):
            tmp[:, :, g] = D[:, :, g]
            tmp[:, :, g + 1] = D[:, :, g + 2]
            tmp[:, :, g + 2] = D[:, :, g + 1]
            tmp[:, :, g + 3] = D[:, :, g + 3]

    # Invert the unnormalized sum/difference pairing.
    A = np.zeros_like(tmp)

    for p in range(r // 2):
        summed = tmp[:, :, 2 * p]
        differenced = tmp[:, :, 2 * p + 1]

        A[:, :, 2 * p] = (summed + differenced) / 2
        A[:, :, 2 * p + 1] = (summed - differenced) / 2

    return A