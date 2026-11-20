
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

video = np.load(ROOT / "outputs" / "synthetic_video.npy")
n, m, T = video.shape

# Reproduce the original detrending and snapshot preparation.
row_mean = np.mean(video, axis=1, keepdims=True)
video_detrended = video - row_mean

X = video_detrended[:, :, :-1]
Y = video_detrended[:, :, 1:]

X_sw = np.transpose(X, (0, 2, 1))
Y_sw = np.transpose(Y, (0, 2, 1))

fX = np.fft.fft(X_sw, axis=2)
fY = np.fft.fft(Y_sw, axis=2)

# Inspect frequency bin zero.
Xi = fX[:, :, 0]
Yi = fY[:, :, 0]

U, s, Vh = np.linalg.svd(Xi, full_matrices=False)

threshold = (
    np.finfo(float).eps * max(Xi.shape) * s[0]
)

inverse_s = np.where(
    s > threshold,
    1.0 / np.maximum(s, 1e-300),
    0.0
)

V = Vh.conj().T

A_tilde = (
    U.conj().T @ Yi @ V
) @ np.diag(inverse_s)

eigenvalues, W = np.linalg.eig(A_tilde)

omega = np.log(
    eigenvalues.astype(np.complex128)
)

tau_re = 2e-3
tau_im = 0.02

is_background = (
    (np.abs(omega.real) < tau_re)
    & (np.abs(omega.imag) < tau_im)
)

print("Frequency bin: 0")
print("Xi shape:", Xi.shape)
print("Maximum absolute value in Xi:", np.max(np.abs(Xi)))
print("Maximum absolute value in Yi:", np.max(np.abs(Yi)))
print("Singular values:")
print(s)
print("Retained singular values:", np.sum(s > threshold))
print("Eigenvalues:")
print(eigenvalues)
print("Selected as background:", is_background)
print("Number selected:", np.sum(is_background))

if not np.any(is_background):
    fallback = np.argmin(np.abs(omega))
    print("Fallback mode index:", fallback)
    print("Fallback eigenvalue:", eigenvalues[fallback])
