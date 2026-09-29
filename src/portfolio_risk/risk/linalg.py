"""Linear algebra helpers: nearest-PSD repair and a robust Cholesky factor."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _validate_square(matrix: FloatArray) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Covariance matrix must be square, got shape {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError("Covariance matrix contains NaN or infinite values.")


def nearest_psd(matrix: FloatArray) -> FloatArray:
    """Repair a symmetric matrix so it is positive definite (eigenvalue clipping).

    Negative and tiny eigenvalues are lifted to a small positive floor, then the result is
    rescaled so the original variances (diagonal) are preserved. Already well-conditioned
    PSD matrices are returned (almost) unchanged.
    """
    _validate_square(matrix)
    sym = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    floor = 1e-10 * max(float(eigvals.max()), 0.0) + 1e-16
    repaired = (eigvecs * np.maximum(eigvals, floor)) @ eigvecs.T
    variances = np.diag(sym)
    fixed = np.diag(repaired)
    scale = np.ones_like(variances)
    positive = (variances > 0) & (fixed > 0)
    scale[positive] = np.sqrt(variances[positive] / fixed[positive])
    repaired = repaired * np.outer(scale, scale)
    return np.asarray((repaired + repaired.T) / 2.0, dtype=np.float64)


def cholesky_factor(cov: FloatArray) -> FloatArray:
    """Lower-triangular ``L`` with ``L @ L.T ~= cov``.

    Falls back to :func:`nearest_psd` when ``cov`` is not positive definite (e.g. perfectly
    correlated or indefinite estimates), so sampling never fails on a degenerate matrix.
    """
    _validate_square(cov)
    try:
        return np.asarray(np.linalg.cholesky(cov), dtype=np.float64)
    except np.linalg.LinAlgError:
        return np.asarray(np.linalg.cholesky(nearest_psd(cov)), dtype=np.float64)
