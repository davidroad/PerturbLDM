import numpy as np
from scipy.spatial.distance import cdist
from scipy import sparse
from scipy.stats import energy_distance
import logging
from typing import Dict, Tuple, Optional, Union, Any
import warnings

# Setup logger for distribution metrics
logger = logging.getLogger(__name__)

# ========= Utils =========
def _to_ndarray(X):
    if sparse.issparse(X):
        return X.toarray().astype(np.float64, copy=False)
    return np.asarray(X, dtype=np.float64)

def _subsample(X, max_n=5000, rng=0):
    if X.shape[0] <= max_n or max_n is None:
        return X
    rs = np.random.RandomState(rng)
    idx = rs.choice(X.shape[0], size=max_n, replace=False)
    return X[idx]

# ========= MMD =========
def _median_heuristic_sigma_from_real_only(X, max_samples=2000, rng=0):
    """
    计算sigma仅基于真实数据X
    这确保了不同预测结果之间的MMD具有可比性

    Args:
        X: Real expression data only
        max_samples: Maximum samples for computation
        rng: Random seed

    Returns:
        sigma: Median heuristic sigma value
    """
    Xs = _subsample(X, max_samples, rng)
    m = min(1000, Xs.shape[0])
    rs = np.random.RandomState(rng)
    idx = rs.choice(Xs.shape[0], size=m, replace=False)
    D = cdist(Xs[idx], Xs[idx], metric='euclidean')
    tri = D[np.triu_indices_from(D, k=1)]
    med = np.median(tri[tri > 0])
    return med if med > 0 else np.mean(tri)

def _rbf_kernel(X, Y, sigma):
    gamma = 1.0 / (2.0 * sigma * sigma)
    D2 = cdist(X, Y, metric='sqeuclidean')
    return np.exp(-gamma * D2)

def mmd_rbf(X, Y, sigma=None, unbiased=True, subsample=None, rng=0):
    """
    Compute MMD with RBF kernel

    Args:
        X: Real data (should be used to compute sigma if sigma is None)
        Y: Predicted data
        sigma: Pre-computed sigma value. If None, will compute from X only
        unbiased: Use unbiased estimator
        subsample: Subsample size for computation
        rng: Random seed

    Returns:
        MMD value
    """
    X = _to_ndarray(X); Y = _to_ndarray(Y)
    if subsample is not None:
        X = _subsample(X, subsample, rng)
        Y = _subsample(Y, subsample, rng+1)

    # IMPORTANT: If sigma is None, compute from real data (X) only
    # This ensures consistency across different predictions
    if sigma is None:
        sigma = _median_heuristic_sigma_from_real_only(X, rng=rng)
        logger.warning(f"Computing sigma from real data only: sigma={sigma:.4f}")

    Kxx = _rbf_kernel(X, X, sigma)
    Kyy = _rbf_kernel(Y, Y, sigma)
    Kxy = _rbf_kernel(X, Y, sigma)
    n, m = X.shape[0], Y.shape[0]

    if unbiased:
        np.fill_diagonal(Kxx, 0.0)
        np.fill_diagonal(Kyy, 0.0)
        mmd2 = (Kxx.sum() / (n*(n-1))
                + Kyy.sum() / (m*(m-1))
                - 2.0 * Kxy.mean())
    else:
        mmd2 = (Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())
    return float(np.sqrt(max(mmd2, 0.0)))

# ========= E-distance =========
def compute_energy_distance(X, Y):
    """Compute Energy Distance between two distributions"""
    try:
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)

        # Subsample if too large for efficiency (fixed: check size first)
        if X.shape[0] > 2000:
            idx_x = np.random.choice(X.shape[0], 2000, replace=False)
            X = X[idx_x]
        if Y.shape[0] > 2000:
            idx_y = np.random.choice(Y.shape[0], 2000, replace=False)
            Y = Y[idx_y]

        # Flatten for energy_distance (expects 1D arrays)
        X_flat = X.flatten()
        Y_flat = Y.flatten()

        return float(energy_distance(X_flat, Y_flat))
    except Exception as e:
        logger.warning(f"Energy distance计算失败: {e}")
        return np.nan

def e_distance(X, Y, subsample=None, rng=0):
    """Legacy wrapper for backward compatibility - use compute_energy_distance for new code"""
    return compute_energy_distance(X, Y)

# ========= Sliced Wasserstein =========
def _swd_equal_n(X, Y, num_projections=128, rng=0):
    d = X.shape[1]
    rs = np.random.RandomState(rng)
    dirs = rs.normal(size=(num_projections, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    w_sum = 0.0
    for v in dirs:
        x1d = np.sort(X @ v)
        y1d = np.sort(Y @ v)
        w_sum += np.mean(np.abs(x1d - y1d))
    return w_sum / num_projections

def _ecdf(values, grid):
    values = np.sort(values)
    return np.searchsorted(values, grid, side="right") / len(values)

def _swd_cdf(X, Y, num_projections=128, grid_size=400, rng=0):
    d = X.shape[1]
    rs = np.random.RandomState(rng)
    dirs = rs.normal(size=(num_projections, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    w_sum = 0.0
    for v in dirs:
        x1d = X @ v
        y1d = Y @ v
        lo, hi = min(x1d.min(), y1d.min()), max(x1d.max(), y1d.max())
        if lo == hi:
            continue
        grid = np.linspace(lo, hi, grid_size)
        Fx = _ecdf(x1d, grid); Fy = _ecdf(y1d, grid)
        w_sum += np.trapz(np.abs(Fx - Fy), grid)
    return w_sum / num_projections

def compute_sliced_wasserstein(X, Y, num_projections=128, grid_size=400, rng=0):
    if X.shape[0] == Y.shape[0]:
        return _swd_equal_n(X, Y, num_projections=num_projections, rng=rng), "sliced_equal_n"
    else:
        return _swd_cdf(X, Y, num_projections=num_projections, grid_size=grid_size, rng=rng), "sliced_cdf"

# ========= OT Wasserstein =========
def _ot_wasserstein(X, Y, reg=None):
    try:
        import ot  # pip install pot
    except ImportError:
        logger.warning("POT package not available. Skipping OT Wasserstein computation.")
        return np.nan, "ot_unavailable"

    n, m = X.shape[0], Y.shape[0]
    a = np.ones(n) / n
    b = np.ones(m) / m
    M = cdist(X, Y, metric='euclidean')

    try:
        if reg is None:
            T = ot.emd(a, b, M)           # exact OT
        else:
            T = ot.sinkhorn(a, b, M, reg) # Sinkhorn regularized OT
        return float((T * M).sum()), ("ot_emd" if reg is None else f"ot_sinkhorn_reg={reg}")
    except Exception as e:
        logger.warning(f"OT computation failed: {e}")
        return np.nan, f"ot_error"

# ========= Unified API =========
def distributional_similarity_metrics(
    X_real,
    X_pred,
    *,
    subsample=5000,
    rng=0,
    mmd_sigma=None,
    sw_projections=128,
    sw_grid_size=400,
    ot_reg=None,
    ot_subsample=2000
):
    """
    Compute distributional similarity metrics (MMD, E-distance, both Wassersteins).
    Returns both Sliced Wasserstein and OT Wasserstein by default.
    """
    Xr = _to_ndarray(X_real); Xp = _to_ndarray(X_pred)
    if subsample is not None:
        Xr = _subsample(Xr, subsample, rng)
        Xp = _subsample(Xp, subsample, rng+1)

    # MMD & E-distance
    mmd_val = mmd_rbf(Xr, Xp, sigma=mmd_sigma, unbiased=True, subsample=None, rng=rng)
    e_val   = e_distance(Xr, Xp, subsample=None, rng=rng)

    # Sliced Wasserstein
    sw_val, sw_mode = compute_sliced_wasserstein(Xr, Xp, num_projections=sw_projections, grid_size=sw_grid_size, rng=rng)

    # OT Wasserstein (with optional downsample for cost control)
    Xr_ot = _subsample(Xr, ot_subsample, rng)
    Xp_ot = _subsample(Xp, ot_subsample, rng+1)
    ot_val, ot_mode = _ot_wasserstein(Xr_ot, Xp_ot, reg=ot_reg)

    return {
        "MMD_RBF": mmd_val,
        "E_distance": e_val,
        "Wasserstein_Sliced": sw_val,
        "Wasserstein_Sliced_type": sw_mode,
        "Wasserstein_OT": ot_val,
        "Wasserstein_OT_type": ot_mode,
        "n_real": int(Xr.shape[0]),
        "n_pred": int(Xp.shape[0])
    }

# ========= Plugin Interface for Condition-wise Analysis =========
def compute_condition_distribution_metrics(
    real_expr: np.ndarray,
    pred_expr: np.ndarray,
    condition_name: str,
    cell_line: str = None,
    drug: str = None,
    dose: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Plugin interface for computing distribution metrics for a single condition.

    Args:
        real_expr: Real expression data (cells x genes)
        pred_expr: Predicted expression data (cells x genes)
        condition_name: Name of the condition
        cell_line: Cell line name (optional)
        drug: Drug name (optional)
        dose: Dose value (optional)
        **kwargs: Additional parameters for distributional_similarity_metrics

    Returns:
        Dict with distribution metrics and metadata
    """
    try:
        logger.info(f"Computing distribution metrics for condition: {condition_name}")

        # Validate inputs
        if real_expr.shape[1] != pred_expr.shape[1]:
            raise ValueError(f"Gene dimension mismatch: real={real_expr.shape[1]}, pred={pred_expr.shape[1]}")

        # Check for valid data
        if real_expr.shape[0] == 0 or pred_expr.shape[0] == 0:
            logger.warning(f"Empty data for condition {condition_name}")
            return _create_empty_metrics_result(condition_name, cell_line, drug, dose)

        # Remove invalid values
        real_valid = np.isfinite(real_expr).all(axis=1)
        pred_valid = np.isfinite(pred_expr).all(axis=1)

        real_clean = real_expr[real_valid]
        pred_clean = pred_expr[pred_valid]

        if real_clean.shape[0] == 0 or pred_clean.shape[0] == 0:
            logger.warning(f"No valid data after cleaning for condition {condition_name}")
            return _create_empty_metrics_result(condition_name, cell_line, drug, dose)

        # Compute distribution metrics
        dist_metrics = distributional_similarity_metrics(real_clean, pred_clean, **kwargs)

        # Add metadata
        result = {
            "condition": condition_name,
            "cell_line": cell_line,
            "drug": drug,
            "dose": dose,
            "n_real_cells": int(real_clean.shape[0]),
            "n_pred_cells": int(pred_clean.shape[0]),
            "n_genes": int(real_clean.shape[1]),
            "n_real_cells_original": int(real_expr.shape[0]),
            "n_pred_cells_original": int(pred_expr.shape[0]),
            "status": "success",
            **dist_metrics
        }

        logger.info(f"Condition {condition_name}: MMD={dist_metrics['MMD_RBF']:.4f}, "
                   f"E-dist={dist_metrics['E_distance']:.4f}, SW={dist_metrics['Wasserstein_Sliced']:.4f}")

        return result

    except Exception as e:
        logger.error(f"Error computing metrics for condition {condition_name}: {e}")
        return _create_error_metrics_result(condition_name, cell_line, drug, dose, str(e))

def _create_empty_metrics_result(condition_name: str, cell_line: str, drug: str, dose: str) -> Dict[str, Any]:
    """Create empty metrics result for invalid data"""
    return {
        "condition": condition_name,
        "cell_line": cell_line,
        "drug": drug,
        "dose": dose,
        "n_real_cells": 0,
        "n_pred_cells": 0,
        "n_genes": 0,
        "n_real_cells_original": 0,
        "n_pred_cells_original": 0,
        "status": "empty_data",
        "MMD_RBF": np.nan,
        "E_distance": np.nan,
        "Wasserstein_Sliced": np.nan,
        "Wasserstein_Sliced_type": "none",
        "Wasserstein_OT": np.nan,
        "Wasserstein_OT_type": "none",
        "n_real": 0,
        "n_pred": 0
    }

def _create_error_metrics_result(condition_name: str, cell_line: str, drug: str, dose: str, error_msg: str) -> Dict[str, Any]:
    """Create error metrics result"""
    return {
        "condition": condition_name,
        "cell_line": cell_line,
        "drug": drug,
        "dose": dose,
        "n_real_cells": 0,
        "n_pred_cells": 0,
        "n_genes": 0,
        "n_real_cells_original": 0,
        "n_pred_cells_original": 0,
        "status": f"error: {error_msg}",
        "MMD_RBF": np.nan,
        "E_distance": np.nan,
        "Wasserstein_Sliced": np.nan,
        "Wasserstein_Sliced_type": "error",
        "Wasserstein_OT": np.nan,
        "Wasserstein_OT_type": "error",
        "n_real": 0,
        "n_pred": 0
    }
