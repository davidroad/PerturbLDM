import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from scipy.stats import rankdata


def _to_numpy_1d(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x).reshape(-1)


def chatterjee_coefficient(x, y):
    """
    Compute Chatterjee's rank correlation coefficient xi(X, Y).

    The coefficient is directional: Y is ranked after sorting by X. For
    prediction evaluation, call this as ``chatterjee_coefficient(y_true,
    y_pred)`` so that the predicted profile is assessed as a function of the
    observed profile.
    """
    x = _to_numpy_1d(x)
    y = _to_numpy_1d(y)
    n = len(x)
    if n < 2:
        return np.nan  # not defined for less than 2 points
    sorted_idx = np.argsort(x)
    y_sorted = y[sorted_idx]
    y_ranks = rankdata(y_sorted, method='average')
    diff_sum = np.sum(np.abs(np.diff(y_ranks)))
    xi = 1 - (3 * diff_sum) / (n**2 - 1)
    return xi
    
def compute_metrics_single(y_true, y_pred):
    """
    Compute single-profile prediction metrics with explicit sklearn order.

    Parameters
    ----------
    y_true
        Observed / ground-truth expression vector.
    y_pred
        Predicted expression vector.

    Notes
    -----
    ``r2_score`` and Chatterjee's coefficient are directional. Keep this
    function's argument order as ``(y_true, y_pred)`` at every call site.
    """
    y_true = _to_numpy_1d(y_true)
    y_pred = _to_numpy_1d(y_pred)
    return {
        'mse': mse(y_true, y_pred),
        'pearson': pearsonr(y_true, y_pred)[0],
        'r2': r2_score(y_true, y_pred),
        'spearman': spearmanr(y_true, y_pred)[0],
        'mae': mae(y_true, y_pred),
        'chatterjee': chatterjee_coefficient(y_true, y_pred),
    }
