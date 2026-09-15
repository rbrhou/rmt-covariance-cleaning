from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
 
from .spectral import spectrum, fit_mp_bulk
from .estimators import build
 
try:                                    
    import cvxpy as cp
    HAVE_CVXPY = True
except Exception:                      
    cp = None
    HAVE_CVXPY = False
 
__all__ = [
    "min_variance", "min_variance_long_only", "rolling_windows",
    "BacktestResult", "run_backtest", "summarise",
]
 
 
# Portfolios
 
def min_variance(Sigma, ridge=0.0):
    """Unconstrained GMV weights. Closed form, no solver."""
    Sigma = np.asarray(Sigma, float)
    N = Sigma.shape[0]
    if ridge:
        Sigma = Sigma + ridge * np.trace(Sigma) / N * np.eye(N)
    w = np.linalg.solve(Sigma, np.ones(N))
    return w / w.sum()
 
 
def min_variance_long_only(Sigma, w_max=None):
    """Long-only GMV via cvxpy.
 
    `psd_wrap` is required: numerically cleaned matrices routinely carry
    eigenvalues around -1e-15 and cvxpy's PSD check rejects them outright.
    """
    if not HAVE_CVXPY:
        raise ImportError("cvxpy not installed; use min_variance instead")
    Sigma = np.asarray(Sigma, float)
    N = Sigma.shape[0]
    w = cp.Variable(N)
    cons = [cp.sum(w) == 1, w >= 0]
    if w_max is not None:
        cons.append(w <= w_max)
    cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma))), cons).solve()
    return np.asarray(w.value, float)
 
 
# Windows
 
def rolling_windows(T, lookback, holdout=21, step=None):
    """Yield (train_slice, test_slice) with no overlap between them.
 
    `step` defaults to `holdout`, giving contiguous non-overlapping test
    periods that tile the sample exactly once.
    """
    step = holdout if step is None else step
    start = 0
    while start + lookback + holdout <= T:
        yield slice(start, start + lookback), \
              slice(start + lookback, start + lookback + holdout)
        start += step
 
 
# Driver
 
@dataclass
class BacktestResult:
    returns: pd.DataFrame          # out-of-sample portfolio returns per estimator
    predicted: pd.DataFrame        # ex-ante vol forecast per rebalance
    realised: pd.DataFrame         # realised vol over the holdout per rebalance
    turnover: pd.DataFrame
    diagnostics: pd.DataFrame      # q_eff, sigma2, spikes per window
    dates: pd.DatetimeIndex | None
 
 
def run_backtest(panel, specs, lookback=500, holdout=21, step=None,
                 long_only=False, annualise=252, verbose=True):
    """Walk forward over `panel`, fitting every estimator on each window.
 
    Parameters
    ----------
    specs : dict
        name -> (estimator_key, kwargs), e.g.
        {"rie": ("rie", {}), "linear": ("linear", {})}
    long_only : bool
        Use the constrained portfolio. Requires cvxpy.
 
    The eigendecomposition is computed once per window and shared across all
    estimators: they differ only in the map applied to the eigenvalues, so
    recomputing `eigh` per estimator is a pure k-fold waste.
 
    Look-ahead discipline: weights for a holdout are built strictly from the
    preceding lookback window, and the raw (not devolatilised) returns are used
    to evaluate them.  `test_backtest.py` asserts that scrambling data after the
    evaluation date leaves the output bit-identical.
    """
    X = np.asarray(panel.X, float)
    R = panel.raw.to_numpy(float) if panel.raw is not None else X
    T, N = X.shape
 
    names = list(specs)
    rets = {k: [] for k in names}
    pred, real, turn, diag, stamps = {k: [] for k in names}, {k: [] for k in names}, \
        {k: [] for k in names}, [], []
    prev = {k: None for k in names}
 
    wins = list(rolling_windows(T, lookback, holdout, step))
    for w, (tr, te) in enumerate(wins):
        Xtr = X[tr]
        spec = spectrum(Xtr)                      # computed once, shared
        f = fit_mp_bulk(spec[0], q0=N / lookback)
        diag.append(dict(window=w, q_naive=N / lookback, q_eff=f["q_eff"],
                         sigma2=f["sigma2"], spikes=f["n_exclude"], ks=f["ks"]))
        Rte = R[te]
 
        # Reassemble a covariance from the cleaned correlation and the window's
        # marginal volatilities: Sigma = D^{1/2} C_hat D^{1/2}.  Estimators are
        # fitted on devolatilised, standardised data, so `sigma_` is a
        # correlation matrix -- using it directly would make the predicted vol
        # dimensionless and the risk ratio meaningless.
        sd = R[tr].std(axis=0, ddof=1)
 
        for k in names:
            key, kw = specs[k]
            est = build(key, **kw).fit(Xtr, spectrum_=spec)
            S = sd[:, None] * est.sigma_ * sd[None, :]
            wt = min_variance_long_only(S) if long_only else min_variance(S)
 
            r = Rte @ wt
            rets[k].append(pd.Series(r))
            pred[k].append(np.sqrt(max(wt @ S @ wt, 0.0)))
            real[k].append(r.std(ddof=1))
            turn[k].append(np.nan if prev[k] is None
                           else np.abs(wt - prev[k]).sum())
            prev[k] = wt
 
        stamps.append(panel.dates[te][0] if panel.dates is not None else w)
        if verbose and (w % 10 == 0 or w == len(wins) - 1):
            print(f"  window {w + 1}/{len(wins)}", end="\r")
    if verbose:
        print(f"  {len(wins)} windows done      ")
 
    idx = panel.dates[lookback:lookback + holdout * len(wins)] \
        if panel.dates is not None else None
    ret_df = pd.DataFrame({k: pd.concat(v, ignore_index=True) for k, v in rets.items()})
    if idx is not None and len(idx) == len(ret_df):
        ret_df.index = idx
 
    mk = lambda d: pd.DataFrame(d, index=stamps)
    return BacktestResult(returns=ret_df, predicted=mk(pred), realised=mk(real),
                          turnover=mk(turn),
                          diagnostics=pd.DataFrame(diag).set_index("window"),
                          dates=idx)
 
 
# Metrics
 
def summarise(res: BacktestResult, annualise=252):
    """The results table. Realised volatility and risk ratio lead."""
    out = {}
    for k in res.returns.columns:
        r = res.returns[k].to_numpy()
        vol = r.std(ddof=1) * np.sqrt(annualise)
        ratio = (res.realised[k] / res.predicted[k]).median()
        out[k] = dict(
            realised_vol=vol,
            risk_ratio=float(ratio),
            vol_reduction_vs_sample=np.nan,
            sharpe=float(r.mean() / r.std(ddof=1) * np.sqrt(annualise)),
            mean_turnover=float(res.turnover[k].mean()),
            max_drawdown=float(_mdd(r)),
        )
    tab = pd.DataFrame(out).T
    if "sample" in tab.index:
        tab["vol_reduction_vs_sample"] = \
            1 - tab["realised_vol"] / tab.loc["sample", "realised_vol"]
    return tab.sort_values("realised_vol")
 
 
def _mdd(r):
    c = np.cumprod(1 + r)
    return float((c / np.maximum.accumulate(c) - 1).min())
