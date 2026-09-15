from __future__ import annotations
from dataclasses import dataclass, field
import warnings
import numpy as np
import pandas as pd
 
__all__ = [
    "Panel", "load_prices", "to_returns", "filter_universe",
    "devolatilise", "winsorise", "standardise", "prepare",
    "scramble", "phase_randomise", "bootstrap_iid",
    "dispersed_spectrum", "factor_correlation", "simulate_returns",
]
 
 
# container
 
@dataclass
class Panel:
    """A prepared return panel plus the metadata needed to interpret it."""
 
    X: np.ndarray                        # (T, N) standardised
    tickers: list = field(default_factory=list)
    dates: pd.DatetimeIndex | None = None
    raw: pd.DataFrame | None = None      # returns before preprocessing
    dropped: list = field(default_factory=list)   # degenerate columns removed
 
    @property
    def shape(self):
        return self.X.shape
 
    @property
    def q(self):
        T, N = self.X.shape
        return N / T
 
    def window(self, start, end):
        """Slice by date, preserving alignment. Both bounds inclusive."""
        if self.dates is None:
            raise ValueError("no date index attached")
        m = (self.dates >= pd.Timestamp(start)) & (self.dates <= pd.Timestamp(end))
        return Panel(self.X[m], self.tickers, self.dates[m],
                     None if self.raw is None else self.raw.loc[m],
                     list(self.dropped))
 
 
# loading
 
def load_prices(path, date_col="date", ticker_col="ticker", price_col="adj_close"):
    """Load a long or wide price file into a wide (dates x tickers) frame.
 
    Accepts .parquet or .csv.  Long format is detected by the presence of
    `ticker_col`.  Prices must already be adjusted for splits and dividends;
    unadjusted prices produce spurious correlated jumps on ex-dates.
    """
    path = str(path)
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    if ticker_col in df.columns:
        df = df.pivot(index=date_col, columns=ticker_col, values=price_col)
    else:
        df = df.set_index(date_col)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()
 
 
def to_returns(prices, kind="log"):
    """Simple or log returns. Log is preferred: it makes the winsorisation
    threshold symmetric and keeps the aggregation additive."""
    if kind == "log":
        r = np.log(prices).diff()
    elif kind == "simple":
        r = prices.pct_change()
    else:
        raise ValueError(kind)
    return r.iloc[1:]
 
 
# Universe filtering
 
def filter_universe(returns, membership=None, asof=None, n_assets=None,
                    dollar_volume=None, min_obs_frac=0.98, max_zero_frac=0.05,
                    max_abs_return=0.9):
    """Point-in-time universe selection.
 
    Parameters
    ----------
    membership : DataFrame of bools (dates x tickers), or None
        As-of-date index membership.  Passing None and relying on a current
        constituent list is the single most common way to bias these results
        upward: survivors are more correlated than the full cross-section.
    asof : timestamp
        Date at which membership is evaluated. Defaults to the first date in
        `returns`, i.e. the start of the estimation window.
    n_assets : int, optional
        Keep the largest `n_assets` by `dollar_volume` at `asof`.
    min_obs_frac : float
        Drop columns with fewer than this fraction of non-missing observations.
    max_zero_frac : float
        Drop columns whose returns are exactly zero too often. Stale prices
        create near-zero eigenvalues and localised eigenvectors that corrupt
        the lower edge, which is exactly where the RIE is weakest.
    max_abs_return : float
        Drop columns containing an implausible single-day move (bad data,
        unadjusted corporate action).
    """
    R = returns.copy()
    asof = pd.Timestamp(asof) if asof is not None else R.index[0]
 
    if membership is not None:
        row = membership.reindex(membership.index.asof(asof), axis=0) \
            if False else membership.loc[membership.index.asof(asof)]
        keep = [c for c in R.columns if bool(row.get(c, False))]
        R = R[keep]
 
    if dollar_volume is not None and n_assets is not None:
        dv = dollar_volume.loc[:asof].tail(63).mean()
        keep = dv.reindex(R.columns).nlargest(n_assets).index
        R = R[keep]
 
    ok = R.notna().mean() >= min_obs_frac
    ok &= (R.fillna(0) == 0).mean() <= max_zero_frac
    ok &= R.abs().max() <= max_abs_return
    R = R.loc[:, ok[ok].index]
 
    R = R.dropna(axis=0, how="all").ffill(limit=2).dropna(axis=1)
    if n_assets is not None and R.shape[1] > n_assets:
        R = R.iloc[:, :n_assets]
    return R
 
 
# Preprocessing
 
def devolatilise(returns, halflife=63, min_periods=21, lag=1, floor=1e-8):
    """Divide each series by a causal, lagged EWMA volatility estimate.
 
    Marchenko-Pastur assumes i.i.d. entries.  Volatility clustering violates
    that and fattens the bulk, which then reads as extra factors.  Removing it
    first is what makes the correlation matrix the right object to clean.
 
    `lag=1` is not cosmetic: without it, today's return scales itself, which
    compresses tails and leaks a small amount of the future.
    """
    R = pd.DataFrame(returns)
    var = (R ** 2).ewm(halflife=halflife, min_periods=min_periods).mean()
    var = var.shift(lag)
    out = R / np.sqrt(var.clip(lower=floor))
    return out.dropna(how="all")
 
 
def winsorise(X, n_sigma=8.0):
    """Clip at n_sigma. One crash day can push an eigenvalue past the edge on
    its own; this is cheap insurance and should be reported, not hidden."""
    return pd.DataFrame(X).clip(-n_sigma, n_sigma)
 
 
def standardise(X, ddof=1, min_std=1e-12):
    """Z-score each column.

    A zero-variance column cannot be standardised at all.  Dividing by its
    std yields a column of NaN, which a later row-wise dropna turns into an
    empty panel -- so refuse it here instead of propagating it.  `prepare`
    removes such columns before this is reached.
    """
    X = pd.DataFrame(X)
    sd = X.std(ddof=ddof)
    if (sd <= min_std).any():
        bad = list(sd.index[sd <= min_std])
        raise ValueError(
            f"cannot standardise {len(bad)} zero-variance column(s): {bad[:8]}"
            f"{' ...' if len(bad) > 8 else ''}")
    return (X - X.mean()) / sd
 
 
def _drop_degenerate(Z, min_std=1e-12):
    """Drop columns with no variance left. Returns (kept frame, dropped names).

    A dead, suspended or fully stale series is constant, and a constant column
    is not a weak signal -- it is an undefined one.  Devolatilisation can also
    flatten a series that was not constant to begin with, so this is checked
    both before and after that step.
    """
    sd = Z.std(ddof=1)
    bad = sd.index[~(sd > min_std)]
    return Z.drop(columns=bad), list(bad)


def prepare(returns, halflife=63, n_sigma=8.0, devol=True, min_std=1e-12):
    """returns -> Panel. The canonical entry point for the rest of the package."""
    R = pd.DataFrame(returns).dropna(axis=1, how="any")
    R, dropped = _drop_degenerate(R, min_std)

    Z = devolatilise(R, halflife=halflife) if devol else R
    Z = winsorise(Z, n_sigma)
    Z, dropped_post = _drop_degenerate(Z, min_std)
    dropped += dropped_post

    Z = standardise(Z, min_std=min_std)
    Z = Z.dropna(axis=0, how="any")
    if Z.shape[1] == 0 or Z.shape[0] == 0:
        raise ValueError(
            f"preparation left an empty panel (shape {Z.shape}); "
            f"{len(dropped)} column(s) were degenerate")
    if dropped:
        warnings.warn(
            f"prepare dropped {len(dropped)} zero-variance column(s): "
            f"{dropped[:8]}{' ...' if len(dropped) > 8 else ''}",
            RuntimeWarning, stacklevel=2)

    # `raw` must stay column-aligned with X: run_backtest indexes both by the
    # same asset positions when it rebuilds Sigma from the window volatilities.
    return Panel(X=Z.to_numpy(float), tickers=list(Z.columns),
                 dates=Z.index if isinstance(Z.index, pd.DatetimeIndex) else None,
                 raw=R.loc[Z.index, Z.columns], dropped=dropped)
 
 
# Null models
 
def scramble(X, rng=None):
    """Permute each column independently in time.
 
    Destroys all cross-sectional structure; preserves every column's marginal
    distribution exactly, fat tails included.  The resulting correlation matrix
    is white Wishart by construction, so this is the empirical null: fitting it
    must recover q ~ N/T and sigma2 ~ 1, which doubles as the unit test on the
    fitting code itself.
    """
    rng = np.random.default_rng() if rng is None else rng
    S = np.array(X, float, copy=True)
    T = S.shape[0]
    for j in range(S.shape[1]):
        S[:, j] = S[rng.permutation(T), j]
    return S
 
 
def phase_randomise(X, rng=None):
    """Randomise Fourier phases per column.
 
    Preserves each column's autocorrelation while destroying cross-correlation.
    Comparing this null against `scramble` isolates how much of the
    q_eff - N/T gap is serial dependence rather than non-stationarity.
    """
    rng = np.random.default_rng() if rng is None else rng
    X = np.asarray(X, float)
    T = X.shape[0]
    F = np.fft.rfft(X, axis=0)
    ph = rng.uniform(0, 2 * np.pi, F.shape)
    ph[0] = 0.0
    if T % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=T, axis=0)
 
 
def bootstrap_iid(X, rng=None):
    """Resample whole rows with replacement.
 
    Preserves the cross-sectional correlation, destroys time-series structure.
    Use it to check that a result is not an artifact of a handful of dates.
    """
    rng = np.random.default_rng() if rng is None else rng
    X = np.asarray(X, float)
    return X[rng.integers(0, X.shape[0], X.shape[0])]
 
 
# Synthetic
 
def dispersed_spectrum(N, levels=(0.5, 1.0, 2.0), weights=(0.2, 0.6, 0.2),
                       spikes=()):
    """Population eigenvalues: a dispersed bulk plus optional spikes.
 
    The (0.5, 1, 2) / (20, 60, 20) configuration is the standard Ledoit-Wolf
    test case; keeping it makes your numbers comparable to the literature.
    """
    w = np.asarray(weights, float) / np.sum(weights)
    counts = np.floor(w * (N - len(spikes))).astype(int)
    counts[-1] += (N - len(spikes)) - counts.sum()
    pop = np.concatenate([np.full(c, l) for c, l in zip(counts, levels)]
                         + [np.asarray(spikes, float)])
    return np.sort(pop)
 
 
def factor_correlation(N, k=3, loading_sd=0.5, rng=None):
    """True correlation from a k-factor model: C = normalise(B B' + I)."""
    rng = np.random.default_rng() if rng is None else rng
    B = rng.normal(0.0, loading_sd, (N, k))
    C = B @ B.T + np.eye(N)
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)
 
 
def simulate_returns(C, T, dist="normal", df=5, rng=None):
    """Draw T observations with population correlation C.
 
    `dist='t'` gives elliptical Student-t returns, which break MP in a specific
    way (heavier right edge) and are the right stress test for O4.
    """
    rng = np.random.default_rng() if rng is None else rng
    C = np.asarray(C, float)
    N = C.shape[0]
    L = np.linalg.cholesky(C + 1e-10 * np.eye(N))
    Z = rng.standard_normal((T, N)) @ L.T
    if dist == "t":
        g = rng.chisquare(df, size=(T, 1)) / df
        Z = Z / np.sqrt(g)
        Z *= np.sqrt((df - 2) / df)
    elif dist != "normal":
        raise ValueError(dist)
    return Z
 
 
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, T = 300, 900
    C = factor_correlation(N, k=3, rng=rng)
    R = pd.DataFrame(simulate_returns(C, T, rng=rng) * 0.01,
                     index=pd.bdate_range("2018-01-01", periods=T))
    # inject volatility clustering so devolatilise has something to remove
    vol = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.05, T))), index=R.index)
    R = R.mul(vol, axis=0)
 
    p = prepare(R)
    print("panel", p.shape, "q =", round(p.q, 3))
 
    from src.spectral import spectrum, fit_mp_bulk, spacing_test, ipr
    for label, Y in [("real", p.X), ("scrambled", scramble(p.X, rng)),
                     ("phase", phase_randomise(p.X, rng))]:
        ev, vec, _ = spectrum(Y)
        f = fit_mp_bulk(ev, q0=p.q)
        ks, _ = spacing_test(ev, f["q_eff"], f["sigma2"])
        print(f"{label:10s} q_eff={f['q_eff']:.3f} sigma2={f['sigma2']:.3f} "
              f"spikes={f['n_exclude']} ks={f['ks']:.4f} spacing_p={ks.pvalue:.3f} "
              f"ipr_med={np.median(ipr(vec)):.5f} (PT={3/p.shape[1]:.5f})")
