from __future__ import annotations
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.stats import kstest

__all__ = [
    "correlation", "spectrum",
    "mp_edges", "mp_pdf", "mp_cdf", "fit_mp_bulk",
    "stieltjes", "default_eta",
    "unfold", "wigner_surmise", "spacing_test", "ipr", "spike_count",
]


# Eigenstructure of Correlation matrix

def correlation(X):
    """Sample correlation matrix with an exactly unit diagonal."""
    X = np.asarray(X, float)
    T = X.shape[0]
    C = (X.T @ X) / T
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def spectrum(X):
    """Return (evals ascending, evecs, correlation matrix)."""
    C = correlation(X)
    evals, evecs = np.linalg.eigh(C)
    return evals, evecs, C


# Marchenko-Pastur law

def mp_edges(q, sigma2 = 1.0):
    """Support of the MP bulk, (lambda_minus, lambda_plus)."""
    return sigma2 * (1 - np.sqrt(q)) ** 2, sigma2 * (1 + np.sqrt(q)) ** 2


def mp_pdf(x, q, sigma2 = 1.0):
    lo, hi = mp_edges(q, sigma2)
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    m = (x > lo) & (x < hi)
    out[m] = np.sqrt((hi - x[m]) * (x[m] - lo)) / (2 * np.pi * q * sigma2 * x[m])
    return out


def mp_cdf(x, q, sigma2=1.0, grid = 4000):
    """CDF of the MP bulk, normalised to one on its support.

    The atom at zero present when q > 1 is deliberately excluded: this is used
    for fitting the bulk of a correlation spectrum, not for the full law.
    """
    lo, hi = mp_edges(q, sigma2)
    g = np.linspace(lo, hi, grid)
    dens = mp_pdf(g, q, sigma2)
    cdf = np.concatenate([[0.0], np.cumsum((dens[1:] + dens[:-1]) / 2 * np.diff(g))])
    cdf /= cdf[-1]
    return np.interp(np.asarray(x, float), g, cdf, left=0.0, right=1.0)


def fit_mp_bulk(evals, n_exclude = None, q0 = 0.5, max_iter = 6,
                q_bounds = (1e-3, 50.0), zero_tol = 1e-8):
    """Fit (q_eff, sigma2) to the bulk by minimising KS distance.

    Both parameters float.  sigma2 < 1 because the spikes carry variance out of
    the bulk; q_eff > N/T because serial dependence and non-stationarity reduce
    the effective sample size.  Reporting the gap q_eff - N/T is a result, not a
    nuisance.

    The number of excluded (signal) eigenvalues is determined self-consistently.

    q > 1 is supported.  There the correlation matrix is rank deficient and
    carries an atom of N - T exact zeros at the origin; `mp_cdf` normalises to
    the continuous bulk only, so those are stripped before fitting rather than
    allowed to drag the lower edge onto zero.  The count is returned as
    `n_zero`.

    Raises
    ------
    RuntimeError
        If the optimiser never reaches a feasible point.  Returning the
        infeasible sentinel as if it were a KS distance would hand every
        downstream consumer -- RIE, Clipping, FactorModel, the backtest
        diagnostics -- a fitted edge with no fit behind it.
    """
    ev = np.sort(np.asarray(evals, float))
    n_zero = int((ev <= zero_tol * max(float(ev[-1]), 1.0)).sum())
    ev = ev[n_zero:]
    N = ev.size
    if N < 10:
        raise ValueError(
            f"only {N} non-degenerate eigenvalues; too few to fit an MP bulk")

    lo_q, hi_q = float(q_bounds[0]), float(q_bounds[1])
    q0 = float(np.clip(q0, lo_q * 1.01, hi_q * 0.99))
    max_iter = max(1, int(max_iter))

    def n_above(q, s2):
        return int((ev > mp_edges(q, s2)[1]).sum())

    if n_exclude is None:
        n_exclude = max(1, n_above(q0, 1.0))

    for _ in range(max_iter):
        n_exclude = int(np.clip(n_exclude, 1, N - 10))
        bulk = ev[: N - n_exclude]
        emp = np.arange(1, bulk.size + 1) / bulk.size

        def obj(theta):
            q, s2 = np.exp(theta)
            if not (lo_q < q < hi_q) or not (1e-3 < s2 < 10.0):
                return 1e6
            return np.max(np.abs(mp_cdf(bulk, q, s2) - emp))

        res = minimize(obj, np.log([q0, bulk.mean()]), method="Nelder-Mead",
                       options=dict(xatol=1e-4, fatol=1e-6, maxiter=800))
        q_eff, sigma2 = np.exp(res.x)
        new = max(1, n_above(q_eff, sigma2))
        if new == n_exclude:
            break
        n_exclude, q0 = new, q_eff

    if not np.isfinite(res.fun) or res.fun >= 1e6:
        raise RuntimeError(
            f"MP bulk fit found no feasible point (q0={q0:.4g}); q must lie in "
            f"{q_bounds} and sigma2 in (1e-3, 10)")

    # A q_eff sitting on a bound is a constrained optimum, not a fitted one:
    # the reported edge then reflects `q_bounds`, not the spectrum.
    if q_eff <= lo_q * 1.01 or q_eff >= hi_q * 0.99:
        warnings.warn(
            f"q_eff = {q_eff:.4g} is pinned at the q_bounds={q_bounds} boundary; "
            f"the fit is constrained and lambda_plus should not be trusted",
            RuntimeWarning, stacklevel=2)

    return dict(q_eff=float(q_eff), sigma2=float(sigma2),
                n_exclude=int(n_exclude), n_zero=n_zero, ks=float(res.fun),
                lambda_minus=float(mp_edges(q_eff, sigma2)[0]),
                lambda_plus=float(mp_edges(q_eff, sigma2)[1]))


# Stieltjes transform

def default_eta(N):
    """Regularisation height for the resolvent, eta ~ N^{-1/2} (BBP)."""
    return float(N) ** -0.5


def stieltjes(evals, z):
    """m(z) = (1/N) sum_j 1/(z - lambda_j), vectorised over complex z.

    Every sample eigenvalue is a pole, so z must carry a nonzero imaginary part.
    """
    ev = np.asarray(evals, float)
    z = np.atleast_1d(np.asarray(z, complex))
    return np.mean(1.0 / (z[:, None] - ev[None, :]), axis=1)


# Universality diagnostics

def unfold(evals, q, sigma2):
    """Bulk eigenvalue spacings rescaled to unit mean via the fitted MP CDF."""
    ev = np.sort(np.asarray(evals, float))
    lo, hi = mp_edges(q, sigma2)
    ev = ev[(ev > lo) & (ev < hi)]
    return np.diff(mp_cdf(ev, q, sigma2) * ev.size)


def wigner_surmise(s, beta=1):
    """Nearest-neighbour spacing density. beta = 1 is GOE."""
    s = np.asarray(s, float)
    if beta == 1:
        return (np.pi / 2) * s * np.exp(-np.pi * s ** 2 / 4)
    if beta == 2:
        return (32 / np.pi ** 2) * s ** 2 * np.exp(-4 * s ** 2 / np.pi)
    raise NotImplementedError(f"beta={beta}")


def spacing_test(evals, q, sigma2):
    """KS test of unfolded spacings against the GOE surmise.

    Level repulsion is a universality signature no factor model reproduces, so
    this is far stronger evidence that the bulk is noise than density agreement.
    Returns (KstestResult, normalised spacings).
    """
    s = unfold(evals, q, sigma2)
    s = s[s > 0]
    s = s / s.mean()
    return kstest(s, lambda x: 1 - np.exp(-np.pi * np.asarray(x) ** 2 / 4)), s


def ipr(evecs):
    """Inverse participation ratio per eigenvector, sum_i v_i^4.

    ~3/N for a delocalised Porter-Thomas vector.  Large values mean the vector
    is concentrated on a few assets, usually a data artifact rather than a factor.
    """
    return (np.asarray(evecs, float) ** 4).sum(axis=0)


def spike_count(evals, fit):
    """Number of eigenvalues above the fitted upper edge."""
    return int((np.asarray(evals) > fit["lambda_plus"]).sum())
