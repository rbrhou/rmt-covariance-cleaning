from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from .spectral import spectrum, fit_mp_bulk, stieltjes, default_eta, mp_edges

__all__ = [
    "CovarianceEstimator", "Sample", "Clipping", "LinearShrinkage",
    "NonlinearShrinkage", "RIE", "CrossValidatedShrinkage",
    "Oracle", "REGISTRY", "build",
    "invert_spike", "spike_overlap",
]


# Base class

class CovarianceEstimator(ABC):
    """Fit on X (T, N); expose the cleaned matrix and the shrinkage map.

    `eigenvalues_` is part of the public surface, not an implementation detail:
    it is what lets every estimator's xi(lambda) map be plotted in one loop.
    """

    name: str = "base"

    def fit(self, X, spectrum_=None):
        """Fit on X (T, N).

        `spectrum_` is an optional (evals, evecs, C) triple from
        `spectral.spectrum(X)`.  Estimators differ only in the map applied to
        the eigenvalues, so when several are fitted to the same window the
        caller computes `eigh` once and passes the result to all of them.
        """
        X = np.asarray(X, float)
        self.T_, self.N_ = X.shape
        self.q_ = self.N_ / self.T_
        if spectrum_ is None:
            spectrum_ = spectrum(X)
        self.lambda_, self.U_, self.C_ = spectrum_
        if self.lambda_.shape[0] != self.N_:
            raise ValueError(
                f"spectrum_ has {self.lambda_.shape[0]} eigenvalues but X has "
                f"{self.N_} columns; it belongs to a different window")
        self.eigenvalues_ = np.asarray(self._shrink(X), float)
        return self

    @abstractmethod
    def _shrink(self, X):
        """Return the cleaned eigenvalues xi_i, aligned with self.lambda_."""

    @property
    def sigma_(self):
        return (self.U_ * self.eigenvalues_) @ self.U_.T

    @property
    def precision_(self):
        return (self.U_ / self.eigenvalues_) @ self.U_.T

    def map_(self):
        """(lambda, xi) pairs — the shrinkage map, for plotting."""
        return self.lambda_, self.eigenvalues_


# Baselines

class Sample(CovarianceEstimator):
    name = "sample"

    def _shrink(self, X):
        return self.lambda_


class Clipping(CovarianceEstimator):
    """Laloux et al.: keep eigenvalues above the MP edge, flatten the bulk.

    Trace-preserving.  Discontinuous at the edge, which is why it produces
    jumpier portfolio weights than the smooth estimators.
    """

    name = "clipping"

    def __init__(self, fit_bulk=True):
        self.fit_bulk = fit_bulk

    def _shrink(self, X):
        if self.fit_bulk:
            f = fit_mp_bulk(self.lambda_, q0=self.q_)
            edge = f["lambda_plus"]
            self.fit_ = f
        else:
            edge = mp_edges(self.q_)[1]
        keep = self.lambda_ > edge
        xi = self.lambda_.copy()
        if (~keep).any():
            xi[~keep] = self.lambda_[~keep].mean()
        return xi


class LinearShrinkage(CovarianceEstimator):
    """Ledoit-Wolf (2004): xi = alpha * lambda + (1 - alpha) * mean(lambda).

    The affine special case of nonlinear shrinkage.  Near-optimal when q is
    small and the population spectrum is tight; visibly suboptimal otherwise.
    """

    name = "linear"

    def __init__(self, alpha=None):
        self.alpha = alpha

    def _shrink(self, X):
        mu = self.lambda_.mean()
        if self.alpha is not None:
            a = float(self.alpha)
        else:
            T, N = self.T_, self.N_
            E = self.C_
            d2 = np.mean((self.lambda_ - mu) ** 2)
            sq = (X ** 2).sum(axis=1)                       # ||x_t||^2 per obs
            bbar2 = (np.sum(sq ** 2) / T
                     - 2 * np.sum(X.T * (E @ X.T)) / T
                     + np.linalg.norm(E, "fro") ** 2) / (T * N)
            a = 1.0 - min(d2, bbar2) / d2
        self.alpha_ = a
        return a * self.lambda_ + (1 - a) * mu


# Nonlinear Shrinkage / RIE

def _lp_formula(lam, q, m):
    """Ledoit-Peche / BBP: xi = lambda / |1 - q + q*lambda*m(lambda)|^2."""
    return lam / np.abs(1 - q + q * lam * m) ** 2


def spike_overlap(theta, q, sigma2=1.0):
    """BBP squared overlap between the sample and population spike vectors.

        omega(theta) = (1 - q / (t - 1)^2) / (1 + q / (t - 1)),   t = theta / sigma2

    Zero at and below the detectability threshold t = 1 + sqrt(q): there the
    outlier has merged into the bulk and its eigenvector carries no information
    about the population direction.  This is the quantity that makes "the spike
    is visible" and "the spike is estimable" different statements.
    """
    t = np.asarray(theta, float) / sigma2
    q = float(q)
    d = t - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        om = (1.0 - q / d ** 2) / (1.0 + q / d)
    om = np.where(t > 1.0 + np.sqrt(q), om, 0.0)
    return np.clip(np.nan_to_num(om, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)


def invert_spike(lam, q, sigma2=1.0):
    """Recover the population spike theta from an observed outlier lambda.

    Inverts the BBP map lambda = sigma2 * t * (1 + q / (t - 1)), which rearranges
    to t^2 - (1 + u - q) t + u = 0 with u = lambda / sigma2.  The larger root is
    the physical branch; the smaller one lies below the detectability threshold
    where the map is decreasing and not invertible.

    Returns
    -------
    theta : ndarray
        Recovered population eigenvalues, NaN where `lam` is not detectable.
    detectable : ndarray of bool
        Whether lambda exceeds sigma2 * (1 + sqrt(q))^2, i.e. whether there is
        anything to recover at all.
    """
    u = np.asarray(lam, float) / sigma2
    q = float(q)
    b = 1.0 + u - q
    disc = b ** 2 - 4.0 * u
    detectable = (u > (1.0 + np.sqrt(q)) ** 2) & (disc >= 0.0)
    root = (b + np.sqrt(np.where(disc > 0.0, disc, 0.0))) / 2.0
    theta = np.where(detectable, root * sigma2, np.nan)
    return theta, detectable

def _isotonic(y):
    """Pool-adjacent-violators: nearest non-decreasing sequence to y in L2."""
    y = np.asarray(y, float)
    n = y.size
    val, wt = np.empty(n), np.empty(n)
    val[0], wt[0] = y[0], 1.0
    top = 0
    for i in range(1, n):
        top += 1
        val[top], wt[top] = y[i], 1.0
        while top > 0 and val[top - 1] > val[top]:
            w = wt[top - 1] + wt[top]
            val[top - 1] = (wt[top - 1] * val[top - 1] + wt[top] * val[top]) / w
            wt[top - 1] = w
            top -= 1
    out, j = np.empty(n), 0
    for b in range(top + 1):
        k = j + int(wt[b])
        out[j:k] = val[b]
        j = k
    return out


class RIE(CovarianceEstimator):
    """Bun-Bouchaud-Potters rotationally invariant estimator.

    Estimates the Stieltjes transform directly from the sample eigenvalues with
    a finite imaginary regularisation eta ~ N^{-1/2}.  `eta` is the only tuning
    knob and the sensitivity plot in eta belongs in the writeup.

    `isotropic=True` applies the small-eigenvalue correction near the lower
    edge, where the asymptotics are weakest.
    """

    name = "rie"

    def __init__(self, eta=None, use_q_eff=True, isotropic=True,
                 preserve_trace=True):
        self.eta = eta
        self.use_q_eff = use_q_eff
        self.isotropic = isotropic
        self.preserve_trace = preserve_trace

    def _shrink(self, X):
        eta = default_eta(self.N_) if self.eta is None else float(self.eta)
        self.eta_ = eta
        f = fit_mp_bulk(self.lambda_, q0=self.q_)
        self.fit_ = f
        q = f["q_eff"] if self.use_q_eff else self.q_
        lo, hi = f["lambda_minus"], f["lambda_plus"]

        m = stieltjes(self.lambda_, self.lambda_ - 1j * eta)
        xi = self.lambda_.copy()

        # The Ledoit-Peche formula is a bulk result.  Applied to outliers it
        # over-shrinks badly (it has no notion of a spike), so the bulk and the
        # outliers are handled on separate branches.
        bulk = self.lambda_ <= hi
        xi[bulk] = _lp_formula(self.lambda_[bulk], q, m[bulk])

        # Lower edge: the asymptotics are weakest here, so flatten rather than
        # trust the formula.  BBP's isotropic correction is the principled v2.
        if self.isotropic:
            small = self.lambda_ < lo
            if small.any() and (~small).any():
                xi[small] = xi[bulk & ~small].min()

        xi = np.maximum(xi, 1e-10)
        if self.preserve_trace:
            xi *= self.lambda_.sum() / xi.sum()
        return xi


class NonlinearShrinkage(CovarianceEstimator):
    """Ledoit-Wolf (2020) analytical nonlinear shrinkage.

    Kernel-estimates the spectral density and its Hilbert transform, then
    applies the same Ledoit-Peche formula.  Mathematically identical to `RIE`
    in the limit; the kernel bandwidth h plays the role of eta.  Running both
    and checking they agree eigenvalue-by-eigenvalue is the cheapest available
    correctness test on the whole pipeline.
    """

    name = "nonlinear"

    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth

    def _shrink(self, X):
        lam = self.lambda_
        N, T, q = self.N_, self.T_, self.q_
        h = T ** -0.35 if self.bandwidth is None else float(self.bandwidth)
        self.h_ = h
        # Epanechnikov kernel density and Hilbert transform, evaluated at each lambda
        L = np.tile(lam, (N, 1)).T
        H = h * lam
        Lm = (L - L.T) / H
        f = np.mean(np.sqrt(np.maximum(0.0, 4 - Lm ** 2)) / (2 * np.pi * Lm * H
                                                             + 1e-300) * Lm, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            Hf = np.mean(
                (np.sign(Lm) * np.sqrt(np.maximum(0.0, Lm ** 2 - 4)) - Lm)
                / (2 * np.pi * H), axis=1)
        denom = (np.pi * q * lam * f) ** 2 + (1 - q - np.pi * q * lam * Hf) ** 2
        return np.maximum(lam / np.maximum(denom, 1e-12), 1e-10)


class CrossValidatedShrinkage(CovarianceEstimator):
    """K-fold cross-validated eigenvalue shrinkage.

    The non-asymptotic counterpart to the RIE.  Rather than invoking a large-N
    limit, it measures the out-of-sample variance along each eigendirection
    directly: eigenvectors come from K-1 folds, the variance along them from the
    held-out fold.  Taking the two from the same data is precisely what biases
    the sample covariance, so never doing that is the whole method.

    Makes no asymptotic assumption and no distributional one, which is what
    makes agreement with the RIE evidence for both.  Costs K extra `eigh` calls
    and is noisier at small T.
    """

    name = "cv"

    def __init__(self, n_folds=5, rng=0, isotonic=True, preserve_trace=True):
        self.n_folds = n_folds
        self.rng = rng
        self.isotonic = isotonic
        self.preserve_trace = preserve_trace

    def _shrink(self, X):
        T, N = self.T_, self.N_
        k = int(max(2, min(self.n_folds, T)))
        rng = (self.rng if isinstance(self.rng, np.random.Generator)
               else np.random.default_rng(self.rng))
        folds = np.array_split(rng.permutation(T), k)
        self.n_folds_ = k

        xi = np.zeros(N)
        for i in range(k):
            te = folds[i]
            tr = np.concatenate([folds[j] for j in range(k) if j != i])
            # Rescale the held-out rows by the training standard deviations so
            # the projection is on the same correlation scale as the training
            # eigenvectors; the test fold must contribute no scale of its own.
            sd = X[tr].std(axis=0, ddof=1)
            sd = np.where(sd > 1e-12, sd, 1.0)
            _, U_tr, _ = spectrum(X[tr])
            xi += (((X[te] / sd) @ U_tr) ** 2).mean(axis=0)
        xi /= k

        # lambda_ is ascending, so xi should be too.  Cross-validation noise
        # breaks that at the resolution of the spacing; project back onto the
        # monotone cone rather than leaving crossings in the map.
        if self.isotonic:
            xi = _isotonic(xi)
        xi = np.maximum(xi, 1e-10)
        if self.preserve_trace:
            xi *= self.lambda_.sum() / xi.sum()
        return xi


class Oracle(CovarianceEstimator):
    """xi_i = u_i' C u_i, the unattainable target.  Synthetic data only."""

    name = "oracle"

    def __init__(self, C_true):
        self.C_true = np.asarray(C_true, float)

    def _shrink(self, X):
        return np.einsum("ij,ik,kj->j", self.U_, self.C_true, self.U_)


# Registry

REGISTRY = {
    cls.name: cls for cls in
    (Sample, Clipping, LinearShrinkage, FactorModel, RIE, NonlinearShrinkage,
     CrossValidatedShrinkage, Oracle)
}


def build(name, **kwargs):
    """Instantiate by name; keeps configs as plain YAML."""
    try:
        return REGISTRY[name](**kwargs)
    except KeyError:
        raise KeyError(f"unknown estimator {name!r}; have {sorted(REGISTRY)}")
