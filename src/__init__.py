"""Random-matrix-theory covariance cleaning.

Four modules, in dependency order:

    data        loading, universe filtering, preprocessing, null models
    spectral    Marchenko-Pastur, the resolvent, universality diagnostics
    estimators  the shrinkage maps, all behind one `fit`/`sigma_` interface
    backtest    walk-forward minimum-variance evaluation

The intended entry point is `prepare` -> `Panel` -> an estimator -> `run_backtest`.
Every public name of every submodule is re-exported here, so `from src import
prepare, build, run_backtest` works alongside `from src.data import prepare`.

Re-export is lazy (PEP 562).  Importing the submodules eagerly would make
`python -m src.data` -- the self-test in that module -- execute it twice and
emit a runpy warning, and it keeps `import src` from pulling in the whole
package when only one module is wanted.
"""

from __future__ import annotations

import importlib

__version__ = "0.1.0"

_SUBMODULES = ("data", "spectral", "estimators", "backtest")


def _load(name):
    return importlib.import_module(f".{name}", __name__)


def __getattr__(name):
    """Resolve a submodule, or any name in a submodule's __all__, on demand."""
    if name in _SUBMODULES:
        return _load(name)
    if name == "__all__":
        names = []
        for mod in _SUBMODULES:
            names += list(getattr(_load(mod), "__all__", ()))
        return list(_SUBMODULES) + names + ["__version__"]
    for mod in _SUBMODULES:
        m = _load(mod)
        if name in getattr(m, "__all__", ()):
            return getattr(m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(list(globals()) + __getattr__("__all__")))
