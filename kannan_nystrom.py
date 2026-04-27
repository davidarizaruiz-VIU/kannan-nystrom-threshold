"""
kannan_nystrom
==============

Top-level convenience namespace for the ``kannan-nystrom`` distribution.

This module re-exports the public API of the two reference modules

* :mod:`picard_nystrom`     - core 1-D framework: Gauss-Legendre /
                              trapezoidal / Simpson assembly, Kannan
                              diagnostics, Picard-Nystroem iteration,
                              smoothed semismooth Newton, two-sided
                              monotone sweep, Green-kernel BVP, Filippov
                              regularisation, real-data ecology fits.

* :mod:`picard_nystrom_2d`  - non-separable 2-D extension on the unit
                              square.

so that, after ``pip install kannan-nystrom``, end users may write the
ergonomic form

>>> from kannan_nystrom import (
...     gauss_legendre_01, NystromSystem,
...     picard_nystrom, kannan_diagnostics,
... )

equivalently to the original

>>> from picard_nystrom import (
...     gauss_legendre_01, NystromSystem,
...     picard_nystrom, kannan_diagnostics,
... )

The latter form is the one used throughout the driver scripts of the
companion paper, and is preserved unchanged for bit-identical
reproducibility of every table and figure of the manuscript.

Both forms expose exactly the same objects.
"""
from __future__ import annotations

# Public 1-D API ------------------------------------------------------------
from picard_nystrom import *           # noqa: F401, F403
from picard_nystrom import __dict__ as _pn_dict

# Public 2-D API ------------------------------------------------------------
from picard_nystrom_2d import *        # noqa: F401, F403
from picard_nystrom_2d import __dict__ as _pn2d_dict

# Distribution metadata -----------------------------------------------------
__version__ = "1.0.0"
__author__  = "David Ariza-Ruiz"
__license__ = "MIT"
__doi__     = "10.5281/zenodo.19814083"  # concept DOI (latest version)

# Build a clean __all__ from the two underlying modules, dropping dunders,
# private names, common third-party imports and ``__future__`` flags so
# that ``from kannan_nystrom import *`` only brings in user-facing
# symbols. Duplicates (e.g. ``NystromSystem`` re-exported by both
# modules) are de-duplicated through the set.
_BLACKLIST = {
    "np", "plt", "math", "os", "sys", "json",
    "Dict", "Tuple", "Optional", "Callable", "Iterable", "Sequence",
    "List", "Any", "Union", "Mapping", "Hashable",
    "annotations",  # from __future__
    "dataclass", "field",
}
def _is_third_party(obj) -> bool:
    """Return True if ``obj`` was imported from numpy/scipy/matplotlib/pandas."""
    mod = getattr(obj, "__module__", None) or ""
    return mod.split(".", 1)[0] in {"numpy", "scipy", "matplotlib", "pandas", "typing", "dataclasses", "collections"}


__all__ = sorted({
    name
    for d in (_pn_dict, _pn2d_dict)
    for name in d
    if not name.startswith("_")
    and name not in _BLACKLIST
    and not _is_third_party(d[name])
})
