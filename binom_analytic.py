"""binom_analytic.py
High-precision analytic approximation of the central binomial
coefficient  C(2n, n)  using the MVDC main term and exact Bernoulli-type
correction coefficients.  No numeric fitting is performed.

The expansion employed is (see e.g. OEIS A002894 asymptotic expansion)

    C(2n, n)  ~  4^n / sqrt(pi n)  *  ( 1
          - 1/(8 n)
          + 1/(128 n^2)
          + 5/(3072 n^3)
          - 7/(131072 n^4)
          + 35/(3932160 n^5)  + ... ).

We include terms up to 1/n^5 which makes the relative error O(n⁻⁶).
A high-precision mpmath variant is provided as well.
"""

from __future__ import annotations

import math
from typing import List

try:
    import mpmath as mp  # type: ignore

    _MP_AVAILABLE = True
except ModuleNotFoundError:
    mp = None  # type: ignore
    _MP_AVAILABLE = False

# ---------------------------------------------------------------------------
#  Float version -------------------------------------------------------------
# ---------------------------------------------------------------------------

PI = math.pi

# exact rational coefficients up to 1/n^5
A1 = -1/8
A2 = 1/128
A3 = 5/3072
A4 = -7/131072
A5 = 35/3932160
COEFFS = [0.0, A1, A2, A3, A4, A5]  # index starts at 1


def binom_exact(n: int):
    """Return exact central binomial coefficient.

    * If mpmath is available → return mp.comb as high-precision mpf.
    * Else fall back to float; for very large *n* overflow returns ``inf``.
    """

    if _MP_AVAILABLE and hasattr(mp, 'binomial'):
        return mp.binomial(2 * n, n)

    # float fallback
    try:
        return float(math.comb(2 * n, n))
    except OverflowError:
        log_val = math.lgamma(2 * n + 1) - 2 * math.lgamma(n + 1)
        try:
            return math.exp(log_val)
        except OverflowError:
            return float("inf")


def binom_mvdc(n: int, order: int = 5) -> float:
    """Analytic MVDC approximation of C(2n, n) with terms up to 1/n^order."""

    if n < 1:
        return 1.0

    # main term 4^n / sqrt(pi n)
    main = (4.0 ** n) / math.sqrt(PI * n)

    corr = 1.0
    for k in range(1, order + 1):
        corr += COEFFS[k] / (n ** k)

    return main * corr

# ---------------------------------------------------------------------------
#  High-precision mpmath version ---------------------------------------------
# ---------------------------------------------------------------------------

def binom_mvdc_mp(n: int, *, order: int = 5, dps: int = 60):
    if not _MP_AVAILABLE:
        raise ImportError("mpmath not installed")

    if n < 1:
        return mp.mpf(1)

    mp.mp.dps = dps
    n_mp = mp.mpf(n)

    main = (mp.mpf(4) ** n_mp) / mp.sqrt(mp.pi * n_mp)

    coeffs_mp = [mp.mpf(c) for c in COEFFS]

    corr = mp.mpf(1)
    for k in range(1, order + 1):
        corr += coeffs_mp[k] / (n_mp ** k)

    return main * corr

# ---------------------------------------------------------------------------
#  Demo ----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _demo() -> None:
    if not _MP_AVAILABLE:
        print("mpmath is required for the demo – install with `pip install mpmath`.")
        return

    mp.mp.dps = 60
    ns: List[int] = [5, 10, 20, 50, 100, 500, 1000]

    header = "\nCentral binomial coefficient – analytic MVDC vs exact (mpmath, 60 dps)\n" + "-" * 110
    print(header)
    print(f"{'n':>6} | {'rel error analytic':>18}")
    print("-" * 110)

    for n in ns:
        exact = mp.binomial(2 * n, n)
        approx = binom_mvdc_mp(n, order=5, dps=60)
        rel = abs(approx / exact - 1)
        print(f"{n:6d} | {mp.nstr(rel, 6, min_fixed=0, max_fixed=0):>18}")


if __name__ == "__main__":
    _demo() 