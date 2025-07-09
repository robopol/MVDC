"""mvdc_factorial_analytic.py
A fully *analytic* MVDC approximation for the factorial n! using exact
Bernoulli-number coefficients; no numeric fitting is involved.

Key points
----------
1.  The MVDC centre is chosen so that the leading terms n ln n − n and the
    square-root term ½ ln(2π n) are absorbed directly into the main term
    H = k^n.  After that the residual series starts with 1/(12 n).
2.  The logarithm of the correction is the classical Stirling series with
    Bernoulli numbers.  We keep as many odd-power terms as requested
    (1/n, 1/n³, 1/n⁵, …).
3.  The module provides both float and *mpmath* high-precision variants
    and a small demo that compares MVDC(order=4 → 1/n⁷) against the
    popular Ramanujan/Gosper 6-th-root formula (error ~O(n⁻⁶)).

All names and comments are deliberately written in English – the previous
Slovak prototype remains untouched for reference.
"""

from __future__ import annotations

import math
from typing import List

try:
    import mpmath as mp  # type: ignore

    _MPMATH_AVAILABLE = True
except ModuleNotFoundError:
    mp = None  # type: ignore
    _MPMATH_AVAILABLE = False

PI = math.pi

# ---------------------------------------------------------------------------
#  Core helpers (float) ------------------------------------------------------
# ---------------------------------------------------------------------------

def _log_factorial_exact(n: int) -> float:
    """Return ln(n!) using math.lgamma for numerical stability."""

    return math.lgamma(n + 1.0)


def _mvdc_main_log(n: int) -> float:
    """MVDC *main* term H = k^n with centre
        k = (n / e) * (2π n)^{1/(2n)}.
    Returns ln H.
    """

    if n < 2:
        return 0.0
    return n * (math.log(n) - 1.0) + 0.5 * (math.log(2 * PI) + math.log(n))


def _stirling_log_correction(n: int, order: int = 4) -> float:
    """Return ln-correction Σ c_k / n^{2k−1} up to the requested order.

    order = 1 → keep term 1/n (B₂)
    order = 2 → keep up to 1/n³ (B₄)
    … and so on.  Internally we hard-code the rational Bernoulli fractions
    to avoid any rounding.
    """

    inv_n = 1.0 / n
    inv_n2 = inv_n * inv_n

    corr = 0.0
    if order >= 1:
        corr += (1.0 / 12.0) * inv_n                                # 1/n
    if order >= 2:
        corr += -(1.0 / 360.0) * inv_n2 * inv_n                     # 1/n³
    if order >= 3:
        corr += (1.0 / 1260.0) * inv_n2 * inv_n2 * inv_n            # 1/n⁵
    if order >= 4:
        corr += -(1.0 / 1680.0) * inv_n2 * inv_n2 * inv_n2 * inv_n  # 1/n⁷
    if order >= 5:
        corr += (1.0 / 1188.0) * inv_n2**4 * inv_n                 # 1/n⁹
    return corr

# ---------------------------------------------------------------------------
#  Public API (float) --------------------------------------------------------
# ---------------------------------------------------------------------------

def factorial_mvdc(n: int, order: int = 4) -> float:
    """Return MVDC approximation of n! (float) with Bernoulli corrections.

    order counts how many odd-power terms to include – by default 4 → 1/n⁷.
    """

    if n < 2:
        return 1.0

    log_H = _mvdc_main_log(n)
    log_corr = _stirling_log_correction(n, order=order)
    return math.exp(log_H + log_corr)

# ---------------------------------------------------------------------------
#  High-precision variant with *mpmath* --------------------------------------
# ---------------------------------------------------------------------------

def factorial_mvdc_mp(n: int, *, order: int = 4, dps: int = 50):
    """High-precision MVDC approximation using *mpmath* (if available)."""

    if not _MPMATH_AVAILABLE:
        raise ImportError("mpmath is not available – `pip install mpmath`. ")

    if n < 2:
        return mp.mpf(1)

    mp.mp.dps = dps
    n_mp = mp.mpf(n)

    # main term (same closed form, but in mp precision)
    log_H = n_mp * (mp.log(n_mp) - 1) + mp.mpf("0.5") * (mp.log(2 * mp.pi) + mp.log(n_mp))

    # Bernoulli-series correction in mp
    inv_n = 1 / n_mp
    inv_n2 = inv_n * inv_n

    coeffs = [mp.mpf(1) / 12,
              -mp.mpf(1) / 360,
              mp.mpf(1) / 1260,
              -mp.mpf(1) / 1680,
              mp.mpf(1) / 1188]

    log_corr = mp.mpf(0)
    if order >= 1:
        log_corr += coeffs[0] * inv_n
    if order >= 2:
        log_corr += coeffs[1] * inv_n2 * inv_n
    if order >= 3:
        log_corr += coeffs[2] * inv_n2 * inv_n2 * inv_n
    if order >= 4:
        log_corr += coeffs[3] * inv_n2 * inv_n2 * inv_n2 * inv_n
    if order >= 5:
        log_corr += coeffs[4] * inv_n2**4 * inv_n

    return mp.e ** (log_H + log_corr)

# ---------------------------------------------------------------------------
#  Reference: Ramanujan / Gosper 6-th-root formula ---------------------------
# ---------------------------------------------------------------------------

def factorial_ramanujan(n: int) -> float:
    """Return Ramanujan–Göspel approximation (k = 6)."""

    if n < 2:
        return 1.0

    # Ramanujan’s polynomial for (n + 1)!^6 root
    poly = 1 + 1/(2*n) + 1/(144*n**2) - 5/(1296*n**3)

    base = (n + 1/2) * math.e**(-1)
    return ( (2*PI)**0.5 * (n + 1)**(n + 0.5) * math.exp(-n - 1) * poly )

# ---------------------------------------------------------------------------
#  Small demo ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def _demo() -> None:
    if not _MPMATH_AVAILABLE:
        print("mpmath not available – demo skipped")
        return

    mp.mp.dps = 60
    ns: List[int] = [5, 10, 20, 50, 100, 500, 1000]

    print("\nHigh-precision comparison (60 dps) – MVDC vs Ramanujan\n" + "-"*95)
    print(f"{'n':>6} | {'rel err MVDC(1/n^7)':>22} | {'rel err Ramanujan':>18}")
    print("-"*95)

    for n in ns:
        exact = mp.factorial(n)
        mvdc = factorial_mvdc_mp(n, order=4, dps=60)

        # Ramanujan – compute fully in mp to avoid overflow
        n_mp = mp.mpf(n)
        poly = 1 + mp.mpf(1)/(2*n_mp) + mp.mpf(1)/(144*n_mp**2) - mp.mpf(5)/(1296*n_mp**3)

        log_ram = (
            0.5 * mp.log(mp.pi)
            + (mp.mpf(1)/6) * mp.log(8*n_mp**3 + 4*n_mp**2 + n_mp + mp.mpf('1')/30)
            + n_mp * mp.log(n_mp) - n_mp
        )
        ram = mp.e ** log_ram

        rel_mvdc = abs(mvdc / exact - 1)
        rel_ram = abs(ram / exact - 1)

        mvdc_str = mp.nstr(rel_mvdc, 6, min_fixed=0, max_fixed=0)
        ram_str = mp.nstr(rel_ram, 6, min_fixed=0, max_fixed=0)
        print(f"{n:6d} | {mvdc_str:>22} | {ram_str:>18}")
    print("-"*95)


if __name__ == "__main__":
    _demo() 