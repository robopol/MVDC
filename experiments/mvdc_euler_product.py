"""Euler tail = exact finite prime block + explicitly labelled density model.

Logarithmic central moments are optional diagnostics. They are not correction
terms to an already exactly evaluated finite log product.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mpmath as mp
from mvdc_utils import DEFAULT_DPS, integer, precision, primes_up_to


def exponent(s):
    value = mp.mpf(s)
    if not mp.isfinite(value) or value <= 1:
        raise ValueError("this real Euler-product experiment requires s > 1")
    return value


def log_factor(p, s):
    return -mp.log1p(-mp.power(p, -s))


def central_moments(primes, s, r_max):
    """Descriptors of logarithmic data only; not a tail expansion."""
    integer(r_max, "r_max", 2)
    s = exponent(s)
    values = [log_factor(p, s) for p in primes]
    if not values:
        return mp.mpf(0), {r: mp.mpf(0) for r in range(2, r_max+1)}
    mean = mp.fsum(values)/len(values)
    return mean, {r: mp.fsum((value-mean)**r for value in values) for r in range(2, r_max+1)}


def mvdc_series(moments, n):
    """Historical diagnostic, deliberately never added to the exact block."""
    integer(n)
    partial, running = {}, mp.mpf(0)
    for r in sorted(moments):
        if n:
            running += (-1)**(r-1)*moments[r]/(r*mp.mpf(n)**(r-1))
        partial[r] = running
    return partial


def tail_integral(limit, s):
    """Prime-density model integral_M^infinity x^-s/log(x) dx."""
    integer(limit, "limit", 2)
    return mp.e1((exponent(s)-1)*mp.log(limit))


def absolute_tail_bound(limit, s):
    """Analytic absolute tail bound, evaluated at the caller's precision.

    This numeric value is not a directed-rounding interval certificate.
    """
    integer(limit, "limit", 2)
    s = exponent(s)
    return mp.power(limit, 1-s)/((s-1)*(1-mp.power(2, -s)))


def tail_mvdc(N_head, s=2, r_max=6, tail_factor=10, dps=DEFAULT_DPS):
    integer(N_head, "N_head", 2)
    integer(r_max, "r_max", 2)
    with mp.workdps(precision(dps)):
        s = exponent(s)
        factor = mp.mpf(tail_factor)
        if not mp.isfinite(factor) or factor <= 1:
            raise ValueError("tail_factor must be finite and > 1")
        limit = int(N_head*factor)
        if limit <= N_head:
            raise ValueError("the tail block must extend past N_head")
        block = [p for p in primes_up_to(limit) if p > N_head]
        finite_block = mp.fsum(log_factor(p, s) for p in block)
        _, moments = central_moments(block, s, r_max)
        diagnostics = mvdc_series(moments, len(block))
        model = tail_integral(limit, s)
        return finite_block+model, finite_block, diagnostics, model, len(block), limit


def experiment_tail(s=2, N_list=None, r_max=6, tail_factor=10):
    with mp.workdps(DEFAULT_DPS):
        s = exponent(s)
        for cutoff in ((1000, 5000, 10000) if N_list is None else N_list):
            head = mp.fsum(log_factor(p, s) for p in primes_up_to(cutoff))
            exact = mp.log(mp.zeta(s))-head
            approximation, block, diagnostic, model, count, limit = tail_mvdc(cutoff, s, r_max, tail_factor)
            print(f"N={cutoff}, M={limit}, finite block primes={count}")
            print("  exact missing tail :", mp.nstr(exact, 16))
            print("  finite block       :", mp.nstr(block, 16))
            print("  density tail model :", mp.nstr(model, 16))
            print("  approximation error:", mp.nstr(approximation-exact, 10))
            print("  omitted-tail bound :", mp.nstr(absolute_tail_bound(limit, s), 10))
            print("  old moment diagnostic (not added):", mp.nstr(diagnostic[r_max], 10))


def experiment(s=2, N_list=None, r_max=6):
    """Compatibility entry point for the corrected tail experiment."""
    experiment_tail(s, N_list, r_max)


if __name__ == "__main__":
    experiment_tail()
