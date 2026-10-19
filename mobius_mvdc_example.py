"""Finite inverse-zeta products: exact centring without a fitted noise series.

This is a full finite-product evaluation, not a prediction of the infinite
Euler-product tail. The tail models are in experiments/mvdc_euler_product.py.
"""
import mpmath as mp
from mvdc_utils import DEFAULT_DPS, integer, order_value, precision, primes_up_to

S = 2.0
MAX_P = 20000


def selected_primes(index):
    integer(index, "prime count")
    values = primes_up_to(MAX_P)
    if index > len(values):
        raise ValueError(f"prime count exceeds the available {len(values)} primes")
    return values[:index]


def inverse_zeta_exact_log(index, dps=DEFAULT_DPS):
    with mp.workdps(precision(dps)):
        s = mp.mpf(str(S))
        if not mp.isfinite(s) or s <= 1:
            raise ValueError("require real s > 1")
        return mp.fsum(mp.log1p(-mp.power(p, -s)) for p in selected_primes(index))


def mvdc_main_log(index, dps=DEFAULT_DPS):
    return inverse_zeta_exact_log(index, dps)


def fit_coeffs(order=5, start_idx=100, stop_idx=2000, step=100):
    """The exact logarithmic centre has identically zero log residual."""
    order_value(order)
    integer(step, "step", 1)
    selected_primes(start_idx)
    selected_primes(stop_idx)
    if start_idx > stop_idx:
        raise ValueError("training interval is reversed")
    return (mp.mpf(0),) * (order+1)


def mvdc_poly_log(index, terms=5):
    order_value(terms)
    return mvdc_main_log(index)


def compare(indices):
    with mp.workdps(DEFAULT_DPS):
        print("Finite inverse-zeta product: logarithm and residual (zero by construction)")
        for index in indices:
            value = mvdc_main_log(index)
            print(f"{index:5d} {mp.nstr(value, 24):>28} 0")


if __name__ == "__main__":
    compare((0, 10, 50, 100, 200, 500, 1000, 1500, 2000))
