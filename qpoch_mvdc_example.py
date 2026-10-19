"""q-Pochhammer: an infinite-product centre with a convergent q**N residual.

The residual is exponential in N; fitting rounding noise to powers 1/N is
unnecessary. All factors are positive for the declared real |a|, |q| < 1.
"""
import mpmath as mp
from mvdc_utils import DEFAULT_DPS, integer, order_value, precision

A, Q = 0.8, 0.3


def parameters():
    a, q = mp.mpf(str(A)), mp.mpf(str(Q))
    if not mp.isfinite(a) or not mp.isfinite(q) or abs(a) >= 1 or abs(q) >= 1:
        raise ValueError("require real |a| < 1 and |q| < 1")
    return a, q


def qpoch_exact_log(n, dps=DEFAULT_DPS):
    integer(n)
    with mp.workdps(precision(dps)):
        a, q = parameters()
        return mp.fsum(mp.log1p(-a*q**k) for k in range(n))


def mvdc_main_log(n, dps=DEFAULT_DPS):
    integer(n)
    with mp.workdps(precision(dps)):
        a, q = parameters()
        return mp.log(mp.qp(a, q))


def fit_coeffs(order=5, **unused):
    """Compatibility helper: exact coefficients of powers (q**N)**r.

    The coefficients are analytic rather than fitted. The zero intercept is
    included explicitly; callers must use the exponential q**N coordinate.
    """
    if unused:
        raise TypeError("q-Pochhammer uses analytic q**N coefficients, not a training interval")
    order_value(order)
    with mp.workdps(DEFAULT_DPS):
        a, q = parameters()
        return (mp.mpf(0), *(a**r/(r*(1-q**r)) for r in range(1, order+1)))


def mvdc_poly_log(n, terms=5, dps=DEFAULT_DPS):
    integer(n)
    order_value(terms)
    with mp.workdps(precision(dps)):
        if n == 0:
            return mp.mpf(0)
        a, q = parameters()
        correction = mp.fsum((a*q**n)**r/(r*(1-q**r)) for r in range(1, terms+1))
        return mvdc_main_log(n, dps) + correction


def log_error_bound(n, terms=5, dps=DEFAULT_DPS):
    integer(n)
    order_value(terms)
    with mp.workdps(precision(dps)):
        if n == 0:
            return mp.mpf(0)
        a, q = parameters()
        z = abs(a*q**n)
        return z**(terms+1)/((terms+1)*(1-z)*(1-abs(q)))


def compare(values):
    with mp.workdps(DEFAULT_DPS):
        print("q-Pochhammer: absolute log error and analytic truncation bound (roundoff excluded)")
        for n in values:
            exact = qpoch_exact_log(n)
            error = abs(mvdc_poly_log(n)-exact)
            print(f"{n:5d} {mp.nstr(error, 10):>16} {mp.nstr(log_error_bound(n), 10):>16}")


if __name__ == "__main__":
    compare((0, 1, 2, 5, 10, 20, 50, 100, 200))
