"""Wallis finite products, fitted limit centres, and correct asymptotic series."""
from functools import lru_cache
import math
import mpmath as mp
from mvdc_utils import (DEFAULT_DPS, finite_float, fit_inverse_powers,
                        fit_log_center, integer, inverse_series, order_value, precision,
                        relative_error, wallis_coefficients)

TRAINING_NS = tuple(range(20, 501, 20))


def wallis_exact_mp(n, dps=DEFAULT_DPS):
    integer(n)
    with mp.workdps(precision(dps)):
        c = math.comb(2*n, n)
        return mp.power(16, n) / ((2*n+1) * mp.mpf(c)**2)


def wallis_exact(n):
    return finite_float(wallis_exact_mp(n))


def wallis_product_mp(n, dps=DEFAULT_DPS):
    integer(n)
    with mp.workdps(precision(dps)):
        return mp.exp(-mp.fsum(mp.log1p(-mp.mpf(1)/(4*k*k)) for k in range(1, n+1)))


def main_log(n):
    return mp.log(mp.pi/2)


def exact_log(n):
    x = mp.mpf(n)
    return mp.log(mp.pi/2) + 2*mp.loggamma(x+1) - mp.loggamma(x+mp.mpf("0.5")) - mp.loggamma(x+mp.mpf("1.5"))


@lru_cache(maxsize=32)
def center_model(order=5, dps=DEFAULT_DPS):
    return fit_log_center(TRAINING_NS, exact_log, main_log, order, dps=dps)


def mvdc_main(n):
    integer(n)
    return 1.0 if n == 0 else math.pi/2


def fit_mvdc_coeffs(order=5, start=20, stop=500, step=20):
    integer(step, "step", 1)
    fit = fit_inverse_powers(range(start, stop+1, step),
                             lambda n: mp.exp(exact_log(n)-main_log(n)), order)
    with mp.workdps(DEFAULT_DPS):
        coefficients = fit.inverse_coefficients()
        c0 = coefficients[0]
        return [c0, *(value/c0 for value in coefficients[1:])]


def wallis_fitted_mp(n, order=5, dps=DEFAULT_DPS):
    integer(n)
    order_value(order)
    with mp.workdps(precision(dps)):
        return mp.mpf(1) if n == 0 else center_model(order, dps).value(n)


def wallis_asymptotic_mp(n, order=2, dps=DEFAULT_DPS):
    integer(n)
    order_value(order)
    with mp.workdps(precision(dps)):
        if n == 0:
            return mp.mpf(1)
        return mp.pi/2 * inverse_series(wallis_coefficients(order), n)


def wallis_asymptotic(n, order=2):
    return finite_float(wallis_asymptotic_mp(n, order))


def compare_Wallis(values):
    with mp.workdps(DEFAULT_DPS):
        print("Wallis relative errors: exact-product evaluation, fitted log centre (5), correct series (5)")
        print("Training points: 20,40,...,500")
        for n in values:
            reference = wallis_exact_mp(n)
            errors = [relative_error(wallis_product_mp(n), reference),
                      relative_error(wallis_fitted_mp(n), reference),
                      relative_error(wallis_asymptotic_mp(n, 5), reference)]
            print(f"{n:6d} " + " ".join(f"{mp.nstr(e, 9):>14}" for e in errors)
                  + f" extrapolation={n < 20 or n > 500}")


if __name__ == "__main__":
    compare_Wallis((1, 2, 5, 10, 25, 50, 75, 125, 250, 500, 1000))
