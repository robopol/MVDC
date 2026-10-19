"""Fitted analytic centres for C(2n,n), with independent exact references."""
from functools import lru_cache
import mpmath as mp
from binom_analytic import binom_exact, binom_mvdc_mp as classical_binom_mp
from mvdc_utils import (DEFAULT_DPS, finite_float, fit_inverse_powers,
                        fit_log_center, fit_log_cascade, integer, order_value, precision,
                        relative_error)

TRAINING_NS = tuple(range(50, 501, 10))


def main_log(n):
    return n * mp.log(4) - mp.log(mp.pi*n)/2


def exact_log(n):
    return mp.loggamma(2*n+1) - 2*mp.loggamma(n+1)


@lru_cache(maxsize=32)
def center_model(order=5, dps=DEFAULT_DPS):
    return fit_log_center(TRAINING_NS, exact_log, main_log, order, dps=dps)


@lru_cache(maxsize=8)
def cascade_model(dps=DEFAULT_DPS):
    return fit_log_cascade(TRAINING_NS, exact_log, main_log, 3, 7, dps=dps)


def mvdc_main(n):
    integer(n)
    with mp.workdps(DEFAULT_DPS):
        return 1.0 if n == 0 else finite_float(mp.exp(main_log(n)))


def fit_mvdc_coeffs(order=5, start=50, stop=500, step=10):
    """Return C0 and relative inverse-power coefficients for a ratio fit."""
    integer(step, "step", 1)
    fit = fit_inverse_powers(range(start, stop+1, step),
                             lambda n: mp.exp(exact_log(n)-main_log(n)), order)
    with mp.workdps(DEFAULT_DPS):
        coefficients = fit.inverse_coefficients()
        c0 = coefficients[0]
        return [c0, *(value/c0 for value in coefficients[1:])]


def fit_log_coeffs(order=5, start=50, stop=500, step=10):
    integer(step, "step", 1)
    fit = fit_log_center(range(start, stop+1, step), exact_log, main_log, order)
    return list(fit.layers[0].inverse_coefficients())


def mvdc_fitted_mp(n, order=5, dps=DEFAULT_DPS):
    integer(n)
    order_value(order)
    with mp.workdps(precision(dps)):
        return mp.mpf(1) if n == 0 else center_model(order, dps).value(n)


def mvdc_cascade2_mp(n, dps=DEFAULT_DPS):
    integer(n)
    with mp.workdps(precision(dps)):
        return mp.mpf(1) if n == 0 else cascade_model(dps).value(n)


def mvdc_cascade2(n):
    return finite_float(mvdc_cascade2_mp(n))


def classical_binom(n, include_terms=5):
    return finite_float(classical_binom_mp(n, order=include_terms))


def compare(values):
    with mp.workdps(DEFAULT_DPS):
        print("C(2n,n), relative errors: fitted log centre (5), cascade (3+7), analytic series (5)")
        print("Training points: 50,60,...,500; extrapolation is explicitly marked.")
        for n in values:
            reference = binom_exact(n)
            errors = [relative_error(mvdc_fitted_mp(n), reference),
                      relative_error(mvdc_cascade2_mp(n), reference),
                      relative_error(classical_binom_mp(n), reference)]
            print(f"{n:6d} " + " ".join(f"{mp.nstr(e, 9):>14}" for e in errors)
                  + f" extrapolation={n < 50 or n > 500}")


if __name__ == "__main__":
    compare((2, 5, 10, 20, 50, 75, 125, 250, 500, 1000))
