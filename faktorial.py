"""Factorial examples sharing one validated centre and correction implementation."""
from functools import lru_cache
import math
import mpmath as mp
from mvdc_utils import (DEFAULT_DPS, finite_float, fit_log_center, fit_log_cascade,
                        integer, mvdc_generic_center, relative_error)
from mvdc_factorial_analytic import (factorial_main_log_mp, factorial_mvdc,
                                     factorial_mvdc_mp)

TRAINING_NS = tuple(range(5, 51, 5))


def exact_factorial(n):
    integer(n)
    return math.factorial(n)


def stirling_approx(n):
    return factorial_mvdc(n, 0)


def stirling_approx_k3(n):
    return factorial_mvdc(n, 3)


def stirling_approx_k5(n):
    return factorial_mvdc(n, 5)


def geometric_center_main(n):
    """Historical endpoint-centre estimate; no accuracy guarantee is implied."""
    integer(n)
    with mp.workdps(DEFAULT_DPS):
        return 1.0 if n < 2 else finite_float(mp.power(2*n, mp.mpf(n-1)/2))


def geometric_center_k1(n):
    """The corrected endpoint decomposition reproduces basic Stirling."""
    return stirling_approx(n)


def geometric_center_k5(n):
    return stirling_approx_k5(n)


def mvdc_center_main(n):
    return stirling_approx(n)


def mvdc_k5(n):
    return factorial_mvdc(n, 5)


def arithmetic_center_k3(n):
    """Arithmetic-centre bookkeeping with the first log-Stirling correction.

    The centre and residual are evaluated consistently; this is a rewrite of
    that standard approximation, not an independently more accurate formula.
    """
    return factorial_mvdc(n, 1)


@lru_cache(maxsize=16)
def center_model(order=5, dps=DEFAULT_DPS):
    return fit_log_center(TRAINING_NS, lambda n: mp.loggamma(n+1),
                          factorial_main_log_mp, order, dps=dps)


@lru_cache(maxsize=8)
def cascade_model(dps=DEFAULT_DPS):
    return fit_log_cascade(TRAINING_NS, lambda n: mp.loggamma(n+1),
                           factorial_main_log_mp, 3, 7, dps=dps)


def mvdc_cascade2(n):
    integer(n)
    return 1.0 if n < 2 else finite_float(cascade_model().value(n))


def compare_methods(values):
    with mp.workdps(DEFAULT_DPS):
        print("Factorial relative errors: analytic log-Stirling (5), fitted centre (5), cascade (3+7)")
        print("The analytic MVDC and log-Stirling formulas are identical for this centre.")
        for n in values:
            reference = exact_factorial(n)
            fitted = mp.mpf(1) if n < 2 else center_model().value(n)
            cascade = mp.mpf(1) if n < 2 else cascade_model().value(n)
            errors = [relative_error(factorial_mvdc_mp(n, order=5), reference),
                      relative_error(fitted, reference), relative_error(cascade, reference)]
            print(f"{n:5d} " + " ".join(f"{mp.nstr(e, 10):>16}" for e in errors))


if __name__ == "__main__":
    compare_methods((2, 3, 5, 10, 17, 25, 37, 50, 100))
