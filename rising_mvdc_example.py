"""Rising factorial: analytic Gamma centre and stable residual fitting."""
from fractions import Fraction
from functools import lru_cache
import mpmath as mp
from mvdc_utils import (DEFAULT_DPS, finite_float, fit_log_center, integer,
                        number, order_value, precision, relative_error)

a = Fraction(1, 3)
TRAINING_NS = tuple(range(10, 201, 10))


def exact_log(n):
    x = number(a)
    return mp.loggamma(n+x) - mp.loggamma(x)


def main_log(n):
    x = number(a)
    z = n+x
    return (z-mp.mpf("0.5"))*mp.log(z) - z + mp.log(2*mp.pi)/2 - mp.loggamma(x)


@lru_cache(maxsize=32)
def center_model(order=5, dps=DEFAULT_DPS):
    return fit_log_center(TRAINING_NS, exact_log, main_log, order, shift=a, dps=dps)


def rising_exact(n):
    """Return a high-precision logarithm of (a)_n."""
    integer(n)
    with mp.workdps(DEFAULT_DPS):
        return exact_log(n)


def mvdc_main_log(n):
    integer(n)
    with mp.workdps(DEFAULT_DPS):
        return mp.mpf(0) if n == 0 else main_log(n)


def fit_coeffs(order=5, start=10, stop=200, step=10):
    integer(step, "step", 1)
    model = fit_log_center(range(start, stop+1, step), exact_log, main_log, order, shift=a)
    return model.layers[0].inverse_coefficients()


def mvdc_cascade_log(n, order=5, dps=DEFAULT_DPS):
    """Return the complete fitted log value, including the fitted intercept."""
    integer(n)
    order_value(order)
    with mp.workdps(precision(dps)):
        if n == 0:
            return mp.mpf(0)
        return center_model(order, dps).log_value(n)


def compare(values):
    with mp.workdps(DEFAULT_DPS):
        print("Rising factorial (1/3)_n: relative errors of independently fitted orders 3 and 5")
        for n in values:
            reference = mp.rf(number(a), n)
            errors = [relative_error(mp.exp(mvdc_cascade_log(n, order)), reference) for order in (3, 5)]
            print(f"{n:5d} {mp.nstr(errors[0], 12):>18} {mp.nstr(errors[1], 12):>18}")


if __name__ == "__main__":
    compare((1, 2, 5, 10, 15, 25, 75, 125, 200, 500))
