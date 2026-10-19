"""Factorial centre with classical Bernoulli log corrections and Ramanujan reference."""
import math
import mpmath as mp
from mvdc_utils import (DEFAULT_DPS, bernoulli_numbers, finite_float, integer,
                        number, order_value, precision, relative_error)


def factorial_main_log_mp(n):
    integer(n, minimum=1)
    return n * (mp.log(n) - 1) + mp.log(2*mp.pi*n) / 2


def factorial_log_correction_mp(n, order=4):
    integer(n, minimum=1)
    order_value(order)
    bernoulli = bernoulli_numbers(2*order)
    return mp.fsum(number(bernoulli[2*k]) / (2*k*(2*k-1)*mp.mpf(n)**(2*k-1))
                   for k in range(1, order+1))


def log_factorial_mvdc_mp(n, *, order=4, dps=DEFAULT_DPS):
    integer(n)
    order_value(order)
    with mp.workdps(precision(dps)):
        if n < 2:
            return mp.mpf(0)
        return factorial_main_log_mp(n) + factorial_log_correction_mp(n, order)


def factorial_mvdc_mp(n, *, order=4, dps=DEFAULT_DPS):
    with mp.workdps(precision(dps)):
        return mp.exp(log_factorial_mvdc_mp(n, order=order, dps=dps))


def factorial_mvdc(n, order=4):
    """Return binary64, or inf when n! exceeds its range."""
    return finite_float(factorial_mvdc_mp(n, order=order))


def factorial_ramanujan_mp(n, dps=DEFAULT_DPS):
    """The sixth-root formula; its leading relative error is O(n^-4)."""
    integer(n)
    with mp.workdps(precision(dps)):
        if n < 2:
            return mp.mpf(1)
        x = mp.mpf(n)
        log_value = (mp.log(mp.pi)/2 + mp.log(8*x**3 + 4*x**2 + x + mp.mpf(1)/30)/6
                     + x*mp.log(x) - x)
        return mp.exp(log_value)


def factorial_ramanujan(n):
    return finite_float(factorial_ramanujan_mp(n))


def _mvdc_main_log(n):
    integer(n)
    with mp.workdps(DEFAULT_DPS):
        return 0.0 if n < 2 else float(factorial_main_log_mp(n))


def _stirling_log_correction(n, order=4):
    with mp.workdps(DEFAULT_DPS):
        return float(factorial_log_correction_mp(n, order))


def main():
    with mp.workdps(DEFAULT_DPS):
        print("Factorial relative error: log-Stirling through n^-7 and Ramanujan sixth root")
        for n in (5, 10, 20, 50, 100, 500, 1000):
            reference = mp.mpf(math.factorial(n))
            e1 = relative_error(factorial_mvdc_mp(n), reference)
            e2 = relative_error(factorial_ramanujan_mp(n), reference)
            print(f"{n:6d} {mp.nstr(e1, 12):>18} {mp.nstr(e2, 12):>18}")


if __name__ == "__main__":
    main()
