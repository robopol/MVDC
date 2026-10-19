"""Central-binomial asymptotics with exact rational correction coefficients.

C(2n,n) ~ 4**n/sqrt(pi*n) * (1 - 1/(8n) + 1/(128n**2)
    + 5/(1024n**3) - 21/(32768n**4) - 399/(262144n**5) + ...).
Coefficients are generated from the classical log-Stirling expansion.
"""
import math
import mpmath as mp
from mvdc_utils import (DEFAULT_DPS, central_binomial_coefficients, finite_float,
                        integer, inverse_series, order_value, precision, relative_error)

COEFFS = central_binomial_coefficients(5)
A1, A2, A3, A4, A5 = COEFFS[1:]


def binom_exact(n):
    integer(n)
    return math.comb(2*n, n)


def binom_mvdc_mp(n, *, order=5, dps=DEFAULT_DPS):
    integer(n)
    order_value(order)
    with mp.workdps(precision(dps)):
        if n == 0:
            return mp.mpf(1)
        return mp.power(4, n) / mp.sqrt(mp.pi*n) * inverse_series(central_binomial_coefficients(order), n)


def binom_mvdc(n, order=5):
    """Binary64 result; overflow is represented by inf. Use mp for large n."""
    return finite_float(binom_mvdc_mp(n, order=order))


def main():
    with mp.workdps(DEFAULT_DPS):
        print("Central binomial coefficient: analytic five-term relative error")
        for n in (5, 10, 20, 50, 100, 500, 1000):
            error = relative_error(binom_mvdc_mp(n), binom_exact(n))
            print(f"{n:6d} {mp.nstr(error, 12)}")


if __name__ == "__main__":
    main()
