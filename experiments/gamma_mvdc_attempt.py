"""Corrected Euler-constant experiment using a convergent logarithmic series.

gamma = sum_{k>=1}(1/k - log(1+1/k)). Its finite sum is H_N-log(N+1).
The former positive log-only series telescoped to log(N+1) and diverged.
The factorial-scaled coefficients below are real, not exact rational numbers.
The accelerated estimate uses the standard harmonic Euler-Maclaurin formula.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mpmath as mp
from mvdc_utils import DEFAULT_DPS, bernoulli_numbers, integer, number, order_value, precision


def coeffs_a(count, dps=DEFAULT_DPS):
    integer(count, "count")
    with mp.workdps(precision(dps)):
        return [mp.factorial(k)*(mp.mpf(1)/k-mp.log1p(mp.mpf(1)/k))
                for k in range(1, count+1)]


def gamma_partial(count, dps=DEFAULT_DPS):
    integer(count, "count", 1)
    with mp.workdps(precision(dps)):
        return mp.fsum(mp.mpf(1)/k-mp.log1p(mp.mpf(1)/k) for k in range(1, count+1))


def gamma_estimate(count, correction_order=5, dps=DEFAULT_DPS):
    integer(count, "count", 1)
    order_value(correction_order)
    with mp.workdps(precision(dps)):
        harmonic = mp.fsum(mp.mpf(1)/k for k in range(1, count+1))
        b = bernoulli_numbers(2*correction_order)
        return (harmonic-mp.log(count)-mp.mpf(1)/(2*count)
                + mp.fsum(number(b[2*j])/(2*j*mp.mpf(count)**(2*j))
                          for j in range(1, correction_order+1)))


def main():
    with mp.workdps(DEFAULT_DPS):
        print("Euler constant absolute error: convergent partial sum and EM-corrected estimate")
        for n in (10, 30, 100, 1000):
            print(n, mp.nstr(abs(gamma_partial(n)-mp.euler), 12),
                  mp.nstr(abs(gamma_estimate(n)-mp.euler), 12))


if __name__ == "__main__":
    main()
