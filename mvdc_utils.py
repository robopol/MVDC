"""Stable centre decomposition, bounded factor moments, and residual fitting.

The geometric mean minimizes squared log deviations and evaluates the full
product. A fast fitted centre must instead use a separately specified main
term. Residual fits use scaled inverse powers, high precision and Householder
QR; they never solve normal equations or manufacture variance shifts.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math
import sys
from typing import Callable, Iterable

import mpmath as mp

DEFAULT_DPS = 80
MAX_ORDER = 20


def integer(value, name="n", minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def order_value(order, maximum=MAX_ORDER):
    integer(order, "order")
    if order > maximum:
        raise ValueError(f"order must be <= {maximum}")
    return order


def precision(dps):
    integer(dps, "dps", 30)
    return dps


def number(value):
    if isinstance(value, Fraction):
        return mp.mpf(value.numerator) / value.denominator
    return mp.mpf(value)


def positive(value, name="value"):
    value = number(value)
    if not mp.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def mvdc_generic_center(factors: Iterable[float], order: int = 2):
    """Return (geometric centre, log product) using compensated summation.

    ``order`` is retained for compatibility. It does not change the unique
    log-least-squares centre. Empty products have no uniquely defined centre.
    """
    integer(order, "order", 1)
    values = [float(value) for value in factors]
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("factors must be a nonempty sequence of finite positive values")
    log_product = math.fsum(math.log(x) for x in values)
    mean = log_product / len(values)
    return math.exp(mean), log_product


def log_product(factors, dps=DEFAULT_DPS):
    """Return the finite product's logarithm, including the empty product."""
    with mp.workdps(precision(dps)):
        return mp.fsum(mp.log(positive(value, "factor")) for value in factors)


def factor_center(factors, criterion="minimax", dps=DEFAULT_DPS):
    """Choose an arithmetic, geometric, or minimax relative-deviation centre.

    (min(a)+max(a))/2 minimizes max(abs(a/k-1)) over positive k and gives
    a convergent log(1+x) series for every finite positive factor set.
    """
    with mp.workdps(precision(dps)):
        values = [positive(x, "factor") for x in factors]
        if not values:
            raise ValueError("a centre requires at least one factor")
        if criterion == "minimax":
            return (min(values) + max(values)) / 2
        if criterion == "arithmetic":
            return mp.fsum(values) / len(values)
        if criterion == "geometric":
            return mp.exp(mp.fsum(mp.log(x) for x in values) / len(values))
        raise ValueError("unknown centre criterion")


def factor_moment_log(factors, center=None, order=5, dps=DEFAULT_DPS):
    """Return (log-product approximation, rigorous truncation-error bound).

    The bound concerns Taylor truncation in exact arithmetic. Numerical
    roundoff is controlled separately by dps.
    """
    order_value(order)
    with mp.workdps(precision(dps)):
        values = [positive(x, "factor") for x in factors]
        if not values:
            return mp.mpf(0), mp.mpf(0)
        k = factor_center(values, dps=dps) if center is None else positive(center, "center")
        xs = [value / k - 1 for value in values]
        if max(abs(x) for x in xs) >= 1:
            raise ValueError("factor moments require max(abs(a_i/center-1)) < 1")
        correction = mp.fsum(
            (-1) ** (r - 1) * mp.fsum(x**r for x in xs) / r
            for r in range(1, order + 1)
        )
        bound = mp.fsum(abs(x)**(order + 1) / ((order + 1) * (1 - abs(x))) for x in xs)
        return len(values) * mp.log(k) + correction, bound


@dataclass(frozen=True)
class InversePowerFit:
    coefficients: tuple
    scale: object
    shift: object
    training_ns: tuple[int, ...]
    dps: int
    max_training_error: object
    constant: bool = False

    def __call__(self, n):
        integer(n, minimum=1)
        with mp.workdps(self.dps):
            denominator = positive(number(n) + self.shift, "n+shift")
            return mp.polyval(tuple(reversed(self.coefficients)), self.scale / denominator)

    def inverse_coefficients(self):
        """Coefficients of powers 1/(n+shift), including the intercept."""
        with mp.workdps(self.dps):
            return tuple(value * self.scale**k for k, value in enumerate(self.coefficients))

    def evaluate_float(self, n):
        """Evaluate the fitted polynomial using binary64 operations only."""
        integer(n, minimum=1)
        denominator = n + float(self.shift)
        if denominator <= 0:
            raise ValueError("n+shift must be positive")
        x = float(self.scale)/denominator
        value = 0.0
        for coefficient in reversed(self.coefficients):
            value = value*x + float(coefficient)
        return value

    def extrapolates(self, n):
        return not min(self.training_ns) <= n <= max(self.training_ns)


def fit_inverse_powers(ns, values, degree=5, *, shift=0, dps=DEFAULT_DPS, noise_floor=None):
    """Fit a callable or observations using scaled inverse powers and QR.

    High-precision callables are evaluated inside workdps. Binary64 input
    observations are never treated as 80-digit data. Near-constant residuals
    become a constant model instead of a high-degree noise fit.
    """
    order_value(degree)
    ns = tuple(integer(n, minimum=1) for n in ns)
    if len(set(ns)) != len(ns) or len(ns) < degree + 1:
        raise ValueError("need at least degree+1 distinct positive training points")
    with mp.workdps(precision(dps)):
        shift = number(shift)
        denominators = [positive(number(n) + shift, "n+shift") for n in ns]
        scale = min(denominators)
        raw = [values(n) for n in ns] if callable(values) else list(values)
        if len(raw) != len(ns):
            raise ValueError("observation and training-point counts differ")
        ys = [number(y) for y in raw]
        if any(not mp.isfinite(y) for y in ys):
            raise ValueError("observations must be finite")
        magnitude = max(mp.mpf(1), max(abs(y) for y in ys))
        floor = mp.power(10, -dps + 12) * magnitude
        if any(isinstance(y, float) for y in raw):
            floor = max(floor, 32 * mp.mpf(2)**-52 * magnitude)
        if noise_floor is not None:
            specified = number(noise_floor)
            if not mp.isfinite(specified) or specified < 0:
                raise ValueError("noise_floor must be finite and nonnegative")
            floor = max(floor, specified)
        if max(ys) - min(ys) <= floor:
            mean = mp.fsum(ys) / len(ys)
            return InversePowerFit((mean,), scale, shift, ns, dps,
                                   max(abs(y - mean) for y in ys), True)
        matrix = mp.matrix([[(scale / denominator)**k for k in range(degree + 1)]
                            for denominator in denominators])
        coefficients, _ = mp.qr_solve(matrix, mp.matrix(ys))
        coefficients = tuple(coefficients)
        residual = max(abs(mp.polyval(tuple(reversed(coefficients)), scale / denominator) - y)
                       for denominator, y in zip(denominators, ys))
        return InversePowerFit(coefficients, scale, shift, ns, dps, residual)


@dataclass(frozen=True)
class LogCenterModel:
    main_log: Callable
    layers: tuple[InversePowerFit, ...]
    dps: int

    def log_value(self, n):
        integer(n, minimum=1)
        with mp.workdps(self.dps):
            return number(self.main_log(n)) + mp.fsum(layer(n) for layer in self.layers)

    def value(self, n):
        with mp.workdps(self.dps):
            return mp.exp(self.log_value(n))

    def log_center(self, n, factor_count=None):
        count = n if factor_count is None else integer(factor_count, "factor_count", 1)
        with mp.workdps(self.dps):
            return self.log_value(n) / count


def fit_log_center(ns, exact_log, main_log, degree=5, *, shift=0, dps=DEFAULT_DPS):
    fit = fit_inverse_powers(ns, lambda n: exact_log(n) - main_log(n), degree,
                             shift=shift, dps=dps)
    return LogCenterModel(main_log, (fit,), dps)


def fit_log_cascade(ns, exact_log, main_log, first_degree=3, second_degree=7, *, shift=0, dps=DEFAULT_DPS):
    """Fit the second layer to the first layer's actual remaining residual."""
    first = fit_log_center(ns, exact_log, main_log, first_degree, shift=shift, dps=dps)
    second = fit_inverse_powers(ns, lambda n: exact_log(n) - first.log_value(n),
                                second_degree, shift=shift, dps=dps)
    return LogCenterModel(main_log, (*first.layers, second), dps)


@lru_cache(maxsize=None)
def bernoulli_numbers(max_order):
    integer(max_order, "max_order")
    values = [Fraction(1)]
    for n in range(1, max_order + 1):
        values.append(-sum(Fraction(math.comb(n + 1, k)) * values[k] for k in range(n)) / (n + 1))
    return tuple(values)


@lru_cache(maxsize=None)
def central_binomial_log_coefficients(order):
    order_value(order)
    result = [Fraction(0)] * (order + 1)
    bernoulli = bernoulli_numbers(order + 1)
    for power in range(1, order + 1, 2):
        j = (power + 1) // 2
        result[power] = bernoulli[2*j] * (Fraction(1, 2**power) - 2) / (2*j * power)
    return tuple(result)


def exp_series(log_coefficients):
    """Exact formal exponential; the input has zero constant coefficient."""
    if log_coefficients[0] != 0:
        raise ValueError("the formal log series must have zero constant term")
    result = [Fraction(1)]
    for n in range(1, len(log_coefficients)):
        result.append(sum(k * log_coefficients[k] * result[n-k] for k in range(1, n+1)) / n)
    return tuple(result)


@lru_cache(maxsize=None)
def central_binomial_coefficients(order=5):
    return exp_series(central_binomial_log_coefficients(order))


@lru_cache(maxsize=None)
def wallis_coefficients(order=5):
    log_c = central_binomial_log_coefficients(order)
    log_w = [Fraction(0)] + [Fraction((-1)**k, k * 2**k) - 2*log_c[k]
                             for k in range(1, order + 1)]
    return exp_series(log_w)


def inverse_series(coefficients, n):
    return mp.polyval(tuple(number(c) for c in reversed(coefficients)), 1 / positive(n, "n"))


def finite_float(value):
    """Convert a high-precision value to binary64, saturating overflow."""
    if abs(value) > sys.float_info.max:
        return math.copysign(math.inf, float(mp.sign(value)))
    return float(value)


def relative_error(value, reference):
    if reference == 0:
        raise ValueError("relative error requires a nonzero reference")
    return abs(number(value) / number(reference) - 1)


@lru_cache(maxsize=8)
def primes_up_to(limit):
    integer(limit, "limit")
    if limit < 2:
        return ()
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[:2] = b"\x00\x00"
    for p in range(2, math.isqrt(limit) + 1):
        if sieve[p]:
            start = p*p
            sieve[start:limit+1:p] = b"\x00" * ((limit-start)//p + 1)
    return tuple(i for i in range(2, limit+1) if sieve[i])
