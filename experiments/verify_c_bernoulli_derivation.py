"""Exact and numerical checks for the corrected C-Bernoulli derivation.

The script uses only the Python standard library.  All algebraic checks
are performed with Fraction, so they do not depend on floating-point
rounding or a computer algebra package.
"""

from __future__ import annotations

import math
from fractions import Fraction


def bernoulli_numbers(max_order: int) -> list[Fraction]:
    """Return B_0, ..., B_max_order with the convention B_1 = -1/2."""

    values = [Fraction(1)]
    for order in range(1, max_order + 1):
        subtotal = sum(
            Fraction(math.comb(order + 1, index)) * values[index]
            for index in range(order)
        )
        values.append(-subtotal / Fraction(order + 1))
    return values


def bernoulli_polynomial(
    order: int,
    x: Fraction,
    bernoulli: list[Fraction],
) -> Fraction:
    """Evaluate the ordinary Bernoulli polynomial B_order(x)."""

    return sum(
        Fraction(math.comb(order, index))
        * bernoulli[index]
        * x ** (order - index)
        for index in range(order + 1)
    )


def central_bernoulli_polynomial(
    order: int,
    x: Fraction,
    bernoulli: list[Fraction],
) -> Fraction:
    """Evaluate B_order^c(x) = 2^order B_order((x + 1) / 2)."""

    return Fraction(2**order) * bernoulli_polynomial(
        order,
        (x + 1) / 2,
        bernoulli,
    )


def c_coefficient(n: int, order: int) -> Fraction:
    """Return C_order(n) from the centered finite-index definition."""

    midpoint = Fraction(n + 1, 2)
    power_sum = sum((Fraction(index) - midpoint) ** order for index in range(1, n + 1))
    return power_sum / math.factorial(order)


def c_from_central_bernoulli(
    n: int,
    order: int,
    bernoulli: list[Fraction],
) -> Fraction:
    """Return C_order(n) from the central Bernoulli polynomial formula."""

    numerator = central_bernoulli_polynomial(
        order + 1,
        Fraction(n),
        bernoulli,
    ) - central_bernoulli_polynomial(
        order + 1,
        Fraction(-n),
        bernoulli,
    )
    denominator = 2 ** (order + 1) * (order + 1) * math.factorial(order)
    return numerator / denominator


def verify_c_bernoulli_bridge() -> None:
    """Check the Faulhaber/central-Bernoulli identity exactly."""

    max_order = 12
    bernoulli = bernoulli_numbers(max_order + 1)
    for n in range(1, 15):
        for order in range(max_order + 1):
            direct = c_coefficient(n, order)
            via_bernoulli = c_from_central_bernoulli(n, order, bernoulli)
            assert direct == via_bernoulli, (n, order, direct, via_bernoulli)

            if order % 2 == 1:
                assert direct == 0, (n, order, direct)

        assert c_coefficient(n, 2) == Fraction(n * (n * n - 1), 24)
        assert c_coefficient(n, 4) == Fraction(
            n * (n * n - 1) * (3 * n * n - 7),
            5760,
        )


def verify_generating_function() -> None:
    """Check sinh(t/2) G_n(t) = sinh(nt/2) coefficient by coefficient."""

    max_degree = 14
    for n in range(1, 15):
        coefficients = [c_coefficient(n, order) for order in range(max_degree + 1)]
        for degree in range(max_degree + 1):
            left = Fraction(0)
            for odd_degree in range(1, degree + 1, 2):
                sinh_coefficient = Fraction(
                    1,
                    2**odd_degree * math.factorial(odd_degree),
                )
                left += sinh_coefficient * coefficients[degree - odd_degree]

            right = (
                Fraction(n**degree, 2**degree * math.factorial(degree))
                if degree % 2 == 1
                else Fraction(0)
            )
            assert left == right, (n, degree, left, right)


def derivative_at(
    polynomial: dict[int, Fraction],
    order: int,
    x: Fraction,
) -> Fraction:
    """Evaluate a derivative of a sparse polynomial exactly."""

    result = Fraction(0)
    for degree, coefficient in polynomial.items():
        if degree < order:
            continue
        falling = math.factorial(degree) // math.factorial(degree - order)
        result += coefficient * falling * x ** (degree - order)
    return result


def polynomial_value(polynomial: dict[int, Fraction], x: Fraction) -> Fraction:
    """Evaluate a sparse polynomial exactly."""

    return sum(coefficient * x**degree for degree, coefficient in polynomial.items())


def verify_midpoint_expansion() -> None:
    """Check the midpoint formula exactly for a non-symmetric polynomial."""

    polynomial = {
        8: Fraction(3),
        5: Fraction(-2),
        2: Fraction(7),
        0: Fraction(-11),
    }
    degree = max(polynomial)

    for n in range(1, 15):
        midpoint = Fraction(n + 1, 2)
        direct = sum(
            polynomial_value(polynomial, Fraction(index))
            for index in range(1, n + 1)
        )
        expanded = sum(
            derivative_at(polynomial, order, midpoint) * c_coefficient(n, order)
            for order in range(0, degree + 1, 2)
        )
        assert direct == expanded, (n, direct, expanded)


def verify_factor_moment_series() -> None:
    """Check the convergent factor-moment formula on a finite product."""

    factors = [2.0, 3.0, 4.0]
    center = sum(factors) / len(factors)
    deviations = [factor / center - 1.0 for factor in factors]
    exact = math.log(math.prod(factors) / center ** len(factors))

    approximation = 0.0
    for order in range(1, 41):
        moment = math.fsum(value**order for value in deviations)
        approximation += (-1) ** (order - 1) * moment / order

    assert abs(approximation - exact) < 1e-15, (approximation, exact)


def verify_factorial_coefficients() -> None:
    """Check the four analytic factorial correction terms numerically."""

    for n in (10, 20, 50, 100):
        inverse = 1.0 / n
        log_approximation = (
            n * (math.log(n) - 1.0)
            + 0.5 * math.log(2.0 * math.pi * n)
            + inverse / 12.0
            - inverse**3 / 360.0
            + inverse**5 / 1260.0
            - inverse**7 / 1680.0
        )
        log_error = abs(log_approximation - math.lgamma(n + 1.0))
        assert log_error < 1e-11, (n, log_error)


def main() -> None:
    verify_c_bernoulli_bridge()
    verify_generating_function()
    verify_midpoint_expansion()
    verify_factor_moment_series()
    verify_factorial_coefficients()

    print("C-Bernoulli bridge: exact checks passed")
    print("Generating function: exact checks passed")
    print("Midpoint expansion: exact checks passed")
    print("Factor-moment series: numerical check passed")
    print("Factorial coefficients: numerical checks passed")


if __name__ == "__main__":
    main()
