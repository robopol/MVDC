"""Exact finite-index C-Bernoulli recurrence certificates.

For fixed n, G_n(t)=sum_i exp((i-(n+1)/2)*t) is annihilated by
prod_i(D-(i-(n+1)/2)). This verifies a finite exponential-sum construction.
It makes no D-finiteness claim about an infinite prime sequence or zeta.
"""
from fractions import Fraction
from math import factorial


def finite_moment_recurrence(n):
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    coefficients = [Fraction(1)]
    midpoint = Fraction(n+1, 2)
    for i in range(1, n+1):
        root = Fraction(i)-midpoint
        product = [Fraction(0)]*(len(coefficients)+1)
        for k, value in enumerate(coefficients):
            product[k] -= root*value
            product[k+1] += value
        coefficients = product
    return tuple(coefficients)


def raw_moment(n, order):
    midpoint = Fraction(n+1, 2)
    return sum((Fraction(i)-midpoint)**order for i in range(1, n+1))


def c_coefficient(n, order):
    return raw_moment(n, order)/factorial(order)


def verify_recurrence(n, checks=20):
    coefficients = finite_moment_recurrence(n)
    return all(sum(coefficient*raw_moment(n, r+j) for j, coefficient in enumerate(coefficients)) == 0
               for r in range(checks))


def main():
    for n in range(1, 13):
        if not verify_recurrence(n):
            raise ArithmeticError(f"finite-moment recurrence failed at n={n}")
        print(f"n={n:2d}: exact order-{n} finite-moment recurrence verified")
    print("This certificate concerns fixed finite index sets only.")


if __name__ == "__main__":
    main()
