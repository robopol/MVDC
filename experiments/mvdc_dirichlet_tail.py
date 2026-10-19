"""Dirichlet chi_3 tails with an infinite-series reference and honest bounds."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mpmath as mp
from mvdc_utils import DEFAULT_DPS, integer, precision, primes_up_to
from experiments.mvdc_euler_product import absolute_tail_bound, exponent, mvdc_series


def chi_mod3(n):
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError("a Dirichlet character is evaluated at an integer")
    residue = n % 3
    return 0 if residue == 0 else (1 if residue == 1 else -1)


def dirichlet_L(s, dps=DEFAULT_DPS):
    """The infinite L(s,chi_3), evaluated by the Hurwitz-zeta identity."""
    with mp.workdps(precision(dps)):
        s = exponent(s)
        return (mp.zeta(s, mp.mpf(1)/3)-mp.zeta(s, mp.mpf(2)/3))/mp.power(3, s)


def dirichlet_L_partial(s, N_terms, dps=DEFAULT_DPS):
    integer(N_terms, "N_terms", 1)
    with mp.workdps(precision(dps)):
        s = exponent(s)
        return mp.fsum(chi_mod3(k)/mp.power(k, s) for k in range(1, N_terms+1))


def log_factor(p, s):
    return -mp.log1p(-chi_mod3(p)*mp.power(p, -s))


def central_moments_dirichlet(primes, s, r_max):
    integer(r_max, "r_max", 2)
    values = [log_factor(p, exponent(s)) for p in primes]
    if not values:
        return mp.mpf(0), {r: mp.mpf(0) for r in range(2, r_max+1)}
    mean = mp.fsum(values)/len(values)
    return mean, {r: mp.fsum((v-mean)**r for v in values) for r in range(2, r_max+1)}


def mvdc_tail_L(N_head, s=2, r_max=0, M_tail=1_000_000, dps=DEFAULT_DPS):
    """Approximate the missing tail by its exact finite block.

    The remaining signed tail is omitted, not claimed to be below 1e-10.
    absolute_tail_bound(M_tail,s) provides a valid (possibly loose) bound.
    A requested historical moment diagnostic is returned but never added.
    """
    integer(N_head, "N_head", 2)
    integer(M_tail, "M_tail", N_head+1)
    integer(r_max, "r_max")
    if r_max == 1:
        raise ValueError("r_max must be zero or at least two")
    with mp.workdps(precision(dps)):
        s = exponent(s)
        block = [p for p in primes_up_to(M_tail) if p > N_head and p != 3]
        finite_block = mp.fsum(log_factor(p, s) for p in block)
        diagnostic = mp.mpf(0)
        if r_max:
            _, moments = central_moments_dirichlet(block, s, r_max)
            diagnostic = mvdc_series(moments, len(block))[r_max]
        return finite_block, finite_block, diagnostic, mp.mpf(0), len(block)


def experiment():
    with mp.workdps(DEFAULT_DPS):
        cutoff, limit, s = 1000, 1_000_000, mp.mpf(2)
        head = mp.fsum(log_factor(p, s) for p in primes_up_to(cutoff))
        exact = mp.log(dirichlet_L(s))-head
        approximation, block, _, _, count = mvdc_tail_L(cutoff, s, M_tail=limit)
        print(f"chi_3, s=2, N={cutoff}, M={limit}, block primes={count}")
        print("exact missing tail :", mp.nstr(exact, 18))
        print("exact finite block :", mp.nstr(block, 18))
        print("omitted signed tail:", mp.nstr(exact-approximation, 12))
        print("absolute tail bound:", mp.nstr(absolute_tail_bound(limit, s), 12))


if __name__ == "__main__":
    experiment()
