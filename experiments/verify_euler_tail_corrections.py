"""Independent checks for the Euler-product values used in the MVDC note.

The script deliberately separates three quantities:

1. the exact finite prime block,
2. the prime-density integral used as a numerical tail model, and
3. the logarithmic central-moment diagnostic from the earlier experiment.

It requires mpmath but does not require SymPy.
"""

from __future__ import annotations

import mpmath as mp


def primes_up_to(limit: int) -> list[int]:
    """Return all primes not exceeding limit using an Eratosthenes sieve."""

    if limit < 2:
        return []

    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    upper = int(limit**0.5)
    for prime in range(2, upper + 1):
        if not sieve[prime]:
            continue
        start = prime * prime
        count = (limit - start) // prime + 1
        sieve[start : limit + 1 : prime] = b"\x00" * count
    return [value for value, is_prime in enumerate(sieve) if is_prime]


def zeta_log_factor(prime: int) -> mp.mpf:
    """Return -log(1-p^-2) at high precision."""

    p = mp.mpf(prime)
    return -mp.log1p(-(p**-2))


def chi_mod3(value: int) -> int:
    """Return the nontrivial real Dirichlet character modulo 3."""

    residue = value % 3
    if residue == 0:
        return 0
    return 1 if residue == 1 else -1


def dirichlet_log_factor(prime: int) -> mp.mpf:
    """Return -log(1-chi_3(p)p^-2) at high precision."""

    character = chi_mod3(prime)
    p = mp.mpf(prime)
    return -mp.log1p(-character * p**-2)


def old_log_moment_diagnostic(values: list[mp.mpf], max_order: int = 6) -> mp.mpf:
    """Evaluate the extra term used by the earlier Euler experiment.

    This quantity is reported only as a diagnostic.  It is not added to
    the exact logarithm of the finite block.
    """

    count = len(values)
    mean = mp.fsum(values) / count
    centered = [value - mean for value in values]
    return mp.fsum(
        (-1) ** (order - 1)
        * mp.fsum(value**order for value in centered)
        / (order * count ** (order - 1))
        for order in range(2, max_order + 1)
    )


def verify_zeta_tail(primes: list[int]) -> None:
    """Check the zeta-tail table and isolate the old moment diagnostic."""

    log_zeta_two = mp.log(mp.pi**2 / 6)
    expected = {
        1_000: (
            mp.mpf("1.26989814755683e-4"),
            mp.mpf("1.27049267685187e-4"),
        ),
        5_000: (
            mp.mpf("2.10858622405e-5"),
            mp.mpf("2.10884676662e-5"),
        ),
        10_000: (
            mp.mpf("9.81570113247e-6"),
            mp.mpf("9.81712884998e-6"),
        ),
    }

    print("Zeta Euler tail at s=2")
    print("N       exact tail           block + integral     error                old moment")
    for cutoff, (expected_tail, expected_model) in expected.items():
        model_limit = 10 * cutoff
        head = [prime for prime in primes if prime <= cutoff]
        block_primes = [prime for prime in primes if cutoff < prime <= model_limit]

        head_log = mp.fsum(zeta_log_factor(prime) for prime in head)
        block_values = [zeta_log_factor(prime) for prime in block_primes]
        exact_tail = log_zeta_two - head_log
        finite_block = mp.fsum(block_values)
        integral_model = mp.e1(mp.log(model_limit))
        approximation = finite_block + integral_model
        error = approximation - exact_tail
        old_moment = old_log_moment_diagnostic(block_values)

        assert mp.almosteq(exact_tail, expected_tail, rel_eps=mp.mpf("2e-11"))
        assert mp.almosteq(approximation, expected_model, rel_eps=mp.mpf("2e-11"))

        print(
            f"{cutoff:<7d} "
            f"{mp.nstr(exact_tail, 13):<20} "
            f"{mp.nstr(approximation, 13):<20} "
            f"{mp.nstr(error, 6):<20} "
            f"{mp.nstr(old_moment, 6)}"
        )


def verify_dirichlet_tail(primes: list[int]) -> None:
    """Check the real character modulo 3 example independently."""

    one_third = mp.mpf(1) / 3
    two_thirds = mp.mpf(2) / 3
    l_value = (mp.zeta(2, one_third) - mp.zeta(2, two_thirds)) / 9
    log_l_value = mp.log(l_value)

    cutoff = 1_000
    model_limit = 1_000_000
    head_log = mp.fsum(
        dirichlet_log_factor(prime)
        for prime in primes
        if prime <= cutoff and prime != 3
    )
    finite_block = mp.fsum(
        dirichlet_log_factor(prime)
        for prime in primes
        if cutoff < prime <= model_limit and prime != 3
    )
    exact_tail = log_l_value - head_log
    omitted_tail = exact_tail - finite_block

    assert mp.almosteq(
        exact_tail,
        mp.mpf("-5.61728190475735e-8"),
        rel_eps=mp.mpf("2e-14"),
    )
    assert mp.almosteq(
        finite_block,
        mp.mpf("-5.61013233973667e-8"),
        rel_eps=mp.mpf("2e-14"),
    )

    print("\nDirichlet L tail for chi_3 at s=2")
    print(f"exact missing tail : {mp.nstr(exact_tail, 16)}")
    print(f"exact finite block : {mp.nstr(finite_block, 16)}")
    print(f"omitted tail       : {mp.nstr(omitted_tail, 16)}")


def main() -> None:
    """Run all high-precision checks."""

    mp.mp.dps = 70
    primes = primes_up_to(1_000_000)
    verify_zeta_tail(primes)
    verify_dirichlet_tail(primes)
    print("\nEuler-product checks passed")


if __name__ == "__main__":
    main()
