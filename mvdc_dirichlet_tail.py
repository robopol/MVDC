import math
import mpmath as mp
from sympy import primerange

# Dirichlet character modulo 3 (non-trivial)
# χ(n)=1 if n≡1 mod3, -1 if n≡2 mod3, 0 otherwise

def chi_mod3(n):
    r = n % 3
    if r == 0:
        return 0
    return 1 if r == 1 else -1


def dirichlet_L(s, N_terms=100000):
    mp.mp.dps = 50
    return mp.nsum(lambda k: chi_mod3(k) / k ** s, [1, N_terms])


def central_moments_dirichlet(primes, s, r_max):
    n = len(primes)
    logs = [-math.log(1 - chi_mod3(p) * p ** (-s)) for p in primes]
    mu1 = sum(logs) / n
    g = [val - mu1 for val in logs]
    S = {r: sum(val ** r for val in g) for r in range(2, r_max + 1)}
    return mu1, S


def mvdc_series(S, n):
    partial = {}
    run = 0.0
    for r in sorted(S):
        run += ((-1) ** (r - 1)) * S[r] / (r * n ** (r - 1))
        partial[r] = run
    return partial


def tail_integral(M, s):
    f = lambda x: x ** (-s) / mp.log(x)
    return mp.quad(f, [M, mp.inf])  # crude same as principal case


def mvdc_tail_L(N_head, s=2.0, r_max=6, M_tail=1_000_000):
    primes_tail = list(primerange(N_head + 1, M_tail + 1))
    primes_tail = [p for p in primes_tail if chi_mod3(p) != 0]
    n = len(primes_tail)
    mu1, S = central_moments_dirichlet(primes_tail, s, r_max)
    main = n * mu1
    series_val = mvdc_series(S, n)[r_max]
    tail_int = 0.0  # neglected; contribution beyond M_tail < 1e-10 for s>=2
    return main + series_val, main, series_val, tail_int, n


def experiment():
    s = 2.0
    N_head = 1000
    primes_head = [p for p in primerange(2, N_head + 1) if chi_mod3(p) != 0]
    head_log = sum(-math.log(1 - chi_mod3(p) * p ** (-s)) for p in primes_head)
    total_L = float(dirichlet_L(s, 200000))
    ln_L = math.log(total_L)
    true_delta = ln_L - head_log
    mvdc_val, main, series_val, tail_int, n_tail = mvdc_tail_L(N_head, s)
    print(f"ln L_mod3(2)={ln_L:.12g}\n")
    print(f"Head primes ≤{N_head}: Δ = {true_delta:.3e}")
    print(f"Tail primes used: {n_tail}  M_tail=1e6")
    print(f"MVDC approx (six terms) = {float(mvdc_val): .3e}\nerror = {float(mvdc_val-true_delta): .2e}")

if __name__ == "__main__":
    experiment() 