import math
from sympy import primerange, zeta, N
import mpmath as mp


def central_moments(primes, s, r_max):
    n = len(primes)
    logs = [-math.log(1 - p ** (-s)) for p in primes]
    mu1 = sum(logs) / n  # exact MVDC centre
    g = [val - mu1 for val in logs]
    S = {r: sum(val ** r for val in g) for r in range(2, r_max + 1)}
    return mu1, S


def mvdc_series(S, n):
    """Return dict partial_sum[r] = \sum_{k=2..r} (-1)^{k-1} S_k /(k n^{k-1})"""
    partial = {}
    running = 0.0
    for r in sorted(S):
        running += ((-1) ** (r - 1)) * S[r] / (r * n ** (r - 1))
        partial[r] = running
    return partial


def tail_integral(M, s):
    """Approximate sum_{p>M} p^{-s} / ln p by integral of x^{-s}/ln x dx."""
    f = lambda x: x ** (-s) / mp.log(x)
    return mp.quad(f, [M, mp.inf])


def tail_mvdc(N_head, s=2.0, r_max=6, tail_factor=10):
    """Compute MVDC approximation for Euler product tail between N_head and M=tail_factor*N_head."""
    M = int(N_head * tail_factor)
    primes = list(primerange(N_head + 1, M + 1))
    if not primes:
        raise ValueError("Tail prime list empty; choose larger tail_factor.")
    n_tail = len(primes)
    mu1, S = central_moments(primes, s, r_max)
    series_dict = mvdc_series(S, n_tail)
    tail_series = series_dict[r_max]
    main_term = n_tail * mu1
    tail_int = tail_integral(M, s)
    return main_term + tail_series + tail_int, main_term, series_dict, tail_int, n_tail, M


def experiment(s=2.0, N_list=None, r_max=6):
    if N_list is None:
        N_list = [1000, 10000, 30000, 100000]
    ln_zeta = math.log(float(N(zeta(s), 50)))
    print(f"ln ζ({s}) = {ln_zeta:.12g}\n")
    for N_max in N_list:
        primes = list(primerange(2, N_max + 1))
        n = len(primes)
        mu1, S = central_moments(primes, s, r_max)
        # exact partial log-product (without centring)
        partial_log = sum(-math.log(1 - p ** (-s)) for p in primes)
        true_residual = ln_zeta - partial_log
        series_dict = mvdc_series(S, n)
        print(f"N={N_max:>6}  π(N)={n:>5}  true Δ= {true_residual: .3e}")
        for r, approx in series_dict.items():
            print(f"   r≤{r}:  MVDC Δ≈ {approx: .3e}   error= {approx-true_residual: .2e}")
        print()


def experiment_tail(s=2.0, N_list=None, r_max=6, tail_factor=10):
    if N_list is None:
        N_list = [1000, 5000, 10000]
    ln_zeta = math.log(float(N(zeta(s), 50)))
    print(f"ln ζ({s}) = {ln_zeta:.12g}\n")
    for N_head in N_list:
        primes_head = list(primerange(2, N_head + 1))
        n_head = len(primes_head)
        partial_log_head = sum(-math.log(1 - p ** (-s)) for p in primes_head)
        true_delta = ln_zeta - partial_log_head
        mvdc_tail, main_term, series_dict, tail_int, n_tail, M = tail_mvdc(N_head, s, r_max, tail_factor)
        err = mvdc_tail - true_delta
        print(f"Head N={N_head:>6}  π(N)={n_head:>5} | Tail primes ({N_head}, {M}] = {n_tail}")
        print(f"   true Δ   = {true_delta: .3e}")
        for r, val in series_dict.items():
            approx = main_term + val + tail_int  # main + truncated series + integral
            print(f"   MVDC r≤{r}: {float(approx): .3e}")
        print(f"   Final MVDC+int (r≤{r_max}) = {float(mvdc_tail): .3e}\n   error = {float(err): .2e}\n")


if __name__ == "__main__":
    experiment_tail() 