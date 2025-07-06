import math
from mvdc_utils import mvdc_generic_center

# Parameter s for ζ(s); choose s > 1 to ensure convergence
S = 2.0

# Generate primes up to limit using simple sieve
def primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0:2] = [False, False]
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            step = i
            sieve[i*i:n+1:step] = [False] * len(range(i*i, n+1, step))
    return [i for i, is_prime in enumerate(sieve) if is_prime]

# Precompute primes up to MAX_P once
MAX_P = 20000  # ~2262 primes
PRIMES = primes_upto(MAX_P)


def inverse_zeta_exact_log(P_idx):
    """Natural log of product_{p<=P} (1 - p^{-s}). P_idx is number of primes to include."""
    total = 0.0
    for p in PRIMES[:P_idx]:
        total += math.log1p(-p ** -S)
    return total


def mvdc_main_log(P_idx):
    factors = [1.0 - p ** -S for p in PRIMES[:P_idx]]
    _, ln_H = mvdc_generic_center(factors, order=2)
    return ln_H


def fit_coeffs(order=5, start_idx=100, stop_idx=2000, step=100):
    rows, y = [], []
    for idx in range(start_idx, stop_idx + 1, step):
        log_exact = inverse_zeta_exact_log(idx)
        ln_H = mvdc_main_log(idx)
        log_r = log_exact - ln_H
        rows.append([1.0] + [1.0 / idx ** k for k in range(1, order + 1)])
        y.append(log_r)
    m = order + 1
    xtx = [[0.0]*m for _ in range(m)]
    xty = [0.0]*m
    for row, yi in zip(rows, y):
        for i in range(m):
            xty[i] += row[i]*yi
            for j in range(m):
                xtx[i][j] += row[i]*row[j]
    # augment
    for i in range(m):
        xtx[i].append(xty[i])
    # elim
    for i in range(m):
        piv = xtx[i][i]
        for j in range(i, m+1):
            xtx[i][j] /= piv
        for k in range(m):
            if k==i:
                continue
            fac = xtx[k][i]
            for j in range(i, m+1):
                xtx[k][j] -= fac*xtx[i][j]
    return [xtx[i][m] for i in range(m)]

COEFFS = fit_coeffs()


def mvdc_poly_log(P_idx, terms):
    ln_H = mvdc_main_log(P_idx)
    log_corr = sum(COEFFS[k]/P_idx**k for k in range(1, terms+1))
    return ln_H + log_corr


def compare(indices):
    print(f"MVDC approximation of truncated Euler product ∏(1 - p^{{-s}}) for s = {S}. Values are base-10 logs.")
    print(f"{'#p':>5} | {'log10 Exact':>15} | {'log10 H':>15} | {'H+3':>15} | {'H+5':>15} | ΔH+5")
    lg10=lambda x: x/math.log(10)
    for idx in indices:
        log_exact = inverse_zeta_exact_log(idx)
        log_H = mvdc_main_log(idx)
        log_H3 = mvdc_poly_log(idx,3)
        log_H5 = mvdc_poly_log(idx,5)
        d = (log_H5-log_exact)/math.log(10)
        print(f"{idx:5d} | {lg10(log_exact):15.6f} | {lg10(log_H):15.6f} | {lg10(log_H3):15.6f} | {lg10(log_H5):15.6f} | {d:.1e}")

if __name__=='__main__':
    compare([10,50,100,200,500,1000,1500,2000]) 