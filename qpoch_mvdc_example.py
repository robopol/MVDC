import math
from mvdc_utils import mvdc_generic_center

# Parameters for q-Pochhammer (a;q)_N
A = 0.8   # a (must satisfy |a|<1 so that 1-aq^n>0)
Q = 0.3   # q (|q|<1)


def qpoch_exact_log(N: int) -> float:
    """Return natural log of (a;q)_N for given N."""
    total = 0.0
    for n in range(N):
        total += math.log1p(-A * (Q ** n))  # ln(1 - aq^n)
    return total


def mvdc_main_log(N: int) -> float:
    factors = [1.0 - A * (Q ** n) for n in range(N)]
    _, ln_H = mvdc_generic_center(factors, order=2)
    return ln_H


def fit_coeffs(order: int = 5, start: int = 20, stop: int = 200, step: int = 10):
    rows, y = [], []
    for N in range(start, stop + 1, step):
        log_exact = qpoch_exact_log(N)
        ln_H = mvdc_main_log(N)
        log_r = log_exact - ln_H
        rows.append([1.0] + [1.0 / N ** k for k in range(1, order + 1)])
        y.append(log_r)
    m = order + 1
    xtx = [[0.0] * m for _ in range(m)]
    xty = [0.0] * m
    for row, yi in zip(rows, y):
        for i in range(m):
            xty[i] += row[i] * yi
            for j in range(m):
                xtx[i][j] += row[i] * row[j]
    # augment
    for i in range(m):
        xtx[i].append(xty[i])
    # Gaussian elimination
    for i in range(m):
        piv = xtx[i][i]
        for j in range(i, m + 1):
            xtx[i][j] /= piv
        for k in range(m):
            if k == i:
                continue
            fac = xtx[k][i]
            for j in range(i, m + 1):
                xtx[k][j] -= fac * xtx[i][j]
    return [xtx[i][m] for i in range(m)]

COEFFS = fit_coeffs()


def mvdc_poly_log(N: int, terms: int) -> float:
    ln_H = mvdc_main_log(N)
    log_corr = sum(COEFFS[k] / N ** k for k in range(1, terms + 1))
    return ln_H + log_corr


def compare(N_values):
    print("MVDC approximation of q-Pochhammer (a;q)_N with a = 0.8, q = 0.3. Values are base-10 logs.")
    print(f"{'N':>4} | {'log10 Exact':>15} | {'log10 H':>15} | {'H+3':>15} | {'H+5':>15} | ΔH+5")
    log10 = lambda x: x / math.log(10)
    for N in N_values:
        log_exact = qpoch_exact_log(N)
        log_H = mvdc_main_log(N)
        log_H3 = mvdc_poly_log(N, 3)
        log_H5 = mvdc_poly_log(N, 5)
        delta = (log_H5 - log_exact) / math.log(10)
        print(f"{N:4d} | {log10(log_exact):15.6f} | {log10(log_H):15.6f} | {log10(log_H3):15.6f} | {log10(log_H5):15.6f} | {delta:.1e}")

if __name__ == "__main__":
    compare([10, 20, 30, 40, 60, 80, 100, 150, 200]) 