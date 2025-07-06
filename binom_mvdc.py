import math
import sys
from typing import List
from mvdc_utils import mvdc_generic_center

# --- Exact central binomial coefficient ---

# Exact central binomial coefficient – returns (float) if small else uses exp(log).


def binom_exact(n: int) -> float:
    """Return C(2n, n) as float (may overflow for very large n)."""
    try:
        return float(math.comb(2 * n, n))
    except OverflowError:
        log_val = math.lgamma(2 * n + 1) - 2 * math.lgamma(n + 1)
        return math.exp(log_val)

# --- MVDC main term via generic centre (order 2) ---


def mvdc_main(n: int) -> float:
    """Main MVDC term for central binomial coefficient using generic centre."""
    # factors representation of C(2n,n) = ∏_{i=1}^{n} (n+i)/i
    factors = [(n + i) / i for i in range(1, n + 1)]
    _, ln_H = mvdc_generic_center(factors, order=2)
    return math.exp(ln_H)

# --- Fit coefficients C0..C_order so that exact/H ≈ C0*(1 + C1/n + C2/n^2 + …) ---

def fit_mvdc_coeffs(order: int = 5, start: int = 50, stop: int = 500, step: int = 10) -> List[float]:
    rows = []
    y = []
    for n in range(start, stop + 1, step):
        # work in log-domain to avoid overflow
        log_exact = math.lgamma(2 * n + 1) - 2 * math.lgamma(n + 1)
        _, ln_H = mvdc_generic_center([(n + i) / i for i in range(1, n + 1)], order=2)
        ratio = math.exp(log_exact - ln_H)
        row = [1.0]  # intercept for C0
        for k in range(1, order + 1):
            row.append(1.0 / (n ** k))
        rows.append(row)
        y.append(ratio)

    m = order + 1
    xtx = [[0.0 for _ in range(m)] for _ in range(m)]
    xty = [0.0 for _ in range(m)]
    for row, yi in zip(rows, y):
        for i in range(m):
            xty[i] += row[i] * yi
            for j in range(m):
                xtx[i][j] += row[i] * row[j]

    # Augment matrix
    for i in range(m):
        xtx[i].append(xty[i])

    # Gaussian elimination
    for i in range(m):
        pivot = xtx[i][i]
        for j in range(i, m + 1):
            xtx[i][j] /= pivot
        for k in range(m):
            if k == i:
                continue
            factor = xtx[k][i]
            for j in range(i, m + 1):
                xtx[k][j] -= factor * xtx[i][j]

    coeffs = [xtx[i][m] for i in range(m)]
    # Normalize C0 to remain separate; subsequent Ck are relative (divide by C0)
    C0 = coeffs[0]
    rel_coeffs = [C0]
    for k in range(1, m):
        rel_coeffs.append(coeffs[k] / C0)
    return rel_coeffs

COEFFS = fit_mvdc_coeffs(order=5)

# --- Fit coefficients for log-residual (second MVDC level) ---


def fit_log_coeffs(order: int = 5, start: int = 50, stop: int = 500, step: int = 10) -> List[float]:
    rows, y = [], []
    for n in range(start, stop + 1, step):
        log_exact = math.lgamma(2 * n + 1) - 2 * math.lgamma(n + 1)
        _, ln_H = mvdc_generic_center([(n + i) / i for i in range(1, n + 1)], order=2)
        log_r = log_exact - ln_H
        row = [1.0]
        for k in range(1, order + 1):
            row.append(1.0 / (n ** k))
        rows.append(row)
        y.append(log_r)

    m = order + 1
    xtx = [[0.0 for _ in range(m)] for _ in range(m)]
    xty = [0.0 for _ in range(m)]
    for row, yi in zip(rows, y):
        for i in range(m):
            xty[i] += row[i] * yi
            for j in range(m):
                xtx[i][j] += row[i] * row[j]

    for i in range(m):
        xtx[i].append(xty[i])
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

    coeffs = [xtx[i][m] for i in range(m)]
    return coeffs  # includes C0(log)


LOG_COEFFS = fit_log_coeffs(order=5)


# --- Two-level cascade approximation ---


def mvdc_cascade2(n: int) -> float:
    """Two-level MVDC approximation computed in log-domain to avoid overflow."""
    # Build factors on the fly to get ln_H directly
    factors = [(n + i) / i for i in range(1, n + 1)]
    _, ln_H = mvdc_generic_center(factors, order=2)

    log_corr = sum(LOG_COEFFS[k] / (n ** k) for k in range(len(LOG_COEFFS)))
    total_log = ln_H + log_corr

    if total_log > math.log(sys.float_info.max):
        return float("inf")

    return math.exp(total_log)

# Classical asymptotic expansion coefficients up to 1/n^5 (from literature)
A1 = -1 / 8
A2 = 1 / 128
A3 = 5 / 3072
A4 = -7 / 131072  # note sign difference from earlier attempt
A5 = 35 / 3932160

PI = math.pi


def classical_binom(n: int, include_terms: int = 5) -> float:
    corr = 1.0
    if include_terms >= 1:
        corr += A1 / n
    if include_terms >= 2:
        corr += A2 / n ** 2
    if include_terms >= 3:
        corr += A3 / n ** 3
    if include_terms >= 4:
        corr += A4 / n ** 4
    if include_terms >= 5:
        corr += A5 / n ** 5
    return (4.0 ** n) / math.sqrt(PI * n) * corr

# Comparison table

def compare(values: List[int]):
    print("Central binomial coefficient C(2n, n) – MVDC vs classical expansions. Values are absolute errors unless noted.")
    print("-" * 260)
    header = f"{'n':>6} | {'Exact':>35} | {'H':>35} | {'H+5':>35} | {'Cascade2':>35} | {'Class.(5)':>35} | {'Err Cas2':>15} | {'Err Cl':>15}"
    print(header)
    print("-" * 260)

    for n in values:
        exact = binom_exact(n)
        H = mvdc_main(n)

        corr5 = sum(COEFFS[k] / (n ** k) for k in range(0, 6))

        mvdc5 = H * corr5
        cas2 = mvdc_cascade2(n)
        class5 = classical_binom(n)

        err_cas2 = abs(cas2 - exact)
        err_class = abs(class5 - exact)

        print(
            f"{n:>6d} | {exact:>35.15e} | {H:>35.15e} | {mvdc5:>35.15e} | {cas2:>35.15e} | {class5:>35.15e} | {err_cas2:>15.4e} | {err_class:>15.4e}"
        )

    print("-" * 260)


if __name__ == "__main__":
    Ns = [2, 5, 10, 20, 50, 100, 200, 500]
    compare(Ns) 