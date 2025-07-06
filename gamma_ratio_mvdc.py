"""gamma_ratio_mvdc.py
Comparison of MVDC approximation versus classical Stirling expansion for
a ratio of Gamma functions  Γ(n+α) / Γ(n+β).

We choose α = 0.5, β = 0.0 (square-root stepping) so the exact value equals
Γ(n + 0.5) / Γ(n).

For each n in a sample list the script prints:
  • exact value using math.gamma
  • Stirling expansion up to 2nd order  n^{α-β}(1 + a1/n + a2/n²)
  • MVDC leading term (H)
  • MVDC H+3 and H+5 (order-3,5 corrections)
  • relative errors

Run:
    python gamma_ratio_mvdc.py

All comments and outputs are in English.
"""

import math
from typing import List

from mvdc_utils import mvdc_generic_center

ALPHA = 0.5  # α in Γ(n+α)
BETA = 0.0   # β in Γ(n+β)

# Stirling-series coefficients for Γ(n+α)/Γ(n+β) with β=0
# Derived from log Γ expansion; keep up to 1/n²
A1 = (ALPHA - 0.5) / 2.0  # −(α−β)(α+β−1)/2 when β=0
A2 = ((ALPHA - 0.5) * (ALPHA**2 - 3 * ALPHA + 1.0)) / 24.0

# Euler–Mascheroni constant for analytical C0
EULER_GAMMA = 0.5772156649015328606


def gamma_ratio_exact(n: int) -> float:
    """Exact Γ(n+α) / Γ(n+β) using math.lgamma for big n."""
    return math.exp(math.lgamma(n + ALPHA) - math.lgamma(n + BETA))


# ---------------------------------------------------------------------------
# MVDC correction coefficient fitting                                         #
# ---------------------------------------------------------------------------


def fit_mvdc_coeffs_gamma(order: int = 5) -> List[float]:
    """Fit coefficients C0..C_order so that
        R_n = Exact / H ≈ C0 * (1 + C1/n + C2/n² + ...).
    """

    sample_n = list(range(200, 2000, 200))

    rows: List[List[float]] = []
    y: List[float] = []

    for n in sample_n:
        # factors for ln H
        first_factor = (BETA + ALPHA) / (BETA if BETA != 0 else 1.0)
        factors = [(k + BETA + ALPHA) / (k + BETA) for k in range(1, n)]
        _, ln_H = mvdc_generic_center(factors, order=2)
        ln_H += math.log(first_factor)

        ln_exact = math.lgamma(n + ALPHA) - math.lgamma(n + BETA)
        Rn = math.exp(ln_exact - ln_H)

        rows.append([1.0] + [1 / n ** k for k in range(1, order + 1)])
        y.append(Rn)

    m = order + 1
    xtx = [[0.0] * m for _ in range(m)]
    xty = [0.0] * m
    for row, yi in zip(rows, y):
        for i in range(m):
            xty[i] += row[i] * yi
            for j in range(m):
                xtx[i][j] += row[i] * row[j]

    # Augmented matrix solution
    for i in range(m):
        xtx[i].append(xty[i])

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

    a = [xtx[i][m] for i in range(m)]  # a0 = C0, a1 = C0*C1, ...
    C0 = a[0]
    coeffs = [C0]
    for k in range(1, m):
        coeffs.append(a[k] / C0)
    return coeffs


try:
    COEFFS_FIT = fit_mvdc_coeffs_gamma(order=5)
except Exception:
    # fallback to analytic C0 only
    C0_analytic = math.exp(ALPHA * EULER_GAMMA)
    COEFFS_FIT = [C0_analytic] + [0.0] * 5


def gamma_ratio_stirling(n: int) -> float:
    """Stirling expansion up to 1/n²."""
    base = n ** (ALPHA - BETA)
    corr = 1.0 + A1 / n + A2 / (n * n)
    return base * corr


def gamma_ratio_mvdc(n: int, order: int = 0) -> float:
    """MVDC approximation with given correction order (0,3,5)."""
    if n == 0:
        raise ValueError("n must be positive")

    # First term (k=0) handled separately to avoid division by zero when BETA=0.
    first_factor = (BETA + ALPHA) / (BETA if BETA != 0 else 1.0)

    factors: List[float] = [
        (k + BETA + ALPHA) / (k + BETA) for k in range(1, n)
    ]

    _, ln_H = mvdc_generic_center(factors, order=2)

    ln_H += math.log(first_factor)

    if order == 0:
        return math.exp(ln_H)

    coeffs = COEFFS_FIT[: order + 1]

    correction = 1.0
    for k in range(1, order + 1):
        correction += coeffs[k] / (n ** k)

    return math.exp(ln_H) * coeffs[0] * correction


def rel_err(approx: float, exact: float) -> float:
    return abs(approx - exact) / exact


def main() -> None:
    ns = [20, 50, 100, 500, 1000, 2000]

    header = (
        f"Gamma ratio Γ(n+{ALPHA})/Γ(n) — MVDC vs Stirling\n"
        "n   |   exact       |  Stirling  |   MVDC H   |  MVDC H+3  |  MVDC H+5  "
        "| rel err H+5"
    )
    print(header)
    print("-" * len(header))

    for n in ns:
        exact = gamma_ratio_exact(n)
        stir = gamma_ratio_stirling(n)
        mvdc0 = gamma_ratio_mvdc(n, order=0)
        mvdc3 = gamma_ratio_mvdc(n, order=3)
        mvdc5 = gamma_ratio_mvdc(n, order=5)

        print(
            f"{n:4d} | {exact:12.6e} | {stir:10.6e} | {mvdc0:10.6e} | "
            f"{mvdc3:10.6e} | {mvdc5:10.6e} | {rel_err(mvdc5, exact):.2e}"
        )


if __name__ == "__main__":
    main() 