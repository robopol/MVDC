import math
from mvdc_utils import mvdc_generic_center


# --- Presný (konečný) Wallisov súčin ---

def wallis_exact(N: int) -> float:
    """Výpočet P_N = ∏_{n=1..N} 4n^2/(4n^2-1)."""
    prod = 1.0
    for n in range(1, N + 1):
        prod *= (4 * n * n) / (4 * n * n - 1)
    return prod

# --- MVDC main term via generic centre (order 2) ---


def mvdc_main(N: int) -> float:
    """Main MVDC term for Wallis product using generic centre of order 2."""
    factors = [(4 * n * n) / (4 * n * n - 1) for n in range(1, N + 1)]
    _, ln_H = mvdc_generic_center(factors, order=2)
    return math.exp(ln_H)

# --- Least-squares fit na koeficienty korekcie 1/N,1/N²,… ---

# Revised fitting that solves for C0 as intercept
def fit_mvdc_coeffs(order: int = 5, start: int = 5000, stop: int = 30000, step: int = 500) -> list[float]:
    """Fit coefficients C0..C_order in r(N)=C0*(1 + C1/N + ...). Return list length order+1."""
    rows = []
    y = []
    for N in range(start, stop + 1, step):
        H = mvdc_main(N)
        K = wallis_exact(N) / H  # exact ratio
        row = [1.0]
        # 1/N,1/N^2,...
        for k in range(1, order + 1):
            row.append(1.0 / (N ** k))
        rows.append(row)
        y.append(K)

    # If H is already machine-precision equal to the exact product, no corrections are needed.
    if max(abs(yi - 1.0) for yi in y) < 1e-12:
        return [1.0] + [0.0] * order

    m = order + 1
    # normal equations
    xtx = [[0.0 for _ in range(m)] for _ in range(m)]
    xty = [0.0 for _ in range(m)]
    for row, yi in zip(rows, y):
        for i in range(m):
            xty[i] += row[i] * yi
            for j in range(m):
                xtx[i][j] += row[i] * row[j]

    # augment and solve via Gaussian elimination
    for i in range(m):
        xtx[i].append(xty[i])

    for i in range(m):
        pivot = xtx[i][i]
        if abs(pivot) < 1e-18:
            continue
        for j in range(i, m + 1):
            xtx[i][j] /= pivot
        for k in range(m):
            if k == i:
                continue
            factor = xtx[k][i]
            for j in range(i, m + 1):
                xtx[k][j] -= factor * xtx[i][j]

    a = [xtx[i][m] for i in range(m)]  # coefficients in original scale
    C0 = a[0]
    coeffs = [C0]
    for k in range(1, m):
        coeffs.append(a[k] / C0)
    return coeffs


COEFFS = fit_mvdc_coeffs(order=5)

# --- Známá asymptotická expanzia (π/2 * (1 - 1/(8N) + 1/(128N^2))) ---

def wallis_asymptotic(N: int) -> float:
    return (math.pi / 2.0) * (1 - 1 / (8 * N) + 1 / (128 * N * N))

# --- Porovnávacia tabuľka ---

def compare_Wallis(values):
    print("Wallis product P_N = ∏ 4n^2/(4n^2-1) – MVDC approximation vs classical expansions.")
    print("-" * 200)
    print(f"{'N':>6} | {'Exact product':>20} | {'MVDC H':>20} | {'H+3':>20} | {'H+5':>20} | {'Asympt.':>20}")
    print("-" * 200)

    for N in values:
        exact = wallis_exact(N)

        H = mvdc_main(N)

        # Build MVDC up to 3rd and 5th order
        correction3 = 1.0
        correction5 = 1.0
        for k in range(1, 6):
            term = COEFFS[k] / (N ** k)
            correction5 += term
            if k <= 3:
                correction3 += term

        mvdc3 = H * COEFFS[0] * correction3
        mvdc5 = H * COEFFS[0] * correction5

        # Classical asymptotic up to 5th order
        a1, a2, a3, a4, a5 = (-1/8, 1/128, -5/3072, 7/98304, -35/3932160)
        asymp_correction = 1 + a1 / N + a2 / N**2 + a3 / N**3 + a4 / N**4 + a5 / N**5
        asymp5 = (math.pi / 2) * asymp_correction

        print(
            f"{N:>6d} | {exact:>20.12e} | {H:>20.12e} | {mvdc3:>20.12e} | {mvdc5:>20.12e} | {asymp5:>20.12e}"
        )

    print("-" * 200)


if __name__ == "__main__":
    Ns = [1, 2, 5, 10, 20, 50, 100, 500, 1000]
    compare_Wallis(Ns) 