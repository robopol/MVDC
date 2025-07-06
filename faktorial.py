import math
import sys
import statistics  # Needed for variance computation in generic MVDC centre

# --- Function definitions ---

# ---------------------------------------------------------------------------
# Generic MVDC centre selector
# ---------------------------------------------------------------------------


def mvdc_generic_center(factors, order: int = 2):
    """Return centre *k* and ln(H) for an arbitrary product using MVDC.

    Parameters
    ----------
    factors : iterable of float
        List/iterator of positive factors a_i whose product is studied.
    order : int, optional (default 2)
        1 → geometric-mean centre (first log-moment cancelled).
        2 → additionally absorb the log-variance (second moment) so that
            the residual series starts with 1/(12 n) instead of √n.

    Returns
    -------
    k : float
        The chosen centre.
    ln_H : float
        Natural logarithm of the main MVDC term H = k^m.

    Notes
    -----
    Let m = len(factors).  For order=1 the centre is simply the geometric
    mean  k₁ = exp( (1/m) Σ ln a_i ).

    For order=2 we further shift the centre by exp(σ²/2) where σ² is the
    log-variance.  This absorbs the √(2π m) term that appears in many
    asymptotic products (e.g., factorial, central binomial coefficient).
    """

    factors = list(factors)
    m = len(factors)
    if m == 0:
        raise ValueError("factors must contain at least one element")

    # First log-moment (mean)
    logs = [math.log(x) for x in factors]
    mu1 = sum(logs) / m

    k = math.exp(mu1)            # order-1 centre
    ln_H = m * mu1

    if order >= 2:
        # Second log-moment (variance)
        var = statistics.fmean((x - mu1) ** 2 for x in logs)
        k *= math.exp(var / 2.0)
        ln_H += m * var / 2.0

    return k, ln_H

def exact_factorial(n: int) -> float:
    """Return the exact value of n! using Python's built-in factorial."""
    if n < 0:
        return float('nan')  # NaN for negative input
    return float(math.factorial(n))

def stirling_approx(n: int) -> float:
    """
    Compute n! using the basic Stirling approximation:
        n! ≈ √(2π n) · (n/e)^n
    """
    if n <= 0:
        return float('nan')
    
    pi = math.pi
    e = math.e
    
    # Work in the log-domain to avoid overflow and precision loss
    log_stirling = 0.5 * math.log(2 * pi * n) + n * math.log(n) - n
    return math.exp(log_stirling)


# --- Extended Stirling approximation with 5 correction terms ---

def stirling_approx_k5(n: int) -> float:
    """Stirling series extended by 5 Bernoulli terms (up to 1/n⁹)."""
    if n < 2:
        return 1.0

    pi = math.pi
    ln_s = (
        n * math.log(n)
        - n
        + 0.5 * (math.log(2 * pi) + math.log(n))
    )

    inv_n = 1.0 / n
    inv_n2 = inv_n * inv_n

    ln_s += (
        (1.0 / 12.0) * inv_n
        - (1.0 / 360.0) * inv_n2 * inv_n
        + (1.0 / 1260.0) * inv_n2 * inv_n2 * inv_n
        - (1.0 / 1680.0) * inv_n2 * inv_n2 * inv_n2 * inv_n
        + (1.0 / 1188.0) * inv_n2 * inv_n2 * inv_n2 * inv_n2 * inv_n
    )

    return math.exp(ln_s)

def geometric_center_main(n):
    """
    Compute n! via the *geometric center* hypothesis (now mostly
    historical/illustrative).  The product P = 2·3·…·n contains n−1 terms.
    Taking k_g = √(2n) as the geometric mean of the endpoints 2 and n gives
        P ≈ k_g^(n-1).
    The routine returns this main term without any MVDC correction.
    """
    if n < 2:
        # For n=0 and n=1, return the exact value, as our method is defined from n=2
        return 1.0

    # Our new center
    k_g = math.sqrt(2 * n)
    
    # Main term of our approximation
    approximation = k_g ** (n - 1)
    
    return approximation


# --- Improved geometric method (MVDC with 1. correction term) ---

def geometric_center_k1(n):
    """
    First improved estimate based on MVDC with one correction term.

    Expressed in the log-domain to prevent overflow:
        ln n! ≈ ln(H_main) + ln(K₁)

        H_main = (√(2n))^(n-1)
        K₁     = 2 √π · n^(n/2 + 1) / (e^n 2^{n/2})

    This is essentially the classical Stirling approximation rewritten as
    "main term × correction" in the MVDC language.
    """
    if n < 2:
        return 1.0

    pi = math.pi

    # ln main term: (n-1)/2 * (ln 2 + ln n)
    ln_main = 0.5 * (n - 1) * (math.log(2) + math.log(n))

    # ln correction term based on derived formula
    ln_correction = (
        math.log(2) + 0.5 * math.log(pi)  # ln(2 * sqrt(pi))
        + (n / 2 + 1) * math.log(n)       # n^(n/2 + 1)
        - n                               # e^{-n}
        - (n / 2) * math.log(2)           # 2^{-(n/2)}
    )

    ln_estimate = ln_main + ln_correction
    return math.exp(ln_estimate)


# --- Extended geometric method with 5 correction terms ---

def geometric_center_k5(n):
    """
    Extended estimate with 5 MVDC terms (equivalent to Stirling series up to 1/n⁹).

    ln(n!) ≈ n ln n - n + 0.5 ln(2π n)
              + 1/(12 n) - 1/(360 n³) + 1/(1260 n⁵)
              - 1/(1680 n⁷) + 1/(1188 n⁹)

    For n < 2, we return the exact 1.0.
    """
    if n < 2:
        return 1.0

    pi = math.pi

    # Core Stirling expansion in log-space
    ln_s = (
        n * math.log(n)
        - n
        + 0.5 * (math.log(2 * pi) + math.log(n))
    )

    # 5 Bernoulli correction terms (B2 … B10)
    inv_n = 1.0 / n
    inv_n2 = inv_n * inv_n  # 1 / n²

    ln_s += (
        (1.0 / 12.0) * inv_n
        - (1.0 / 360.0) * inv_n2 * inv_n  # 1/n³
        + (1.0 / 1260.0) * inv_n2 * inv_n2 * inv_n  # 1/n⁵
        - (1.0 / 1680.0) * inv_n2 * inv_n2 * inv_n2 * inv_n  # 1/n⁷
        + (1.0 / 1188.0) * inv_n2 * inv_n2 * inv_n2 * inv_n2 * inv_n  # 1/n⁹
    )

    return math.exp(ln_s)

# ===================================================================
#     MVDC WITH  ENHANCED  CENTER  k = (n/e) * (2π n)^{1/(2n)}
# ===================================================================


def mvdc_center_main(n: int) -> float:
    """
    Main MVDC term for n! using an *enhanced* center

        k  =  (n / e) · (2π n)^{1/(2n)}

    so that
        ln H = n ln(n/e) + ½ ln(2π n)

    This absorbs not only the leading component n ln n − n, but also the
    square-root term ½ ln(2π n).  The residual series therefore starts at
    1/(12n), giving smaller corrections and keeping true to the MVDC idea.
    """
    if n < 2:
        return 1.0
    k = (n / math.e) * (2.0 * math.pi * n) ** (0.5 / n)
    return k ** n


# ---------------------------------------------------------------------------
# 5-term multiplicative correction  (ratio fitted on top of the main term)
# ---------------------------------------------------------------------------


def mvdc_k5(n: int) -> float:
    """MVDC with enhanced center and 5 Bernoulli correction terms (starts at 1/n)."""
    if n < 2:
        return 1.0

    H = mvdc_center_main(n)

    # ln of correction K; leading sqrt term already in H, so start at 1/(12n)
    inv_n = 1.0 / n
    inv_n2 = inv_n * inv_n
    ln_K = (
        (1.0 / 12.0) * inv_n
        - (1.0 / 360.0) * inv_n2 * inv_n               # 1/n³
        + (1.0 / 1260.0) * inv_n2 * inv_n2 * inv_n      # 1/n⁵
        - (1.0 / 1680.0) * inv_n2 * inv_n2 * inv_n2 * inv_n  # 1/n⁷
        + (1.0 / 1188.0) * inv_n2 * inv_n2 * inv_n2 * inv_n2 * inv_n  # 1/n⁹
    )

    return H * math.exp(ln_K)


# --- Cascade2 for natural center (log residual) ---


def _fit_mvdc_log(order: int = 5, start: int = 5, stop: int = 50, step: int = 5):
    rows, y = [], []
    for n in range(start, stop + 1, step):
        exact = exact_factorial(n)
        H = mvdc_center_main(n)
        ratio = exact / H
        log_r = math.log(ratio)
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

    return [xtx[i][m] for i in range(m)]


MVDC_LOG = _fit_mvdc_log()


def mvdc_cascade2(n: int) -> float:
    # ln of main term H with the enhanced center (includes sqrt term)
    log_H = n * (math.log(n) - 1.0) + 0.5 * (math.log(2 * math.pi) + math.log(n))
    log_corr = sum(MVDC_LOG[k] / (n ** k) for k in range(len(MVDC_LOG)))
    total_log = log_H + log_corr
    if total_log > math.log(sys.float_info.max):
        return float('inf')
    return math.exp(total_log)

# --- Arithmetic center with 3 correction terms (illustrative only) ---

def arithmetic_center_k3(n):
    """
    Approximation of n! based on the arithmetic center k_a = (n+2)/2.

    Main term: H = k_a^(n-1)
    Correction: first three terms of the MVDC log-series.

    ln K ≈ n(ln 2 − 1) + ln n − ln 2 + ½ ln(2π n) − 2 + 1/(12n)
    """
    if n < 2:
        return 1.0

    pi = math.pi
    ln2 = math.log(2)

    k_a = (n + 2) / 2.0
    ln_H = (n - 1) * math.log(k_a)

    ln_K = (
        n * (ln2 - 1.0)
        + math.log(n) - ln2
        + 0.5 * (math.log(2 * pi) + math.log(n))
        - 2.0
        + 1.0 / (12.0 * n)
    )

    return math.exp(ln_H + ln_K)

# --- Stirling approximation with 3 correction terms ---

def stirling_approx_k3(n: int) -> float:
    """Stirling series up to 1/n⁵ (three Bernoulli terms)."""
    if n < 2:
        return 1.0

    pi = math.pi
    ln_s = n * math.log(n) - n + 0.5 * (math.log(2 * pi) + math.log(n))

    inv_n = 1.0 / n
    inv_n2 = inv_n * inv_n

    ln_s += (
        (1.0 / 12.0) * inv_n
        - (1.0 / 360.0) * inv_n2 * inv_n  # 1/n^3
        + (1.0 / 1260.0) * inv_n2 * inv_n2 * inv_n  # 1/n^5
    )

    return math.exp(ln_s)

# --- Comparison helper ---

def compare_methods(n_values):
    """Print a table comparing the accuracy for the supplied n values."""
    print("-" * 150)
    print(
        f"{'n':>4} | {'Exact n!':>20} | {'Stirling+5':>20} | {'MVDC k=n/e +5':>20} | {'Cascade2':>20}"
    )
    print("-" * 150)

    for n in n_values:
        exact_value = exact_factorial(n)
        stirling5 = stirling_approx_k5(n)
        mvdc5 = mvdc_k5(n)
        cas2 = mvdc_cascade2(n)
        
        abs_err_stirling5 = abs(stirling5 - exact_value)
        abs_err_mvdc5 = abs(mvdc5 - exact_value)
        abs_err_cas2 = abs(cas2 - exact_value)

        # Relative errors not printed per user request

        # Nicely formatted output
        print(
            f"{n:>4d} | {exact_value:>20.10e} | {stirling5:>20.10e} | {mvdc5:>20.10e} | {cas2:>20.10e}"
        )

    print("-" * 150)

# List of n values for which we want the comparison
n_list = [2, 3, 5, 10, 20, 50, 100]
compare_methods(n_list)