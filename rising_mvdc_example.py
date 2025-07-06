import math, sys
from mvdc_utils import mvdc_generic_center

a = 1/3  # parameter of rising factorial (a)_n

# (a)_n can overflow; work in log-domain and fall back to inf if needed
def rising_exact(n):
    return math.lgamma(a + n) - math.lgamma(a)  # return log value directly

def mvdc_main_log(n):
    factors = [a + k for k in range(n)]
    _, ln_H = mvdc_generic_center(factors, order=2)
    return ln_H

# Fit polynomial coeffs for log-residual up to 5th order

def fit_coeffs(order=5, start=10, stop=200, step=10):
    rows, y = [], []
    for n in range(start, stop + 1, step):
        log_exact = math.lgamma(a + n) - math.lgamma(a)
        _, ln_H = mvdc_generic_center([a + k for k in range(n)], order=2)
        log_r = log_exact - ln_H
        rows.append([1.0] + [1.0 / n**k for k in range(1, order + 1)])
        y.append(log_r)
    m = order + 1
    xtx = [[0.0]*m for _ in range(m)]
    xty = [0.0]*m
    for row, yi in zip(rows, y):
        for i in range(m):
            xty[i] += row[i]*yi
            for j in range(m):
                xtx[i][j] += row[i]*row[j]
    # augment and Gaussian eliminate
    for i in range(m):
        xtx[i].append(xty[i])
    for i in range(m):
        piv = xtx[i][i]
        for j in range(i, m+1):
            xtx[i][j] /= piv
        for k in range(m):
            if k==i: continue
            fac=xtx[k][i]
            for j in range(i, m+1):
                xtx[k][j]-=fac*xtx[i][j]
    return [xtx[i][m] for i in range(m)]

COEFFS = fit_coeffs()


def mvdc_cascade_log(n):
    factors=[a+k for k in range(n)]
    _, ln_H = mvdc_generic_center(factors, order=2)
    log_corr = sum(COEFFS[k]/n**k for k in range(len(COEFFS)))
    return ln_H + log_corr


def compare(ns):
    print("MVDC approximation of the rising factorial (a)_n with a = 1/3. Values are base-10 logs.")
    print(f"{'n':>4} | {'log10 Exact':>15} | {'log10 H':>15} | {'H+3':>15} | {'H+5':>15} | ΔH+5")
    for n in ns:
        log_exact = rising_exact(n)
        log_H = mvdc_main_log(n)
        # polynomial corrections
        log_corr3 = sum(COEFFS[k]/n**k for k in range(1, 4))  # k=1..3
        log_corr5 = sum(COEFFS[k]/n**k for k in range(1, 6))  # k=1..5
        log_H3 = log_H + log_corr3
        log_H5 = log_H + log_corr5
        dlog10 = (log_H5 - log_exact)/math.log(10)
        print(f"{n:4d} | {log_exact/math.log(10):15.6f} | {log_H/math.log(10):15.6f} | {log_H3/math.log(10):15.6f} | {log_H5/math.log(10):15.6f} | {dlog10:.1e}")

if __name__=='__main__':
    compare([1,2,5,10,20,50,100,200]) 