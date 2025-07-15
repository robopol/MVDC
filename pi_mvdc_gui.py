#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pi_mvdc_gui.py – Simple Tkinter interface for the MVDC approximation of π(x).

Features:
• enter x (integer ≥ 2)
• choose automatic mode (target relative error) or tweak manual parameters
• display MVDC estimate, runtime, and – if SymPy is installed – the exact value
  with relative error.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import scrolledtext
from fractions import Fraction
from time import perf_counter
import os
import sys

# Ensure we can import sibling module when executed as a script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

# ----------------- Computational backend (self-contained) ------------------

import math
from typing import List, Tuple

# Fast prime generation
try:
    import primesieve  # type: ignore

    def primes_up_to(n: int) -> List[int]:
        return primesieve.primes(n)
except ModuleNotFoundError:  # fallback
    try:
        from sympy import primerange  # type: ignore

        def primes_up_to(n: int) -> List[int]:
            return list(primerange(2, n + 1))
    except ImportError:
        def primes_up_to(_n: int) -> List[int]:
            raise ImportError("Need primesieve or sympy for prime generation")


# Möbius function (minimal fallback if sympy missing)
try:
    from sympy.functions.combinatorial.numbers import mobius  # type: ignore
except ImportError:
    try:
        from sympy.ntheory import mobius  # type: ignore
    except ImportError:
        def mobius(n: int) -> int:  # type: ignore
            if n == 1:
                return 1
            primes_f = []
            d = 2
            nn = n
            while d * d <= nn:
                if nn % d == 0:
                    primes_f.append(d)
                    nn //= d
                    if nn % d == 0:
                        return 0  # square factor
                else:
                    d += 1
            if nn > 1:
                primes_f.append(nn)
            if len(set(primes_f)) != len(primes_f):
                return 0
            return -1 if len(primes_f) % 2 else 1


def mvdc_block(primes: List[int], x: int, moments: int = 2) -> Tuple[float, int]:
    ln_x = math.log(x)
    if moments >= 2:
        logs: List[float] = []
        for p in primes:
            k_max = int(ln_x // math.log(p))
            logs.extend([math.log(p)] * k_max)

        n = len(logs)
        if n == 0:
            return 0.0, 0

        mu1 = sum(logs) / n
        centred = [v - mu1 for v in logs]
        log_sum = n * mu1
        for r in range(2, moments + 1):
            Sr = sum(c ** r for c in centred)
            coeff = ((-1) ** (r - 1)) / (r * n ** (r - 1))
            log_sum += coeff * Sr
        return log_sum, n
    else:
        log_sum = 0.0
        n = 0
        for p in primes:
            k_max = int(ln_x // math.log(p))
            log_p = math.log(p)
            log_sum += k_max * log_p
            n += k_max
        return log_sum, n


def approximate_pi(
    x: int,
    *,
    moments: int = 2,
    correction_terms: int = 5,
    alpha: float = 0.9,
    beta: float = 0.3,
    K_factor: float = 2.0,
    return_details: bool = False,
) -> tuple[float, float, float] | float:
    # Note: For small x the error grows, but calculation is still valid.

    if not (0 < beta < alpha < 1):
        raise ValueError("Require 0 < beta < alpha < 1")

    ln_x = math.log(x)
    N = int(x ** beta)
    M = int(x ** alpha)

    # front
    theta_front = 0.0
    for p in primes_up_to(N):
        k_max = int(ln_x // math.log(p))
        theta_front += k_max * math.log(p)

    # block
    block_primes = [p for p in primes_up_to(M) if p > N]
    log_block, _ = mvdc_block(block_primes, x, moments)

    # tail via Möbius-Rosser series
    K = int(K_factor * math.log2(x))
    tail = 0.0
    for k in range(1, K + 1):
        mu = mobius(k)
        if mu == 0:
            continue
        term = mu * (x ** (1 / k) - M ** (1 / k)) / k
        tail += term

    theta = theta_front + log_block + tail

    denom = ln_x - 1 - 1 / ln_x  # first two terms (B0 implicit)
    for m in range(1, correction_terms + 1):
        idx = 2 * m
        B = _BERN.get(idx, Fraction(0, 1))
        denom -= float(B) / (idx * ln_x ** idx)
    pi_est = theta / denom
    if return_details:
        return pi_est, theta, denom
    return pi_est


def _auto_parameters(x: int, target: float = 1e-4):
    ln_x = math.log(x)
    alpha = max(0.85, min(1 - 1 / ln_x, 0.97))
    beta = 0.25 if alpha > 0.92 else 0.3
    c = max(2, math.ceil(-math.log10(target)))
    K_factor = float(c)

    J = 0
    fact = 1
    inv_ln = 1 / ln_x
    term = inv_ln
    while term > target / 10:
        J += 1
        fact *= J
        term = fact * inv_ln ** (J + 1)
    correction_terms = J + 1
    return alpha, beta, K_factor, correction_terms


def approximate_pi_auto(x: int, target: float = 1e-4, *, moments: int | None = None, return_details: bool = False):
    alpha, beta, Kf, corr = _auto_parameters(x, target)
    if moments is None:
        moments = 4 if x >= 1_000_000 else 2
    return approximate_pi(
        x,
        moments=moments,
        correction_terms=corr,
        alpha=alpha,
        beta=beta,
        K_factor=Kf,
        return_details=return_details,
    )

# ----------------------------------------------------------------------


try:
    from sympy import primepi 
except ImportError:
    primepi = None  # type: ignore


def safe_float(value: str, default: float):
    try:
        return float(value.replace(',', '.'))
    except ValueError:
        return default


def on_compute():
    try:
        x_val = int(entry_x.get())
        if x_val < 2:
            raise ValueError
    except ValueError:
        messagebox.showerror("Error", "Enter an integer x ≥ 2")
        return

    # manual always active; no toggling needed

    t0 = perf_counter()
    try:
        moments = int(spin_moments.get())
        correction_terms = int(spin_corr.get())
        corr_terms = correction_terms
        alpha = safe_float(entry_alpha.get(), 0.9)
        beta = safe_float(entry_beta.get(), 0.3)
        kfac = safe_float(entry_kfac.get(), 2.0)
        est, theta, denom = approximate_pi(
            x_val,
            moments=moments,
            correction_terms=correction_terms,
            alpha=alpha,
            beta=beta,
            K_factor=kfac,
            return_details=True,
        )
    except Exception as exc:
        messagebox.showerror("Computation failed", str(exc))
        return
    t1 = perf_counter()

    # Výstup
    lines = [f"π_MVDC({x_val}) ≈ {int(est):,}", f"elapsed: {t1 - t0:.3f} s"]

    # add formula components
    lines.append(f"Θ(x) = {theta:.6g}")
    lines.append(f"π ≈ Θ / D = {int(est):,}")

    # symbolic formula explanation
    lines.append("-- formula components --")
    lines.append("Θ(x) = θ_front + θ_block + θ_tail")
    lines.append("θ_front   = Σ_{p≤N} k_max·ln p   (all powers up to x)")
    lines.append("θ_block   = n·μ₁ + Σ_{r=2..m} (−1)^{r−1} S_r /(r n^{r−1})  ; m = moments")
    lines.append("θ_tail    = Σ_{k=1..K} μ(k)(x^{1/k} − M^{1/k})/k  ;  K ≈ K_factor·log₂x")

    N_val = int(x_val ** beta)
    M_val = int(x_val ** alpha)
    K_val = int(kfac * math.log2(x_val))
    lines.append(f"N = x^β = {N_val:,}  ,  M = x^α = {M_val:,}  ,  K = {K_val}")
    lines.append(f"parameters: α={alpha:.3f}, β={beta:.3f}, K_fac={kfac}, moments={moments}, Li_terms={corr_terms}")

    # Build LaTeX denominator with Bernoulli fractions
    denom_parts = ["\\ln x", "-1", "-\\dfrac{1}{\\ln x}"]
    for m in range(1, corr_terms + 1):
        idx = 2 * m
        B = _BERN.get(idx, Fraction(0, 1))
        if B == 0:
            continue
        sign = "-" if B > 0 else "+"
        frac = f"\\dfrac{{{abs(B.numerator)}}}{{{B.denominator}}}"
        denom_parts.append(f"{sign} {frac}/({idx} \\ln^{idx}x)")
    denom_tex = " ".join(denom_parts)

    latex_formula = (
        r"\[\pi(x)=\frac{\theta_{\mathrm{front}}+\theta_{\mathrm{block}}+\theta_{\mathrm{tail}}}{" +
        denom_tex + r"}\]"
    )
    lines.append("--- LaTeX formula ---")
    lines.append(latex_formula)

    # Build symbolic denominator string
    ln_x = math.log(x_val)
    denom_terms = ["ln x", "- 1", "- 1/ln x"]
    for m in range(1, corr_terms + 1):
        idx = 2 * m
        B = _BERN.get(idx, Fraction(0, 1))
        sign = "-" if B > 0 else "+"
        b_str = f"{abs(B.numerator)}/{B.denominator}"
        term_str = f"{sign} {b_str}/({idx} ln^{idx}x)"
        denom_terms.append(term_str)
    denom_expr = " ".join(denom_terms)
    lines.append(f"D = {denom_expr}")
    lines.append(f"D ≈ {denom:.6g}")

    if primepi is not None:
        exact = int(primepi(x_val))
        abs_err = int(est) - exact
        rel = abs(abs_err) / exact
        lines.append(f"π_exact = {exact:,}")
        lines.append(f"abs error: {abs_err:+,}")
        lines.append(f"relative error: {rel:.3e}")

    txt_result.configure(state="normal")
    txt_result.delete(1.0, tk.END)
    txt_result.insert(tk.END, "\n".join(lines))
    txt_result.configure(state="disabled")


# ----------------------------- GUI -----------------------------------
root = tk.Tk()
root.title("MVDC approximation of the prime-counting function π(x)")

mainfrm = ttk.Frame(root, padding=10)
mainfrm.grid(row=0, column=0, sticky="nsew")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
# allow main frame rows/cols to expand
mainfrm.columnconfigure(0, weight=1)
mainfrm.rowconfigure(2, weight=1)

# x entry
lbl_x = ttk.Label(mainfrm, text="x (≥2):")
lbl_x.grid(row=0, column=0, sticky="e")
entry_x = ttk.Entry(mainfrm, width=20)
entry_x.insert(0, "1000000")
entry_x.grid(row=0, column=1, sticky="w")

# Manual parameters frame
frm_manual = ttk.LabelFrame(mainfrm, text="Manual parameters")
frm_manual.grid(row=1, column=0, columnspan=2, pady=(10, 4), sticky="ew")
frm_manual.columnconfigure((1, 3, 5), weight=1)

# moments
ttk.Label(frm_manual, text="moments:").grid(row=0, column=0, sticky="e")
spin_moments = ttk.Spinbox(frm_manual, from_=0, to=10, width=5)
spin_moments.set(2)
spin_moments.grid(row=0, column=1, sticky="w")

# correction_terms
ttk.Label(frm_manual, text="Li terms:").grid(row=0, column=2, sticky="e")
spin_corr = ttk.Spinbox(frm_manual, from_=1, to=15, width=5)
spin_corr.set(5)
spin_corr.grid(row=0, column=3, sticky="w")

# alpha
ttk.Label(frm_manual, text="α (M = x^α):").grid(row=1, column=0, sticky="e")
entry_alpha = ttk.Entry(frm_manual, width=8)
entry_alpha.insert(0, "0.9")
entry_alpha.grid(row=1, column=1, sticky="w")

# beta
ttk.Label(frm_manual, text="β (N = x^β):").grid(row=1, column=2, sticky="e")
entry_beta = ttk.Entry(frm_manual, width=8)
entry_beta.insert(0, "0.3")
entry_beta.grid(row=1, column=3, sticky="w")

# K_factor
ttk.Label(frm_manual, text="K factor:").grid(row=2, column=0, sticky="e")
entry_kfac = ttk.Entry(frm_manual, width=8)
entry_kfac.insert(0, "2.0")
entry_kfac.grid(row=2, column=1, sticky="w")

# Compute button
btn_compute = ttk.Button(mainfrm, text="Compute", command=on_compute)
btn_compute.grid(row=3, column=0, columnspan=2, pady=8)

# Result box
txt_result = scrolledtext.ScrolledText(mainfrm, state="disabled")
txt_result.grid(row=2, column=0, columnspan=2, pady=(0, 10), sticky="nsew")

# manual always active; no toggling needed

# (Conversion method fixed to Bernoulli denominator – no UI needed)

# Precompute Bernoulli numbers up to B_20
_BERN = {
    2: Fraction(1, 6),
    4: Fraction(-1, 30),
    6: Fraction(1, 42),
    8: Fraction(-1, 30),
    10: Fraction(5, 66),
    12: Fraction(-691, 2730),
    14: Fraction(7, 6),
    16: Fraction(-3617, 510),
    18: Fraction(43867, 798),
}

# --- updated compute for conversion choice ---

root.update_idletasks()
root.geometry("900x650")
root.mainloop() 