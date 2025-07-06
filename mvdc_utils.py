# mvdc_utils.py
import math
import statistics
from typing import Iterable, Tuple

def mvdc_generic_center(factors: Iterable[float], order: int = 2) -> Tuple[float, float]:
    """Return (k, ln_H) chosen automatically.

    Parameters
    ----------
    factors : iterable of positive floats
    order   : 1 or 2 (default 2).  order=1 → geometric mean; order>=2 → try
               three options (no variance, +σ²/2, −σ²/2) and select the one
               with minimal absolute third central moment (skew).

    The routine stays purely data-driven – it never imports external asymptotics.
    """

    values = list(factors)
    if not values:
        raise ValueError("factors list is empty")

    m = len(values)
    logs = [math.log(x) for x in values]

    # 1st moment – geometric mean centre
    mu1 = sum(logs) / m
    ln_H0 = m * mu1  # ln H for k0

    if order < 2:
        return math.exp(mu1), ln_H0

    # second central moment
    var = statistics.fmean((x - mu1) ** 2 for x in logs)
    shift = var / 2.0

    # First choose by how well the residual R (first moment) is cancelled
    def residual_abs(sign: int) -> float:
        return abs(sum(logs) - m * (mu1 + sign * shift))

    res_candidates = {s: residual_abs(s) for s in (0, 1, -1)}
    best_sign = min(res_candidates, key=res_candidates.get)

    # If two signs tie within 1e-12, fall back to minimising absolute 3rd moment
    def third_moment_abs(sign: int) -> float:
        delta = sign * shift
        return abs(statistics.fmean(((x - delta) ** 3) for x in logs))

    best_res = res_candidates[best_sign]
    near = [s for s, r in res_candidates.items() if abs(r - best_res) < 1e-12]
    if len(near) > 1:
        # choose one with smallest skewness
        skew_cands = {s: third_moment_abs(s) for s in near}
        best_sign = min(skew_cands, key=skew_cands.get)

    ln_H = ln_H0 + best_sign * m * shift
    k = math.exp(mu1 + best_sign * shift)
    return k, ln_H 