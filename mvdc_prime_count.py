"""Bounded prime-counting research backend, independent of the GUI.

The former Bernoulli-denominator and log-moment conversion had no valid error
bound. This utility now separates an exact finite prime count from an explicit
Riemann-R or logarithmic-integral tail model. It does not claim a new MVDC
prime-count theorem or a certified relative error for a nonempty modelled tail.
"""
from dataclasses import dataclass
import math
import mpmath as mp
from mvdc_utils import integer, precision, primes_up_to

MAX_EXACT_CUTOFF = 5_000_000
MAX_X = 10**18


@dataclass(frozen=True)
class PrimeEstimate:
    x: int
    cutoff: int
    exact_front: int
    model: str
    tail: object
    estimate: object
    exact_result: bool
    requested_target: float | None = None

    @property
    def target_certified(self):
        return self.exact_result


def prime_count_exact(x):
    integer(x, "x")
    if x > MAX_EXACT_CUTOFF:
        raise ValueError(f"the built-in exact sieve is capped at {MAX_EXACT_CUTOFF}")
    return len(primes_up_to(x))


def prime_power_count(p, x):
    """Count p**k <= x using integers, including exact power boundaries."""
    integer(p, "p", 2)
    integer(x, "x", 2)
    count, power = 0, p
    while power <= x:
        count += 1
        if power > x // p:
            break
        power *= p
    return count


def mvdc_block(primes, x, moments=2):
    """Exact finite prime-power log block; moments do not add a false correction."""
    integer(moments, "moments")
    integer(x, "x", 2)
    counts = [(p, prime_power_count(p, x)) for p in primes]
    return math.fsum(k*math.log(p) for p, k in counts), sum(k for _, k in counts)


def approximate_pi(x, *, cutoff=None, model="riemannr", dps=50, return_details=False):
    integer(x, "x", 2)
    if x > MAX_X:
        raise ValueError(f"this research utility supports x <= {MAX_X}")
    cutoff = min(x, 100_000) if cutoff is None else integer(cutoff, "cutoff", 2)
    if cutoff > MAX_EXACT_CUTOFF:
        raise ValueError(f"cutoff must be <= {MAX_EXACT_CUTOFF}")
    cutoff = min(cutoff, x)
    if model not in ("riemannr", "li"):
        raise ValueError("model must be 'riemannr' or 'li'")
    with mp.workdps(precision(dps)):
        front = prime_count_exact(cutoff)
        model_fn = mp.riemannr if model == "riemannr" else mp.li
        tail = mp.mpf(0) if cutoff == x else model_fn(x)-model_fn(cutoff)
        result = PrimeEstimate(x, cutoff, front, model, tail, front+tail, cutoff == x)
    return result if return_details else result.estimate


def select_cutoff(x, target=1e-4, max_cutoff=1_000_000):
    """Use the requested target to select finite work, with an explicit cap.

    This bounded effort rule is not an analytic prediction-error certificate.
    """
    integer(x, "x", 2)
    integer(max_cutoff, "max_cutoff", 2)
    if max_cutoff > MAX_EXACT_CUTOFF:
        raise ValueError(f"max_cutoff must be <= {MAX_EXACT_CUTOFF}")
    target = float(target)
    if not math.isfinite(target) or not 0 < target < 1:
        raise ValueError("target must lie strictly between zero and one")
    # Clamp before taking a reciprocal, so even a subnormal target is safe.
    effort = max_cutoff if target <= 1/max_cutoff else math.ceil(1/target)
    return min(x, max_cutoff, max(1000, effort))


def approximate_pi_auto(x, target=1e-4, *, max_cutoff=1_000_000,
                        model="riemannr", dps=50, return_details=False):
    cutoff = select_cutoff(x, target, max_cutoff)
    base = approximate_pi(x, cutoff=cutoff, model=model, dps=dps, return_details=True)
    result = PrimeEstimate(base.x, base.cutoff, base.exact_front, base.model,
                           base.tail, base.estimate, base.exact_result, float(target))
    return result if return_details else result.estimate


def format_result(result, verify=False):
    lines = [f"pi({result.x}) estimate: {mp.nstr(result.estimate, 20)}",
             f"Exact front: pi({result.cutoff}) = {result.exact_front}",
             f"Tail model: {result.model}; tail estimate = {mp.nstr(result.tail, 15)}",
             f"Entire result exact: {result.exact_result}",
             "A modelled tail has no certified relative-error guarantee."]
    if verify:
        reference = prime_count_exact(result.x)
        with mp.workdps(50):
            error = abs(result.estimate-reference)/reference
        lines.extend([f"Exact reference: {reference}", f"Observed relative error: {mp.nstr(error, 12)}"])
    return "\n".join(lines)
