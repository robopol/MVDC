"""A finite Riemann-type prime-count model, evaluated by centred MVDC moments.

Model: pi(M) + sum_{k=1}^K mu(k)/k * integral_M^x t**(1/k-1)/log(t) dt.
The model and Mobius weights are classical; MVDC supplies only the numerical
centred expansion of the smooth tail. No target pi(x), long Mobius table,
zeta zeros, li or Riemann-R call enters the estimator. Numerical truncation
bounds exclude floating-point roundoff and the prime-distribution model error.
"""
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
import math
from time import perf_counter
import mpmath as mp
from mvdc_utils import integer, primes_up_to

MAX_X = 10**18
MAX_EXACT_CUTOFF = 5_000_000
MAX_REFERENCE_X = 10_000_000
DEFAULT_PREFIX_CAP = 10_000
DEFAULT_ORDER = 12
MAX_ORDER = 30
MAX_TAIL_TERMS = 256


def floor_power(x, exponent):
    integer(x, 'x', 2)
    if x > MAX_X:
        raise ValueError(f'x must be <= {MAX_X}')
    if not math.isfinite(float(exponent)) or not 0 < float(exponent) < 1:
        raise ValueError('alpha must lie strictly between zero and one')
    fraction = Fraction(str(exponent))
    with mp.workdps(60):
        guess = int(mp.floor(mp.power(x, mp.mpf(fraction.numerator)/fraction.denominator)))
    if fraction.denominator <= 1000:
        target = x**fraction.numerator
        while guess**fraction.denominator > target:
            guess -= 1
        while (guess+1)**fraction.denominator <= target:
            guess += 1
    return guess


def tail_term_count(x, factor=2.0):
    integer(x, 'x', 2)
    if x > MAX_X:
        raise ValueError(f'x must be <= {MAX_X}')
    if not math.isfinite(float(factor)) or float(factor) <= 0:
        raise ValueError('K_factor must be finite and positive')
    fraction = Fraction(str(factor))
    if fraction.numerator <= 256 and fraction.denominator <= 1000:
        terms = ((x**fraction.numerator).bit_length()-1)//fraction.denominator
    else:
        with mp.workdps(60):
            terms = int(mp.floor(mp.mpf(fraction.numerator)/fraction.denominator*mp.log(x, 2)))
    if not 1 <= terms <= MAX_TAIL_TERMS:
        raise ValueError(f'the finite model requires 1 <= K <= {MAX_TAIL_TERMS}')
    return terms


@lru_cache(maxsize=64)
def mobius_values(limit):
    """Only the short classical coefficient table, never a table through x/2."""
    integer(limit, 'K', 1)
    if limit > MAX_TAIL_TERMS:
        raise ValueError(f'K must be <= {MAX_TAIL_TERMS}')
    values = [1]*(limit+1)
    values[0] = 0
    prime = [True]*(limit+1)
    for p in range(2, limit+1):
        if prime[p]:
            for n in range(p, limit+1, p):
                values[n] = -values[n]
            for n in range(p*p, limit+1, p*p):
                values[n] = 0
            for n in range(2*p, limit+1, p):
                prime[n] = False
    return tuple(values)


def validate_order(order):
    integer(order, 'order')
    if order > MAX_ORDER or order % 2:
        raise ValueError(f'order must be even and between 0 and {MAX_ORDER}')
    return order


@dataclass(frozen=True)
class TailEvaluation:
    value: float
    truncation_bound: float
    panels: int
    nonzero_coefficients: int
    kernel_panels: int
    order: int


def _kernel_panel(k, centre, half, order):
    """Integrate exp(u/k)/u by continuous centred moments on a log panel.

For u=c*(1+v), exp(c*v/k)/(1+v) has coefficients
q_0=1, q_r=(c/k)**r/r!-q_{r-1}. Odd moments cancel on [-h,h].
The continuous moments are the grid-limit of the finite C-Bernoulli moments;
this is a centred Taylor quadrature, not a new prime-distribution theorem.
"""
    rho = half/centre
    scale = centre/k
    exp_coefficient, coefficient = 1.0, 1.0
    terms = [1.0]
    for r in range(1, order+1):
        exp_coefficient *= scale/r
        coefficient = exp_coefficient-coefficient
        if r % 2 == 0:
            terms.append(coefficient*rho**r/(r+1))
    value = 2*half*math.exp(centre/k)/centre*math.fsum(terms)
    # Cauchy's bound in the complex s=u-c disk, which excludes u=0.
    radius = min(.75*centre, k*(order+2))
    ratio = half/radius
    maximum = math.exp((centre+radius)/k)/(centre-radius)
    bound = 2*half*maximum*ratio**(order+2)/((order+3)*(1-ratio**2))
    return value, bound


def mvdc_log_tail(x, cutoff, terms, order=DEFAULT_ORDER):
    """Evaluate the finite Riemann-type tail; no scalar Bernoulli denominator."""
    integer(x, 'x', 2)
    integer(cutoff, 'cutoff', 2)
    if x > MAX_X or cutoff > x:
        raise ValueError('require 2 <= cutoff <= x <= MAX_X')
    validate_order(order)
    mu = mobius_values(terms)
    weights = [(k, mu[k]/k) for k in range(1, terms+1) if mu[k]]
    if x == cutoff:
        return TailEvaluation(0.0, 0.0, 0, len(weights), 0, order)
    # log1p retains close integer endpoints; no subtraction of two li values.
    base = math.log(cutoff)
    span = math.log1p((x-cutoff)/cutoff)
    offset = 0.0
    pieces, bounds, panels = [], [], 0
    while offset < span:
        # h/c <= 1/8 and h <= 1/2: geometric convergence on every panel.
        width = min(span-offset, 2*(base+offset)/7, 1.0)
        following = offset+width
        if following <= offset:
            raise ArithmeticError('log-panel endpoints no longer distinguishable in binary64')
        centre, half = base+offset+width/2, width/2
        for k, weight in weights:
            value, bound = _kernel_panel(k, centre, half, order)
            pieces.append(weight*value)
            bounds.append(abs(weight)*bound)
        panels += 1
        offset = following
    return TailEvaluation(math.fsum(pieces), math.fsum(bounds), panels,
                          len(weights), panels*len(weights), order)


@dataclass(frozen=True)
class PrimeEstimate:
    x: int
    cutoff: int
    requested_cutoff: int
    prefix_cap: int
    exact_front: int
    tail_terms: int
    order: int
    tail: float
    estimate: float
    numerical_truncation_bound: float
    panels: int
    nonzero_coefficients: int
    kernel_panels: int
    elapsed_seconds: float
    requested_numerical_tolerance: float | None = None
    numerical_tolerance_met: bool | None = None
    model: str = 'finite_riemann_type_tail'
    evaluator: str = 'mvdc_continuous_centred_moments'

    @property
    def exact_result(self):
        return self.cutoff == self.x

    @property
    def target_certified(self):
        return self.exact_result


def approximate_pi(x, *, order=DEFAULT_ORDER, alpha=.9, K_factor=2.0,
                   cutoff=None, max_cutoff=DEFAULT_PREFIX_CAP, return_details=False):
    integer(x, 'x', 2)
    if x > MAX_X:
        raise ValueError(f'x must be <= {MAX_X}')
    validate_order(order)
    integer(max_cutoff, 'max_cutoff', 2)
    if max_cutoff > MAX_EXACT_CUTOFF:
        raise ValueError(f'max_cutoff must be <= {MAX_EXACT_CUTOFF}')
    started = perf_counter()
    requested = max(2, floor_power(x, alpha))
    if cutoff is None:
        limit = min(x, max_cutoff, requested)
    else:
        integer(cutoff, 'cutoff', 2)
        if cutoff > MAX_EXACT_CUTOFF:
            raise ValueError(f'explicit cutoff must be <= {MAX_EXACT_CUTOFF}')
        limit = min(x, cutoff)
    terms = tail_term_count(x, K_factor)
    # The known finite prefix is explicit work, not attributed to MVDC.
    front = len(primes_up_to(limit))
    tail = mvdc_log_tail(x, limit, terms, order)
    result = PrimeEstimate(x, limit, requested, max_cutoff, front, terms, order,
        tail.value, front+tail.value, tail.truncation_bound, tail.panels,
        tail.nonzero_coefficients, tail.kernel_panels, perf_counter()-started)
    return result if return_details else result.estimate


def approximate_pi_auto(x, numerical_tolerance=1e-8, *, return_details=False, **kwargs):
    """Control numerical tail truncation only, not the model's error in pi(x)."""
    if not math.isfinite(float(numerical_tolerance)) or numerical_tolerance <= 0:
        raise ValueError('numerical_tolerance must be finite and positive')
    if 'order' in kwargs:
        raise ValueError('automatic numerical order does not accept order')
    for order in range(2, MAX_ORDER+1, 2):
        result = approximate_pi(x, order=order, return_details=True, **kwargs)
        if result.numerical_truncation_bound <= numerical_tolerance:
            break
    result = replace(result, requested_numerical_tolerance=numerical_tolerance,
        numerical_tolerance_met=result.numerical_truncation_bound <= numerical_tolerance)
    return result if return_details else result.estimate


def prime_count_exact(x):
    """Separate bounded reference, never called by approximate_pi."""
    integer(x, 'x')
    if x > MAX_REFERENCE_X:
        raise ValueError(f'local exact verification is capped at {MAX_REFERENCE_X}')
    return len(primes_up_to(x))


def format_result(result, verify=False):
    lines = [f'Approximate pi({result.x}): {result.estimate:.17g}',
        'Model: classical finite Riemann-type root tail, anchored at the known prefix.',
        'Numerical evaluation: MVDC continuous centred moments (Taylor quadrature).',
        f'Known prefix: M={result.cutoff}, pi(M)={result.exact_front}; requested M={result.requested_cutoff}',
        f'Classical Mobius coefficients: k <= {result.tail_terms}; no table through x/2.',
        f'MVDC order: {result.order}; log panels: {result.panels}; kernel panels: {result.kernel_panels}',
        f'Tail estimate: {result.tail:.16g}',
        f'Numerical truncation bound, excluding roundoff: {result.numerical_truncation_bound:.9g}',
        'The numerical bound does not bound the model error against pi(x).',
        f'Elapsed approximation time: {1000*result.elapsed_seconds:.6f} ms',
        f'Entire range counted exactly: {result.exact_result}']
    if result.requested_numerical_tolerance is not None:
        lines.append(f'Requested numerical truncation tolerance met: {result.numerical_tolerance_met}')
    if verify:
        exact = prime_count_exact(result.x)
        difference = Fraction.from_float(result.estimate)-exact
        lines.extend((f'Exact pi(x), verification only: {exact}',
                      f'Signed model-plus-numerical error: {float(difference):.12g}',
                      f'Observed relative error: {float(abs(difference)/exact):.12g}'))
    return '\n'.join(lines)
