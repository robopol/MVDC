"""The original MVDC approximation, with a derived finite factor-moment block.

Only primes p <= M are enumerated. The short original root-tail model and
Bernoulli denominator are retained as explicit, unproved modelling choices.
The finite block follows the paper's log(1+u) identity, with bounded geometric
subblocks. Its truncation bound is NOT a bound on the complete pi estimate.
Exact pi(x), li, Riemann-R and long Mobius inversion are not estimator inputs.
"""
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
import bisect
import math
import mpmath as mp
from mvdc_utils import bernoulli_numbers, integer, primes_up_to

MAX_X = 10**18
MAX_EXACT_CUTOFF = 5_000_000
MAX_REFERENCE_X = 10_000_000
MAX_MOMENTS = 20
MAX_CORRECTIONS = 20
MAX_TAIL_TERMS = 4096
DEFAULT_MOMENTS = 8


def prime_count_exact(x):
    """Explicit verification only; never used to construct an estimate."""
    integer(x, 'x')
    if x > MAX_REFERENCE_X:
        raise ValueError(f'exact verification is capped at {MAX_REFERENCE_X}')
    return len(primes_up_to(x))


def prime_power_count(p, x):
    integer(p, 'p', 2)
    integer(x, 'x', 2)
    count, power = 0, p
    while power <= x:
        count += 1
        if power > x // p:
            break
        power *= p
    return count


def floor_power(x, exponent):
    integer(x, 'x', 2)
    if x > MAX_X:
        raise ValueError(f'x must be <= {MAX_X}')
    if not math.isfinite(float(exponent)) or not 0 < float(exponent) < 1:
        raise ValueError('cutoff exponent must lie strictly between zero and one')
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
    """floor(factor*log2(x)), including zero and exact power boundaries."""
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
    if terms > MAX_TAIL_TERMS:
        raise ValueError(f'tail is limited to {MAX_TAIL_TERMS} terms')
    return terms


@lru_cache(maxsize=32)
def mobius_values(limit):
    """Small coefficient table for the original finite root tail only."""
    integer(limit, 'tail limit')
    if limit > MAX_TAIL_TERMS:
        raise ValueError(f'tail is limited to {MAX_TAIL_TERMS} terms')
    values = [1]*(limit+1)
    values[0] = 0
    prime = [True]*(limit+1)
    for p in range(2, limit+1):
        if prime[p]:
            for multiple in range(p, limit+1, p):
                values[multiple] = -values[multiple]
            for multiple in range(p*p, limit+1, p*p):
                values[multiple] = 0
            for multiple in range(2*p, limit+1, p):
                prime[multiple] = False
    return tuple(values)


@dataclass(frozen=True)
class BlockDetails:
    value: float
    truncation_bound: float
    power_count: int
    subblocks: int
    max_relative_deviation: float


def block_details(primes, x, moments=DEFAULT_MOMENTS):
    """Approximate sum k_p log(p) by the paper's factor-moment identity.

For each group p_max <= 2*p_min, c=(p_min+p_max)/2 and u=p/c-1.
Use b*log(c) + sum_r (-1)^(r-1)/r * sum_p k_p*u_p**r.
The omitted log-series tail has the stated absolute bound. No exact block
logarithm is evaluated and then shifted by unrelated log central moments.
"""
    integer(x, 'x', 2)
    integer(moments, 'moments')
    if moments > MAX_MOMENTS:
        raise ValueError(f'moments must be <= {MAX_MOMENTS}')
    values = tuple(primes)
    if any(not isinstance(p, int) or isinstance(p, bool) or p < 2 or p > x for p in values):
        raise ValueError('block factors must be integers between 2 and x')
    if any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError('block factors must be strictly increasing')
    approximations, bounds = [], []
    total_count, subblocks, max_rho, index = 0, 0, 0.0, 0
    while index < len(values):
        stop = bisect.bisect_right(values, 2*values[index], lo=index)
        group = values[index:stop]
        centre = (group[0]+group[-1])/2
        weighted = [(prime_power_count(p, x), (p-centre)/centre) for p in group]
        count = sum(k for k, _ in weighted)
        correction = math.fsum((-1)**(r-1)*math.fsum(k*u**r for k, u in weighted)/r
                               for r in range(1, moments+1))
        approximations.append(count*math.log(centre)+correction)
        bounds.append(math.fsum(k*abs(u)**(moments+1)/((moments+1)*(1-abs(u)))
                                for k, u in weighted))
        total_count += count
        subblocks += 1
        max_rho = max(max_rho, max(abs(u) for _, u in weighted))
        index = stop
    return BlockDetails(math.fsum(approximations), math.fsum(bounds), total_count, subblocks, max_rho)


def mvdc_block(primes, x, moments=DEFAULT_MOMENTS):
    result = block_details(primes, x, moments)
    return result.value, result.power_count


def bernoulli_denominator(x, correction_terms=5):
    """Original experimental denominator; it is not derived from C coefficients."""
    integer(x, 'x', 2)
    integer(correction_terms, 'correction_terms')
    if correction_terms > MAX_CORRECTIONS:
        raise ValueError(f'correction_terms must be <= {MAX_CORRECTIONS}')
    logarithm = math.log(x)
    coefficients = bernoulli_numbers(2*correction_terms)
    result = math.fsum([logarithm, -1.0, -1/logarithm] +
        [-float(coefficients[2*j])/(2*j*logarithm**(2*j)) for j in range(1, correction_terms+1)])
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f'original denominator is not positive at x={x}: D={result:.12g}')
    return result


def mobius_root_tail(x, cutoff, terms):
    """Preserve the original short empirical tail; no full-range inversion."""
    integer(x, 'x', 2)
    integer(cutoff, 'cutoff', 2)
    integer(terms, 'tail terms')
    if cutoff > x:
        raise ValueError('cutoff must be <= x')
    mu = mobius_values(terms)
    if terms == 0 or cutoff == x:
        return 0.0
    log_cutoff = math.log(cutoff)
    log_ratio = math.log1p((x-cutoff)/cutoff)
    return math.fsum(mu[k]*math.exp(log_cutoff/k)*math.expm1(log_ratio/k)/k
                     for k in range(1, terms+1) if mu[k])


@dataclass(frozen=True)
class PrimeEstimate:
    x: int
    front_cutoff: int
    cutoff: int
    requested_cutoff: int
    tail_terms: int
    moments: int
    correction_terms: int
    front_log: float
    block_log: float
    block_truncation_bound: float
    block_count_error_bound: float
    block_subblocks: int
    block_max_relative_deviation: float
    tail: float
    numerator: float
    denominator: float
    estimate: float
    primes_evaluated: int
    power_count: int
    requested_block_tolerance: float | None = None
    block_tolerance_met: bool | None = None
    model: str = 'original_approximation_with_derived_mvdc_block'

    @property
    def exact_result(self):
        return False

    @property
    def target_certified(self):
        return False


def approximate_pi(x, *, moments=DEFAULT_MOMENTS, correction_terms=5, alpha=.9,
                   beta=.3, K_factor=2.0, cutoff=None, return_details=False):
    integer(x, 'x', 2)
    if x > MAX_X:
        raise ValueError(f'x must be <= {MAX_X}')
    if not (math.isfinite(float(alpha)) and math.isfinite(float(beta)) and 0 < beta < alpha < 1):
        raise ValueError('require 0 < beta < alpha < 1')
    integer(moments, 'moments')
    if moments > MAX_MOMENTS:
        raise ValueError(f'moments must be <= {MAX_MOMENTS}')
    terms = tail_term_count(x, K_factor)
    denominator = bernoulli_denominator(x, correction_terms)
    requested = max(2, floor_power(x, alpha))
    limit = requested if cutoff is None else integer(cutoff, 'cutoff', 2)
    limit = min(limit, x)
    if limit > MAX_EXACT_CUTOFF:
        raise ValueError(f'M={limit} exceeds {MAX_EXACT_CUTOFF}; choose an explicit smaller cutoff')
    front_limit = min(floor_power(x, beta), limit)
    primes = primes_up_to(limit)
    split = bisect.bisect_right(primes, front_limit)
    front_counts = [(p, prime_power_count(p, x)) for p in primes[:split]]
    front = math.fsum(k*math.log(p) for p, k in front_counts)
    block = block_details(primes[split:], x, moments)
    tail = mobius_root_tail(x, limit, terms)
    numerator = math.fsum((front, block.value, tail))
    result = PrimeEstimate(x, front_limit, limit, requested, terms, moments, correction_terms,
        front, block.value, block.truncation_bound, block.truncation_bound/denominator,
        block.subblocks, block.max_relative_deviation, tail, numerator, denominator,
        numerator/denominator, len(primes), sum(k for _, k in front_counts)+block.power_count)
    return result if return_details else result.estimate


def approximate_pi_auto(x, block_tolerance=1e-6, *, return_details=False, **kwargs):
    """Adapt only the MVDC block order; do not promise a total pi error target."""
    if not math.isfinite(float(block_tolerance)) or block_tolerance <= 0:
        raise ValueError('block_tolerance must be finite and positive')
    if 'moments' in kwargs:
        raise ValueError('automatic block order does not accept moments')
    for moments in range(2, MAX_MOMENTS+1, 2):
        result = approximate_pi(x, moments=moments, return_details=True, **kwargs)
        if result.block_count_error_bound <= block_tolerance:
            break
    result = replace(result, requested_block_tolerance=block_tolerance,
                     block_tolerance_met=result.block_count_error_bound <= block_tolerance)
    return result if return_details else result.estimate


def format_result(result, verify=False):
    lines = [f'MVDC approximation pi({result.x}): {result.estimate:.17g}',
        f'N={result.front_cutoff}, M={result.cutoff}, K={result.tail_terms}',
        f'Primes enumerated through M: {result.primes_evaluated}; prime-power terms: {result.power_count}',
        f'MVDC factor-moment order: {result.moments}; subblocks: {result.block_subblocks}',
        f'Maximum relative factor deviation: {result.block_max_relative_deviation:.9g}',
        f'Front log sum: {result.front_log:.16g}', f'MVDC block: {result.block_log:.16g}',
        f'Original root-tail model: {result.tail:.16g}', f'Original denominator: {result.denominator:.16g}',
        f'MVDC block truncation contribution to count: <= {result.block_count_error_bound:.9g}',
        'This bound excludes tail-model error, denominator-model error and roundoff.',
        'The original tail and denominator remain unproved modelling choices.']
    if result.requested_block_tolerance is not None:
        lines.append(f'Requested block tolerance met: {result.block_tolerance_met}')
    if verify:
        reference = prime_count_exact(result.x)
        error = result.estimate-reference
        lines.extend((f'Exact pi(x), verification only: {reference}',
                      f'Signed error: {error:.12g}', f'Observed relative error: {abs(error)/reference:.12g}'))
    return '\n'.join(lines)
