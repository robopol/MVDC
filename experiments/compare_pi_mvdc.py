"""Separate classical model error, MVDC evaluation error and approximation cost.

Exact reference counts never enter the estimator. Local reference counts use
an independent sieve through 10^7. Larger fixed points use published exact
counts; these are lookup references, not computations performed by MVDC.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import bisect
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
from time import perf_counter
import mpmath as mp
import mvdc_prime_count as mvdc
from experiments.archive import pi_mvdc_factor_block_20261019 as previous

POINTS = (2, 3, 4, 5, 6, 7, 8, 10, 16, 25, 32, 49, 64, 100, 243, 1000,
          4096, 6561, 10000, 30000, 100000, 300000, 390625, 1000000, 3000000,
          10000000, 10**8, 10**9, 10**10, 10**12, 10**15, 10**18)
PUBLISHED_REFERENCE_ONLY = {
    10**8: 5761455, 10**9: 50847534, 10**10: 455052511,
    10**12: 37607912018, 10**15: 29844570422669, 10**18: 24739954287740860,
}
REFERENCE_URL = 'https://t5k.org/howmany.html'


def reference_primes(limit):
    if limit < 2:
        return []
    size = (limit+1)//2
    flags = bytearray(b'\x01')*size
    flags[0] = 0
    for index in range(1, (math.isqrt(limit)+1)//2):
        if flags[index]:
            p = 2*index+1
            start = p*p//2
            flags[start::p] = bytes((size-1-start)//p+1)
    return [2]+[2*i+1 for i in range(1, size) if flags[i]]


def model_reference(result):
    """Independent 70-digit evaluation of the SAME finite classical model."""
    with mp.workdps(70):
        mu = mvdc.mobius_values(result.tail_terms)
        tail = mp.fsum(mp.mpf(mu[k])/k *
            (mp.li(mp.root(result.x, k))-mp.li(mp.root(result.cutoff, k)))
            for k in range(1, result.tail_terms+1) if mu[k])
        return mp.mpf(result.exact_front)+tail


def encode_error(estimate, exact):
    with mp.workdps(70):
        value = mp.mpf(estimate)
        error = value-exact
        return {'estimate': mp.nstr(value, 25), 'signed_error': mp.nstr(error, 25),
                'relative_error': mp.nstr(abs(error)/exact, 25),
                'rounded_count': round(estimate), 'rounded_count_error': round(estimate)-exact}


def collect(points=POINTS, dense_limit=1000):
    points = tuple(sorted(set(points)))
    if not points or min(points) < 2 or max(points) > mvdc.MAX_X:
        raise ValueError('points are outside the supported domain')
    if any(x > 10**7 and x not in PUBLISHED_REFERENCE_ONLY for x in points):
        raise ValueError('large comparison points must have a published exact reference')
    if not 2 <= dense_limit <= 10000:
        raise ValueError('dense_limit must be in [2, 10000]')
    predictions = []
    for x in points:
        # Cold in-process prefix/coefficient caches for each reported timing.
        mvdc.primes_up_to.cache_clear()
        mvdc.mobius_values.cache_clear()
        result = mvdc.approximate_pi(x, return_details=True)
        try:
            old = previous.approximate_pi(x, cutoff=result.cutoff)
        except ValueError as error:
            old = str(error)
        predictions.append((result, old))
        print(f'x={x}: estimate={result.estimate:.16g}; M={result.cutoff}; K={result.tail_terms}; '
              f'MVDC panels={result.panels}; cold={1000*result.elapsed_seconds:.3f} ms', flush=True)
    dense = [mvdc.approximate_pi(x, return_details=True) for x in range(2, dense_limit+1)]
    convergence = [mvdc.approximate_pi(x, order=order, cutoff=2, return_details=True)
                   for x in (10000, 10**12) for order in (0, 2, 4, 8, 12, 16, 20)]
    sensitivity = [mvdc.approximate_pi(x, cutoff=cutoff, return_details=True)
                   for x in (10000, 10**7, 10**12) for cutoff in (2, 100, 10000) if cutoff < x]
    # All predictions are fixed before the exact target-reference sieve is built.
    started = perf_counter()
    # The fixed prefix-sensitivity grid also contains 10^7, even when a custom
    # selected-point list is smaller.
    sieve_limit = max([10**7, dense_limit]+[x for x in points if x <= 10**7])
    primes = reference_primes(sieve_limit)
    sieve_seconds = perf_counter()-started
    for x, count in ((10, 4), (100, 25), (1000, 168), (10000, 1229),
                     (100000, 9592), (1000000, 78498), (10000000, 664579)):
        if x <= sieve_limit:
            assert bisect.bisect_right(primes, x) == count
    def reference(x):
        if x <= sieve_limit:
            return bisect.bisect_right(primes, x), 'independent local odd-only sieve'
        return PUBLISHED_REFERENCE_ONLY[x], REFERENCE_URL
    rows = []
    for result, old in predictions:
        exact, source = reference(result.x)
        model = model_reference(result)
        with mp.workdps(70):
            numerical_error = mp.mpf(result.estimate)-model
            model_error = model-exact
            allowance = result.numerical_truncation_bound+128*math.ulp(result.estimate)
            if abs(numerical_error) > allowance:
                raise ArithmeticError(f'MVDC evaluation check failed at {result.x}')
            row = {'x': result.x, 'exact_pi_reference_only': exact, 'exact_reference_source': source,
                'approximation': {**encode_error(result.estimate, exact), 'details': asdict(result)},
                'finite_classical_model_70_digit': mp.nstr(model, 60),
                'mvdc_numerical_error_vs_same_model': mp.nstr(numerical_error, 35),
                'classical_model_error_vs_pi': mp.nstr(model_error, 35),
                'previous_formula_same_prefix': {'error': old} if isinstance(old, str) else encode_error(old, exact),
                'exact_prefix_only': result.exact_result}
        rows.append(row)
    dense_rows = []
    for result in dense:
        exact, _ = reference(result.x)
        dense_rows.append({'x': result.x, 'exact_pi_reference_only': exact,
            **encode_error(result.estimate, exact), 'cutoff': result.cutoff,
            'exact_prefix_only': result.exact_result})
    convergence_rows = []
    for result in convergence:
        model = model_reference(result)
        with mp.workdps(70):
            convergence_rows.append({'x': result.x, 'order': result.order,
                'cutoff': result.cutoff, 'estimate': format(result.estimate, '.17g'),
                'numerical_error_vs_same_model': mp.nstr(mp.mpf(result.estimate)-model, 35),
                'numerical_truncation_bound': result.numerical_truncation_bound})
    sensitivity_rows = []
    for result in sensitivity:
        exact, _ = reference(result.x)
        sensitivity_rows.append({'x': result.x, **encode_error(result.estimate, exact),
                                 'details': asdict(result)})
    nonexact = [row for row in dense_rows if not row['exact_prefix_only']]
    worst = max(nonexact, key=lambda row: abs(float(row['relative_error'])), default=None)
    files = ('mvdc_prime_count.py', 'mvdc_utils.py', 'pi_mvdc_gui.py', 'experiments/compare_pi_mvdc.py',
             'tests/test_prime_mvdc.py', 'experiments/archive/pi_mvdc_factor_block_20261019.py')
    return {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'python_version': platform.python_version(), 'platform': platform.platform(),
        'classical_model': 'pi(M) + finite sum mu(k)/k [li(x**(1/k))-li(M**(1/k))]; no zeta zeros',
        'mvdc_role': 'Numerical centred Taylor evaluation via continuous central moments, the grid limit of finite C coefficients. The model is not attributed to MVDC.',
        'parameters_fixed_before_evaluation': {'alpha': .9, 'K_factor': 2,
            'max_cutoff': mvdc.DEFAULT_PREFIX_CAP, 'order': mvdc.DEFAULT_ORDER},
        'training': 'none; exact target pi(x) is used only for error measurement',
        'timing': 'Single cold in-process prefix/Mobius-cache call per selected x; reference computations are excluded. Not a comparison with optimized prime-counting libraries.',
        'error_separation': 'MVDC-versus-70-digit model error and 70-digit model-versus-exact-pi error are reported separately. Taylor bounds exclude roundoff and prime-model error.',
        'reference_sieve': {'limit': sieve_limit, 'seconds': sieve_seconds},
        'published_exact_references': {'source': REFERENCE_URL, 'checked_on': '2026-10-19', 'values': PUBLISHED_REFERENCE_ONLY},
        'rows': rows, 'convergence': convergence_rows, 'prefix_sensitivity': sensitivity_rows,
        'dense_rows': dense_rows, 'dense_summary': {'first_x': 2, 'last_x': dense_limit,
            'exact_prefix_cases_excluded_from_approximation_summary': [row['x'] for row in dense_rows if row['exact_prefix_only']],
            'approximate_cases': len(nonexact),
            'correct_after_rounding': sum(row['rounded_count_error'] == 0 for row in nonexact),
            'worst_x': worst['x'] if worst else None,
            'worst_relative_error': worst['relative_error'] if worst else None},
        'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--points', type=int, nargs='+', default=POINTS)
    parser.add_argument('--dense-limit', type=int, default=1000)
    args = parser.parse_args()
    report = collect(args.points, args.dense_limit)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('Saved:', args.output_json)


if __name__ == '__main__':
    main()
