"""Separate numerical MVDC accuracy, classical model accuracy and exact references."""
from fractions import Fraction
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import mpmath as mp
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import mvdc_prime_count as mvdc


def direct_tail(x, cutoff, terms):
    mu = mvdc.mobius_values(terms)
    return mp.fsum(mp.mpf(mu[k])/k*(mp.li(mp.root(x, k))-mp.li(mp.root(cutoff, k)))
                   for k in range(1, terms+1) if mu[k])


class FastTailTests(unittest.TestCase):
    def test_exact_tail_term_floors(self):
        for power in range(2, 60):
            for x in (2**power-1, 2**power, 2**power+1):
                self.assertEqual(mvdc.tail_term_count(x), (x*x).bit_length()-1)
        self.assertEqual(mvdc.floor_power(10**10, .9), 10**9)

    def test_short_mobius_table(self):
        mu = mvdc.mobius_values(100)
        for n in range(1, 101):
            self.assertEqual(sum(mu[d] for d in range(1, n+1) if n % d == 0), int(n == 1))

    def test_mvdc_panel_against_independent_quadrature(self):
        with mp.workdps(80):
            for k, centre, half in ((1, 1., .1), (3, 5., .5), (1, 25., .5), (31, 20., .25)):
                reference = mp.quad(lambda s: mp.exp((mp.mpf(centre)+s)/k)/(mp.mpf(centre)+s), [-half, half])
                for order in (0, 2, 6, 12, 20):
                    value, bound = mvdc._kernel_panel(k, centre, half, order)
                    self.assertLessEqual(abs(mp.mpf(value)-reference), bound+64*math.ulp(value))

    def test_finite_riemann_model_equivalence(self):
        with mp.workdps(80):
            for x, cutoff in ((3, 2), (10, 7), (1000, 501), (10**7, 10000), (10**12, 10000)):
                terms = mvdc.tail_term_count(x)
                result = mvdc.mvdc_log_tail(x, cutoff, terms)
                expected = direct_tail(x, cutoff, terms)
                self.assertLessEqual(abs(mp.mpf(result.value)-expected), result.truncation_bound+64*math.ulp(result.value))

    def test_close_endpoints_do_not_cancel(self):
        with mp.workdps(80):
            x, cutoff = 10**18, 10**18-1
            result = mvdc.mvdc_log_tail(x, cutoff, 1, 12)
            exact = mp.li(x)-mp.li(cutoff)
            self.assertGreater(result.value, 0)
            self.assertLess(abs(mp.mpf(result.value)/exact-1), mp.mpf('1e-14'))

    def test_mvdc_order_converges_to_the_model(self):
        with mp.workdps(80):
            x, cutoff, terms = 10000, 2, 26
            expected = direct_tail(x, cutoff, terms)
            errors = [abs(mp.mpf(mvdc.mvdc_log_tail(x, cutoff, terms, order).value)-expected)
                      for order in (0, 2, 4, 8)]
            self.assertTrue(all(b < a for a, b in zip(errors, errors[1:])))

    def test_prefix_is_bounded_for_huge_x(self):
        prime_calls, mu_calls = [], []
        original_prime, original_mu = mvdc.primes_up_to, mvdc.mobius_values
        def primes(limit):
            prime_calls.append(limit)
            return original_prime(limit)
        def mu(limit):
            mu_calls.append(limit)
            return original_mu(limit)
        with patch.object(mvdc, 'primes_up_to', side_effect=primes), \
             patch.object(mvdc, 'mobius_values', side_effect=mu), \
             patch.object(mvdc, 'prime_count_exact', side_effect=AssertionError('target leaked')), \
             patch('mpmath.li', side_effect=AssertionError('black-box evaluator')), \
             patch('mpmath.riemannr', side_effect=AssertionError('black-box evaluator')):
            result = mvdc.approximate_pi(10**12, return_details=True)
        self.assertEqual(prime_calls, [10000])
        self.assertEqual(mu_calls, [79])
        self.assertLess(result.panels, 100)
        self.assertGreater(result.kernel_panels, 0)
        self.assertFalse(result.exact_result)
        self.assertEqual(result.model, 'finite_riemann_type_tail')

    def test_small_inputs_have_no_pole_or_exact_fallback(self):
        with patch.object(mvdc, 'prime_count_exact', side_effect=AssertionError('fallback')):
            for x in range(3, 101):
                result = mvdc.approximate_pi(x, return_details=True)
                self.assertTrue(math.isfinite(result.estimate))
                self.assertGreater(result.estimate, 0)
                self.assertLess(result.cutoff, x)
                self.assertFalse(result.exact_result)
        boundary = mvdc.approximate_pi(2, return_details=True)
        self.assertEqual(boundary.estimate, 1)
        self.assertTrue(boundary.exact_result)
        self.assertEqual(boundary.panels, 0)

    def test_explicit_exact_prefix_is_labelled(self):
        result = mvdc.approximate_pi(100, cutoff=100, return_details=True)
        self.assertEqual(result.estimate, 25)
        self.assertTrue(result.exact_result)
        self.assertEqual(result.tail, 0)

    def test_auto_tolerance_is_numerical_not_count_accuracy(self):
        result = mvdc.approximate_pi_auto(10**6, 1e-8, return_details=True)
        self.assertTrue(result.numerical_tolerance_met)
        self.assertLessEqual(result.numerical_truncation_bound, 1e-8)
        self.assertGreater(abs(result.estimate-78498), 1)
        self.assertFalse(result.target_certified)

    def test_domains_and_work_limits(self):
        for x in (1, 0, -1, True, mvdc.MAX_X+1):
            with self.assertRaises(ValueError):
                mvdc.approximate_pi(x)
        for kwargs in ({'order': 3}, {'order': 32}, {'alpha': 1}, {'cutoff': 1},
                       {'max_cutoff': 5000001}, {'K_factor': .0001}, {'K_factor': 1e8}):
            with self.assertRaises(ValueError):
                mvdc.approximate_pi(1000, **kwargs)


if __name__ == '__main__':
    unittest.main()
