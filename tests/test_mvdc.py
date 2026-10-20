"""Regression checks against exact identities and independent references."""
import importlib
import math
from fractions import Fraction
from pathlib import Path
import subprocess
import sys
import unittest

import mpmath as mp
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mvdc_utils as u
import gamma_ratio_mvdc as gamma
import binom_analytic as analytic_binom
import binom_mvdc as fitted_binom
import wallis_mvdc as wallis
import mvdc_factorial_analytic as factorial
import faktorial
import rising_mvdc_example as rising
import qpoch_mvdc_example as qpoch
import mobius_mvdc_example as mobius
from experiments import reference_pi_models as primes
from experiments import mvdc_euler_product as euler
from experiments import mvdc_dirichlet_tail as dirichlet
from experiments import gamma_mvdc_attempt as euler_constant
from experiments import mvdc_d_finiteness_test as recurrence


class NumericCase(unittest.TestCase):
    def setUp(self):
        self.context = mp.workdps(80)
        self.context.__enter__()

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def relative_close(self, value, reference, tolerance="1e-65"):
        self.assertLessEqual(u.relative_error(value, reference), mp.mpf(tolerance))


class CentreTests(NumericCase):
    def test_geometric_centre_and_log_product(self):
        k, log_h = u.mvdc_generic_center((2, 3, 7), order=2)
        self.assertAlmostEqual(log_h, math.log(42), places=14)
        self.assertAlmostEqual(k**3, 42, places=12)
        self.assertEqual(u.mvdc_generic_center((2, 3, 7), 1), (k, log_h))

    def test_minimax_factor_centre_and_bound(self):
        values = [mp.mpf(2), mp.mpf(3), mp.mpf(4)]
        center = u.factor_center(values)
        self.assertEqual(center, 3)
        approximation, bound = u.factor_moment_log(values, order=6)
        self.assertLessEqual(abs(approximation-mp.log(24)), bound)
        self.assertEqual(u.factor_moment_log([], order=6), (0, 0))
        with self.assertRaises(ValueError):
            u.factor_moment_log([1, 100], center=1)

    def test_invalid_centres(self):
        for values in ([], [0], [-1], [math.inf], [math.nan]):
            with self.assertRaises(ValueError):
                u.mvdc_generic_center(values)

    def test_constant_residual_is_not_overfit(self):
        ns = range(200, 2000, 200)
        model = u.fit_inverse_powers(ns, lambda n: mp.log(mp.pi)/2, 5)
        self.assertTrue(model.constant)
        self.assertEqual(len(model.coefficients), 1)
        self.relative_close(mp.exp(model(20)), mp.sqrt(mp.pi))

    def test_binary64_noise_is_not_treated_as_high_precision(self):
        fit = u.fit_inverse_powers(range(10, 80, 10), [1 + i*1e-16 for i in range(7)], 5)
        self.assertTrue(fit.constant)

    def test_scaled_qr_recovers_nonzero_intercept(self):
        target = lambda n: mp.mpf("0.7") + mp.mpf(3)/n - mp.mpf(2)/n**3
        fit = u.fit_inverse_powers((200, 400, 600, 800, 1000, 1200), target, 3)
        for n in (20, 350, 2000):
            self.relative_close(fit(n), target(n))
        self.assertTrue(fit.extrapolates(20))

    def test_fit_requires_valid_observations(self):
        for nodes in ((1, 1, 2), (1, 2)):
            with self.assertRaises(ValueError):
                u.fit_inverse_powers(nodes, lambda n: n, 2)
        with self.assertRaises(ValueError):
            u.fit_inverse_powers((1, 2, 3), (1, mp.nan, 2), 2)

    def test_fitted_center_reconstructs_log_value(self):
        model = u.fit_log_center((10, 20, 30, 40),
                                 lambda n: n + mp.mpf(2)/n, lambda n: mp.mpf(n), 2)
        self.relative_close(model.log_center(15)*15, mp.mpf(15)+mp.mpf(2)/15)


class GammaTests(NumericCase):
    def test_product_normalization_is_exact_identity(self):
        for n in (1, 2, 20, 100, 500, 2000):
            reference = mp.gamma(mp.mpf(n)+mp.mpf("0.5"))/mp.gamma(n)
            self.relative_close(gamma.gamma_ratio_product_mp(n), reference)
            self.relative_close(gamma.gamma_ratio_exact_mp(n), reference)

    def test_correct_stirling_baseline(self):
        for n in (20, 100):
            expected = mp.sqrt(n)*(1-mp.mpf(1)/(8*n)+mp.mpf(1)/(128*n*n))
            self.relative_close(gamma.gamma_ratio_stirling_mp(n), expected)

    def test_fitted_center_on_unseen_points(self):
        for n in (25, 75, 250, 750, 1250, 2500):
            self.assertNotIn(n, gamma.TRAINING_NS)
            self.relative_close(gamma.gamma_ratio_mvdc_mp(n), gamma.gamma_ratio_exact_mp(n), "1e-10")

    def test_gamma_domain_and_order(self):
        for n in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                gamma.gamma_ratio_mvdc_mp(n)
        with self.assertRaises(ValueError):
            gamma.gamma_ratio_stirling_mp(10, 21)


class ProductFamilyTests(NumericCase):
    def test_binomial_coefficients(self):
        self.assertEqual(u.central_binomial_coefficients(5),
                         (Fraction(1), Fraction(-1, 8), Fraction(1, 128),
                          Fraction(5, 1024), Fraction(-21, 32768), Fraction(-399, 262144)))

    def test_binomial_asymptotic_and_fitted_accuracy(self):
        for n in (20, 75, 125, 250, 1000):
            reference = math.comb(2*n, n)
            self.relative_close(analytic_binom.binom_mvdc_mp(n), reference, "1e-10")
            self.relative_close(fitted_binom.mvdc_fitted_mp(n), reference, "1e-9")
            self.relative_close(fitted_binom.mvdc_cascade2_mp(n), reference, "1e-10")

    def test_ratio_fit_normalization(self):
        coefficients = fitted_binom.fit_mvdc_coeffs()
        n = 125
        approximation = mp.exp(fitted_binom.main_log(n))*coefficients[0]*(1+mp.fsum(
            coefficients[k]/mp.mpf(n)**k for k in range(1, len(coefficients))))
        self.relative_close(approximation, math.comb(2*n, n), "1e-11")

    def test_binomial_zero_large_and_invalid(self):
        self.assertEqual(analytic_binom.binom_exact(0), 1)
        self.assertEqual(analytic_binom.binom_mvdc_mp(0), 1)
        self.assertTrue(math.isfinite(analytic_binom.binom_mvdc(512)))
        self.assertEqual(analytic_binom.binom_mvdc(1000), math.inf)
        with self.assertRaises(ValueError):
            analytic_binom.binom_mvdc_mp(-1)

    def test_empty_product_does_not_hide_invalid_order(self):
        for function in (analytic_binom.binom_mvdc_mp, fitted_binom.mvdc_fitted_mp,
                         wallis.wallis_asymptotic_mp, wallis.wallis_fitted_mp,
                         rising.mvdc_cascade_log):
            with self.subTest(function=function.__name__):
                with self.assertRaises(ValueError):
                    function(0, order=-1)

    def test_wallis_reference_and_first_coefficients(self):
        self.assertEqual(u.wallis_coefficients(2), (Fraction(1), Fraction(-1, 4), Fraction(5, 32)))
        for n in (0, 1, 2, 10, 50):
            exact = math.prod(Fraction(4*k*k, 4*k*k-1) for k in range(1, n+1))
            self.relative_close(wallis.wallis_exact_mp(n), u.number(exact))
            self.relative_close(wallis.wallis_product_mp(n), u.number(exact))

    def test_wallis_fitted_holdout_and_corrected_series(self):
        for n in (25, 75, 125, 250, 1000):
            reference = wallis.wallis_exact_mp(n)
            self.relative_close(wallis.wallis_fitted_mp(n), reference, "1e-8")
            self.relative_close(wallis.wallis_asymptotic_mp(n, 5), reference, "1e-8")

    def test_factorial_analytic_accuracy_and_domain(self):
        for n in (10, 20, 50, 100):
            self.relative_close(factorial.factorial_mvdc_mp(n), math.factorial(n), "1e-12")
        self.assertEqual(factorial.factorial_mvdc(0), 1)
        self.assertEqual(factorial.factorial_mvdc(1000), math.inf)
        self.assertTrue(mp.isfinite(factorial.log_factorial_mvdc_mp(10000)))
        with self.assertRaises(ValueError):
            factorial.factorial_mvdc(-1)
        with self.assertRaises(ValueError):
            factorial.factorial_mvdc_mp(10, order=21)

    def test_ramanujan_public_function_matches_sixth_root(self):
        n = mp.mpf(20)
        expected = mp.sqrt(mp.pi)*mp.power(n/mp.e, n)*mp.root(8*n**3+4*n**2+n+mp.mpf(1)/30, 6)
        self.relative_close(factorial.factorial_ramanujan_mp(20), expected)
        self.relative_close(factorial.factorial_ramanujan(20), expected, "2e-16")

    def test_factorial_cascade_is_actual_residual_cascade(self):
        model = faktorial.cascade_model()
        self.assertEqual(len(model.layers), 2)
        self.relative_close(model.value(17), math.factorial(17), "1e-9")

    def test_rising_factorial_includes_full_log_fit(self):
        for n in (0, 15, 25, 75, 125):
            reference = mp.rf(mp.mpf(1)/3, n)
            self.relative_close(mp.exp(rising.mvdc_cascade_log(n)), reference, "1e-8")

    def test_qpochhammer_uses_convergent_coordinate_and_bound(self):
        for n in (0, 1, 2, 5, 10, 20):
            expected = mp.log(mp.qp(mp.mpf("0.8"), mp.mpf("0.3"), n))
            self.assertLess(abs(qpoch.qpoch_exact_log(n)-expected), mp.mpf("1e-70"))
            error = abs(qpoch.mvdc_poly_log(n)-expected)
            self.assertLessEqual(error, qpoch.log_error_bound(n)+mp.mpf("1e-70"))

    def test_inverse_zeta_has_no_fitted_noise_and_checks_count(self):
        selected = mobius.selected_primes(100)
        expected = mp.log(mp.fprod(1-mp.mpf(1)/(p*p) for p in selected))
        self.assertLess(abs(mobius.mvdc_poly_log(100)-expected), mp.mpf("1e-65"))
        self.assertEqual(mobius.mvdc_poly_log(0), 0)
        self.assertTrue(all(c == 0 for c in mobius.fit_coeffs()))
        with self.assertRaises(ValueError):
            mobius.selected_primes(100000)


class TailAndExperimentTests(NumericCase):
    def test_euler_tail_omits_old_moment_series(self):
        result = euler.tail_mvdc(100, r_max=2)
        other = euler.tail_mvdc(100, r_max=6)
        approximation, block, diagnostics, model, _, limit = result
        self.relative_close(approximation, block+model)
        self.relative_close(approximation, other[0])
        self.assertNotEqual(diagnostics[2], 0)
        head = mp.fsum(euler.log_factor(p, 2) for p in u.primes_up_to(100))
        omitted = mp.log(mp.zeta(2))-head-block
        self.assertLessEqual(abs(omitted), euler.absolute_tail_bound(limit, 2))

    def test_dirichlet_infinite_reference_and_tail_bound(self):
        expected = mp.dirichlet(2, [0, 1, -1])
        self.relative_close(dirichlet.dirichlet_L(2), expected)
        value, block, diagnostic, _, _ = dirichlet.mvdc_tail_L(100, r_max=2, M_tail=1000)
        self.assertEqual(value, block)
        self.assertNotEqual(diagnostic, 0)
        head = mp.fsum(dirichlet.log_factor(p, 2) for p in u.primes_up_to(100))
        omitted = mp.log(expected)-head-block
        self.assertLessEqual(abs(omitted), euler.absolute_tail_bound(1000, 2))

    def test_real_tail_domains(self):
        with self.assertRaises(ValueError):
            euler.tail_mvdc(100, s=1)
        with self.assertRaises(ValueError):
            dirichlet.mvdc_tail_L(100, M_tail=50)
        with self.assertRaises(ValueError):
            dirichlet.chi_mod3(1.5)

    def test_euler_constant_series_and_acceleration(self):
        n = 30
        partial = euler_constant.gamma_partial(n)
        self.assertLess(abs(partial-(mp.harmonic(n)-mp.log(n+1))), mp.mpf("1e-65"))
        coefficients = euler_constant.coeffs_a(n)
        recovered = mp.fsum(c/mp.factorial(k) for k, c in enumerate(coefficients, 1))
        self.assertLess(abs(recovered-partial), mp.mpf("1e-65"))
        self.assertLess(abs(euler_constant.gamma_estimate(n)-mp.euler), mp.mpf("1e-17"))

    def test_finite_d_finiteness_certificate(self):
        for n in range(1, 9):
            self.assertTrue(recurrence.verify_recurrence(n))
        self.assertEqual(recurrence.c_coefficient(5, 2), Fraction(5*(25-1), 24))


class ReferencePrimeBackendTests(NumericCase):
    def test_small_exact_counts_and_power_boundaries(self):
        for x, expected in ((2, 1), (10, 4), (100, 25), (1000, 168)):
            result = primes.approximate_pi_auto(x, return_details=True)
            self.assertEqual(result.estimate, expected)
            self.assertTrue(result.target_certified)
        self.assertEqual(primes.prime_power_count(3, 243), 5)
        self.assertEqual(primes.prime_power_count(3, 242), 4)

    def test_modelled_tail_has_no_false_certificate(self):
        result = primes.approximate_pi_auto(100000, target=1e-4, max_cutoff=1000, return_details=True)
        self.assertEqual(result.cutoff, 1000)
        self.assertFalse(result.target_certified)
        self.assertLess(abs(result.estimate-9592)/9592, mp.mpf("0.01"))

    def test_auto_rule_is_bounded_for_tiny_targets(self):
        self.assertEqual(primes.select_cutoff(2, 1e-100), 2)
        self.assertEqual(primes.select_cutoff(10**12, 1e-300), 1000000)
        for target in (0, -1, 1, math.nan):
            with self.assertRaises(ValueError):
                primes.select_cutoff(100, target)

    def test_imports_are_headless_and_do_not_run_demos(self):
        modules = ["faktorial", "pi_mvdc_gui", "experiments.pi_mvdc_gui",
                   "experiments.test_mvdc_range", "gamma_ratio_mvdc", "wallis_mvdc", "binom_mvdc"]
        code = "import importlib; " + "; ".join(f"importlib.import_module({name!r})" for name in modules)
        process = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", code],
                                 cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout, "")


if __name__ == "__main__":
    unittest.main()
