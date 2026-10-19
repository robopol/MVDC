"""Gamma-ratio comparisons: exact product, fitted centre, and correct series.

The original product requires only the exact normalization sqrt(pi). The fast
fitted branch uses sqrt(n) as its analytic main term and fits the genuine log
residual on training points only. All reported errors use independent 80-digit
references. The product and fitted branches have different evaluation costs.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform

import mpmath as mp
from mvdc_utils import (DEFAULT_DPS, central_binomial_coefficients,
                        central_binomial_log_coefficients, finite_float,
                        fit_log_center, integer, inverse_series, order_value,
                        precision, relative_error)

ALPHA, BETA = 0.5, 0.0
A1, A2 = -1/8, 1/128
TRAINING_NS = tuple(range(200, 2000, 200))
EVALUATION_NS = (20, 50, 100, 500, 1000, 2000)
HOLDOUT_NS = (25, 75, 150, 250, 750, 1250, 1750, 2500, 5000)
REFERENCE_DPS = DEFAULT_DPS


def gamma_ratio_exact_mp(n, dps=DEFAULT_DPS):
    integer(n, minimum=1)
    with mp.workdps(precision(dps)):
        # Exact integer central binomial coefficient supplies an independent reference.
        return n * mp.mpf(math.comb(2*n, n)) * mp.sqrt(mp.pi) / mp.power(4, n)


def gamma_ratio_exact(n):
    return finite_float(gamma_ratio_exact_mp(n))


def gamma_ratio_product_mp(n, dps=DEFAULT_DPS):
    """Finite-product branch; no polynomial fit of the constant sqrt(pi)."""
    integer(n, minimum=1)
    with mp.workdps(precision(dps)):
        log_h = -mp.log(2) + mp.fsum(mp.log1p(mp.mpf(1)/(2*k)) for k in range(1, n))
        return mp.sqrt(mp.pi) * mp.exp(log_h)


def _main_log(n):
    return mp.log(n) / 2


def _exact_log(n):
    return mp.loggamma(mp.mpf(n) + mp.mpf("0.5")) - mp.loggamma(n)


@lru_cache(maxsize=32)
def gamma_center_model(order=5, dps=DEFAULT_DPS):
    return fit_log_center(TRAINING_NS, _exact_log, _main_log, order, dps=dps)


def fit_mvdc_coeffs_gamma(order=5):
    """Return the fitted log-residual coefficients, including the intercept.

    These are log coefficients, not the old multiplicative noise-fit values.
    """
    return gamma_center_model(order).layers[0].inverse_coefficients()


def gamma_ratio_mvdc_mp(n, order=5, dps=DEFAULT_DPS):
    integer(n, minimum=1)
    order_value(order, len(TRAINING_NS)-1)
    with mp.workdps(precision(dps)):
        if order == 0:
            return mp.sqrt(n)
        return gamma_center_model(order, dps).value(n)


def gamma_ratio_mvdc(n, order=5):
    integer(n, minimum=1)
    order_value(order, len(TRAINING_NS)-1)
    if order == 0:
        return math.sqrt(n)
    correction = gamma_center_model(order).layers[0].evaluate_float(n)
    return math.sqrt(n)*math.exp(correction)


def gamma_ratio_stirling_mp(n, order=2, dps=DEFAULT_DPS):
    integer(n, minimum=1)
    with mp.workdps(precision(dps)):
        return mp.sqrt(n) * inverse_series(central_binomial_coefficients(order), n)


def gamma_ratio_stirling(n, order=2):
    integer(n, minimum=1)
    value = 0.0
    for coefficient in reversed(central_binomial_coefficients(order)):
        value = value/n + float(coefficient)
    return math.sqrt(n)*value


def gamma_ratio_product(n):
    integer(n, minimum=1)
    log_h = -math.log(2) + math.fsum(math.log1p(0.5/k) for k in range(1, n))
    return math.sqrt(math.pi)*math.exp(log_h)


def gamma_ratio_log_stirling(n, max_power=5):
    integer(n, minimum=1)
    correction = 0.0
    for coefficient in reversed(central_binomial_log_coefficients(max_power)):
        correction = correction/n + float(coefficient)
    return math.sqrt(n)*math.exp(correction)


def gamma_ratio_log_stirling_mp(n, max_power=5, dps=DEFAULT_DPS):
    integer(n, minimum=1)
    with mp.workdps(precision(dps)):
        return mp.sqrt(n) * mp.exp(inverse_series(central_binomial_log_coefficients(max_power), n))


def build_comparison(dps=DEFAULT_DPS):
    rows = []
    with mp.workdps(precision(dps)):
        model = gamma_center_model(5, dps)
        for n in (*EVALUATION_NS, *HOLDOUT_NS):
            reference = gamma_ratio_exact_mp(n, dps)
            direct_gamma = mp.gamma(mp.mpf(n) + mp.mpf("0.5")) / mp.gamma(n)
            if relative_error(direct_gamma, reference) > mp.power(10, -dps+10):
                raise ArithmeticError("independent Gamma references disagree")
            values = {
                "product": gamma_ratio_product_mp(n, dps),
                "fitted_center_5": model.value(n),
                "stirling_2": gamma_ratio_stirling_mp(n, 2, dps),
                "stirling_5": gamma_ratio_stirling_mp(n, 5, dps),
                "log_stirling_through_n5": gamma_ratio_log_stirling_mp(n, 5, dps),
            }
            errors = {key: relative_error(value, reference) for key, value in values.items()}
            native_values = {
                "product": gamma_ratio_product(n),
                "fitted_center_5": gamma_ratio_mvdc(n),
                "stirling_2": gamma_ratio_stirling(n, 2),
                "stirling_5": gamma_ratio_stirling(n, 5),
                "log_stirling_through_n5": gamma_ratio_log_stirling(n, 5),
            }
            rows.append({
                "n": n, "original_demo_point": n in EVALUATION_NS,
                "training_point": n in TRAINING_NS, "extrapolation": model.layers[0].extrapolates(n),
                "reference": mp.nstr(reference, dps),
                "relative_errors": {key: mp.nstr(value, 20) for key, value in errors.items()},
                "binary64_relative_errors": {
                    key: mp.nstr(relative_error(float(value), reference), 20)
                    for key, value in native_values.items()
                },
            })
        source = Path(__file__).resolve()
        return {
            "problem": "Gamma(n+0.5)/Gamma(n)", "reference_dps": dps,
            "python_version": platform.python_version(), "mpmath_version": mp.__version__,
            "training_n": list(TRAINING_NS), "holdout_n": list(HOLDOUT_NS),
            "fitted_log_coefficients": [mp.nstr(c, 35) for c in model.layers[0].inverse_coefficients()],
            "max_training_log_error": mp.nstr(model.layers[0].max_training_error, 20),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "mvdc_utils_sha256": hashlib.sha256(source.with_name("mvdc_utils.py").read_bytes()).hexdigest(),
            "notes": [
                "The fitted branch uses an analytic sqrt(n) main term and scaled high-precision QR.",
                "Product evaluation uses n-1 factors and is reported separately from fixed-order approximations.",
                "Stirling-2, Stirling-5 and log-Stirling through n^-5 are distinct declared baselines.",
                "Binary64 columns evaluate each formula using native float arithmetic; fitting and error references use 80 digits.",
                "No universal or equal-cost superiority follows from this finite accuracy comparison.",
            ], "rows": rows,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    report = build_comparison()
    print("Gamma ratio: native binary64 relative errors; fit and independent reference use 80 digits")
    print("n     Stirling-2    fitted-center-5   Stirling-5    product       training extrapolation")
    for row in report["rows"]:
        e = row["binary64_relative_errors"]
        print(f"{row['n']:5d} {float(e['stirling_2']):12.3e} {float(e['fitted_center_5']):16.3e} "
              f"{float(e['stirling_5']):12.3e} {float(e['product']):12.3e} "
              f"{str(row['training_point']):8s} {row['extrapolation']}")
    if args.output_json:
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
