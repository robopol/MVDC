"""Export reproducible held-out accuracy data without changing website content."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import mpmath as mp
import binom_analytic
import binom_mvdc
import gamma_ratio_mvdc
import wallis_mvdc
import mvdc_factorial_analytic
from mvdc_utils import DEFAULT_DPS, relative_error


def collect():
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "working_precision_dps": DEFAULT_DPS,
              "gamma_ratio": gamma_ratio_mvdc.build_comparison(),
              "binomial": [], "wallis": [], "factorial": [],
              "scope": "Finite accuracy measurements with declared orders and training points; no website or universal claim is changed."}
    with mp.workdps(DEFAULT_DPS):
        for n in (25, 75, 125, 255, 750, 1250):
            reference = math.comb(2*n, n)
            report["binomial"].append({
                "n": n, "training_point": n in binom_mvdc.TRAINING_NS,
                "extrapolation": binom_mvdc.center_model().layers[0].extrapolates(n),
                "fitted_log_center_5_error": mp.nstr(relative_error(binom_mvdc.mvdc_fitted_mp(n), reference), 20),
                "cascade_3_7_error": mp.nstr(relative_error(binom_mvdc.mvdc_cascade2_mp(n), reference), 20),
                "analytic_series_5_error": mp.nstr(relative_error(binom_analytic.binom_mvdc_mp(n), reference), 20),
            })
            w = wallis_mvdc.wallis_exact_mp(n)
            report["wallis"].append({
                "n": n, "training_point": n in wallis_mvdc.TRAINING_NS,
                "extrapolation": wallis_mvdc.center_model().layers[0].extrapolates(n),
                "fitted_log_center_5_error": mp.nstr(relative_error(wallis_mvdc.wallis_fitted_mp(n), w), 20),
                "analytic_series_5_error": mp.nstr(relative_error(wallis_mvdc.wallis_asymptotic_mp(n, 5), w), 20),
            })
        for n in (10, 20, 50, 100, 500):
            reference = math.factorial(n)
            report["factorial"].append({
                "n": n,
                "log_stirling_through_n7_error": mp.nstr(relative_error(mvdc_factorial_analytic.factorial_mvdc_mp(n), reference), 20),
                "ramanujan_sixth_root_error": mp.nstr(relative_error(mvdc_factorial_analytic.factorial_ramanujan_mp(n), reference), 20),
            })
    report["source_sha256"] = {
        name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ("mvdc_utils.py", "gamma_ratio_mvdc.py", "binom_analytic.py", "binom_mvdc.py",
                     "wallis_mvdc.py", "mvdc_factorial_analytic.py")
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    report = collect()
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        print("Saved:", args.output_json)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
