"""Bounded headless prime-count model comparison; importing runs no experiment."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mpmath as mp
from mvdc_prime_count import approximate_pi_auto, prime_count_exact


def test_mvdc_multiple():
    print("Exact-front plus Riemann-R tail: observed errors, not guaranteed targets")
    print("x          cutoff       estimate        exact       relative error")
    with mp.workdps(50):
        for x in (100, 1000, 10000, 100000, 1000000):
            result = approximate_pi_auto(x, target=1e-4, return_details=True)
            reference = prime_count_exact(x)
            error = abs(result.estimate-reference)/reference
            print(f"{x:10d} {result.cutoff:8d} {mp.nstr(result.estimate, 12):>15} "
                  f"{reference:10d} {mp.nstr(error, 9)}")


if __name__ == "__main__":
    test_mvdc_multiple()
