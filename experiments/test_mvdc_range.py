"""Compare a finite Riemann-type model evaluated by MVDC with exact counts."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mvdc_prime_count import approximate_pi, prime_count_exact


def test_mvdc_multiple():
    print('Model: finite Riemann-type tail. Numerical evaluator: centred MVDC moments.')
    print('x          M          approximation     exact pi    relative error')
    for x in (3, 10, 100, 1000, 10000):
        result = approximate_pi(x, return_details=True)
        exact = prime_count_exact(x)
        print(f'{x:10d} {result.cutoff:8d} {result.estimate:17.9f} {exact:10d} {abs(result.estimate/exact-1):.9g}')


if __name__ == '__main__':
    test_mvdc_multiple()
