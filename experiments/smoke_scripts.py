"""Run the numerical entry points with bounded timeouts and no GUI windows."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = (
    ("gamma_ratio_mvdc.py",), ("binom_analytic.py",), ("binom_mvdc.py",),
    ("wallis_mvdc.py",), ("mvdc_factorial_analytic.py",), ("faktorial.py",),
    ("rising_mvdc_example.py",), ("qpoch_mvdc_example.py",), ("mobius_mvdc_example.py",),
    ("mvdc_euler_product.py",), ("mvdc_dirichlet_tail.py",),
    ("experiments/mvdc_euler_product.py",), ("experiments/mvdc_dirichlet_tail.py",),
    ("experiments/gamma_mvdc_attempt.py",), ("experiments/mvdc_d_finiteness_test.py",),
    ("experiments/test_mvdc_range.py",),
    ("pi_mvdc_gui.py", "--x", "10000", "--cutoff", "1000", "--verify"),
    ("experiments/pi_mvdc_gui.py", "--x", "10000", "--cutoff", "1000", "--verify"),
)


def main():
    for command in COMMANDS:
        result = subprocess.run([sys.executable, "-X", "utf8", "-B", *command], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8", timeout=45)
        if result.returncode:
            print("FAILED:", " ".join(command))
            print(result.stdout)
            print(result.stderr)
            raise SystemExit(result.returncode)
        print("OK:", " ".join(command), flush=True)
    print(f"All {len(COMMANDS)} numerical entry points completed.")


if __name__ == "__main__":
    main()
