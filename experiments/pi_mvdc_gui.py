"""Compatibility entry point for the attributed approximation GUI/CLI."""
from pathlib import Path
import runpy
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mvdc_prime_count import approximate_pi, approximate_pi_auto, mvdc_log_tail
if __name__ == '__main__':
    runpy.run_path(str(ROOT/'pi_mvdc_gui.py'), run_name='__main__')
