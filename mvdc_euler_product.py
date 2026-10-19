"""Compatibility entry point for the corrected Euler-product experiment."""
from experiments.mvdc_euler_product import (
    absolute_tail_bound, central_moments, experiment, experiment_tail,
    exponent, log_factor, mvdc_series, tail_integral, tail_mvdc,
)


if __name__ == "__main__":
    experiment_tail()
