"""Compatibility entry point for the corrected Dirichlet-tail experiment."""
from experiments.mvdc_dirichlet_tail import (
    absolute_tail_bound, central_moments_dirichlet, chi_mod3, dirichlet_L,
    dirichlet_L_partial, experiment, exponent, log_factor, mvdc_series, mvdc_tail_L,
)


if __name__ == "__main__":
    experiment()
