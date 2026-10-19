# MVDC: centre decomposition and numerical experiments

This repository studies finite-product centring, residual expansions and
numerical approximation. The repaired Python examples distinguish three tasks:

1. **Exact finite-product evaluation.** The geometric log centre computes the
   full log product. Its residual is zero; a polynomial fit of roundoff is not
   a new asymptotic correction.
2. **Fitted analytic centres.** Gamma ratios, central binomial coefficients,
   Wallis products, factorials and rising factorials use explicit main terms.
   Genuine log residuals are fitted in scaled inverse powers using 80-digit
   Householder QR. Each fit records its training interval and error. Predictions
   outside the interval are reported as extrapolation.
3. **Tail models.** Euler and Dirichlet experiments separate an exact finite
   prime block, an optional tail approximation and a rigorous absolute bound.

The generic log centre is the geometric mean; the former variance shifts did
not follow from moment minimization. For a convergent factor-moment expansion,
`factor_center(..., criterion="minimax")` minimizes the maximum relative factor
deviation. The expansion requires `max(abs(a_i/k - 1)) < 1` and returns an
explicit truncation bound.

## Install

Python 3.10 or newer is required.

```sh
python -m pip install -r requirements.txt
```

Only `mpmath` is required for the numerical examples. The standard-library
`tkinter` module is needed only to open the optional GUI. On Windows it is
normally supplied by the Python installer; it is not installed with pip.

## Numerical examples

```sh
python -X utf8 -B gamma_ratio_mvdc.py
python -X utf8 -B binom_analytic.py
python -X utf8 -B binom_mvdc.py
python -X utf8 -B wallis_mvdc.py
python -X utf8 -B mvdc_factorial_analytic.py
python -X utf8 -B faktorial.py
python -X utf8 -B rising_mvdc_example.py
python -X utf8 -B qpoch_mvdc_example.py
python -X utf8 -B mobius_mvdc_example.py
```

The Gamma example retains the original nine training points
`200, 400, ..., 1800`. The six original evaluation points remain visible, and
additional points are held out of training. `n=1000` is explicitly marked as a
training point. The report separates:

- the normalized full product, whose correction is exactly `sqrt(pi)`;
- the fast fitted log centre, evaluated without the full product;
- correct multiplicative Stirling expansions through `1/n^2` and `1/n^5`;
- the classical log expansion through `1/n^5`;
- high-precision evaluation and native binary64 evaluation.

The polynomial fit includes its intercept. Orders 3 and 5 are fitted
independently. A cascade fits the second layer to the first layer's actual
remaining residual; its higher degree is stated explicitly. None of these
finite measurements establishes universal or equal-cost superiority.

`binom_analytic.py` generates exact rational coefficients from the log series.
`mvdc_factorial_analytic.py` uses classical Bernoulli log corrections. Its
public Ramanujan function and its demo use the same sixth-root formula.
Binary64 APIs return `inf` on overflow; high-precision and log APIs remain
available for large products. Negative indices and unsupported orders raise
errors instead of silently returning a plausible value.

The q-Pochhammer example uses its natural `q**N` coordinate and a convergent
analytic residual series with a bound, rather than fitting a near-zero
residual to `1/N`. The finite inverse-zeta example performs full finite-product
evaluation with zero log residual; it does not claim to approximate the
uncomputed infinite tail.

## Euler products and corrected diagnostics

```sh
python -X utf8 -B experiments/mvdc_euler_product.py
python -X utf8 -B experiments/mvdc_dirichlet_tail.py
python -X utf8 -B experiments/gamma_mvdc_attempt.py
python -X utf8 -B experiments/mvdc_d_finiteness_test.py
```

The repository-root `mvdc_euler_product.py` and `mvdc_dirichlet_tail.py`
entry points delegate to the corrected implementations in `experiments/`.

- Euler-product estimates use the exact finite block plus a prime-density
  integral. The integral is a model; the absolute tail bound is separate.
- Dirichlet `L(s, chi_3)` is evaluated through a Hurwitz-zeta identity. Finite
  character sums have a separately named function. No unchecked `1e-10`
  remainder guarantee is used.
- Historical central-log-moment expressions may be printed as diagnostics,
  but they are never added to an already exact finite logarithmic block.
- The Euler-constant experiment now uses the convergent series
  `sum(1/k - log(1+1/k))` and a standard Euler-Maclaurin acceleration. Its
  factorial-scaled coefficients are real numbers, not exact rationals.
- The D-finiteness script proves an exact recurrence for each fixed finite
  centred-index set. It makes no inference about an infinite prime sequence.

## Prime-counting research utility

```sh
python -X utf8 -B pi_mvdc_gui.py
python -X utf8 -B pi_mvdc_gui.py --x 100000 --cutoff 10000 --verify
python -X utf8 -B experiments/test_mvdc_range.py
```

Importing the GUI never opens a window. The former Bernoulli-denominator
conversion and artificial log-moment correction had no established prime-count
error bound. They have been replaced in this utility by an explicit exact-front
plus Riemann-R (or logarithmic-integral) tail model. This is a declared reference
model, not a claimed new MVDC theorem. The effort target selects a bounded exact
cutoff; it does not guarantee a relative error for the modelled tail.

The built-in exact sieve is capped at five million. The GUI and CLI reject
inputs outside the documented range, and an exact-reference comparison is
optional. `experiments/pi_mvdc_gui.py` delegates to the same backend and GUI.

## Verification and reproducible data

```sh
python -X utf8 -B -m unittest discover -s tests -v
python -X utf8 -B experiments/verify_c_bernoulli_derivation.py
python -X utf8 -B experiments/verify_euler_tail_corrections.py
python -X utf8 -B experiments/smoke_scripts.py
python -X utf8 -B experiments/compare_accuracy.py --output-json reports/mvdc-validation-2026-10-19.json
```

Regression checks use exact integer or rational identities, independent
high-precision references, held-out points, domain checks and import-safety
checks. The two earlier verification scripts remain independent checks of the
corrected paper's formulas and tables. Reports record precision, training
points, runtime-library versions and source hashes. `-B` avoids generating
bytecode caches during these runs.

## Corrected publications

The current mathematical papers are dated **11 August 2026**:

- [English PDF](c_bernoulli_derivation.pdf) and
  [LaTeX source](c_bernoulli_derivation.tex).
- [Slovak PDF](c_bernoulli_derivation_sk.pdf) and
  [LaTeX source](c_bernoulli_derivation_sk.tex).

The English PDF is byte-identical to the corrected paper published on
12 August 2026 in [Zenodo record 21898952](https://zenodo.org/records/21898952).
Both PDFs distinguish exact centre identities, factor moments, finite centred
index coefficients, and the modelled or bounded infinite tails.

The **19 October 2026 numerical update** is supplied separately in
[the accuracy report](reports/mvdc-validation-2026-10-19.json) and
[the research article](https://robopol.sk/research/mvdc-method.html).
The August PDFs do not contain these later fitted-centre measurements.

The superseded 2025 general papers (`mvdc_publication_en.pdf` and
`mvdc_publication_sk.pdf`) have been removed from the current branch because
their centre-optimality and accuracy claims were incorrect or unsupported.
Earlier versions remain recoverable in Git history.

References for the classical components:

- [NIST DLMF: Gamma asymptotic expansions](https://dlmf.nist.gov/5.11)
- [NIST DLMF: sums of powers](https://dlmf.nist.gov/24.4)
- [NIST DLMF: generalized Bernoulli polynomials](https://dlmf.nist.gov/24.16)

MIT License, copyright 2025 Ing. Robert Polak.
