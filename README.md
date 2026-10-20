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

## Fast prime-count approximation: classical model, MVDC evaluator

The model and numerical evaluator are explicitly separated. The model is a
classical **finite Riemann-type approximation**, anchored by a bounded known
prefix. MVDC supplies a **centred numerical expansion of the smooth tail**.
The output is not an exact prime count and is not evidence that MVDC itself
predicts the unknown distribution of primes.

### Consistent tail and conversion

Write `theta(t)=sum_{p<=t} log(p)` and
`Theta_K(t)=sum_{k=1}^K mu(k)*t**(1/k)`. The latter is a smooth model suggested
by the classical Mobius relation between theta and Chebyshev psi, with the
leading approximation psi(t) ~ t. The omitted prime-distribution fluctuations
remain model error. Correct conversion uses `1/log(t)` inside a Stieltjes
integral; it does not divide the entire weighted sum by a guessed scalar:

\[
\pi(x)-\pi(M)=\int_{(M,x]}\frac{d\theta(t)}{\log t}.
\]

Substituting the smooth model and using u=log(t) gives

\[
\widehat\pi_K(x;M)=\pi(M)+
\sum_{k=1}^K\frac{\mu(k)}k
\int_{\log M}^{\log x}\frac{e^{u/k}}u\,du.
\]

This is identically the finite classical expression

\[
\pi(M)+\sum_{k=1}^K\frac{\mu(k)}k
\left[\operatorname{li}(x^{1/k})-\operatorname{li}(M^{1/k})\right].
\]

This equivalence is stated rather than hidden. The production estimator uses
the centred numerical expansion below; a separate 70-digit li evaluation
checks that numerical implementation. Neither finite K nor the numerical
error bound certifies accuracy against the true pi(x). The finite smooth
model is also not identical to an infinite Riemann-R evaluation.

The old Bernoulli denominator is removed: the correct local divisor is u in
the integral, where u>=log(2)>0. The old mixture of a prime-power weighted
block with a differently weighted root tail is not retained.

### What the MVDC calculation actually does

The finite C-Bernoulli identity in the paper expands equally spaced sums
about their midpoint. For a midpoint grid in [c-h,c+h], its Riemann-sum limit
has coefficients

\[
\int_{c-h}^{c+h}g(u)\,du=
\sum_{j\ge0}\frac{2h^{2j+1}}{(2j+1)!}g^{(2j)}(c).
\]

Indeed, `(2h/n)^(r+1)*C_r(n)` tends to `2*h^(r+1)/(r+1)!` for even r;
odd moments vanish. This is the continuous centred-moment form of the MVDC
apparatus, mathematically a Taylor quadrature. It is not claimed as a new
prime-density law or as the discrete C coefficient itself.

For `g_k(u)=exp(u/k)/u`, let u=c(1+v), A=c/k and rho=h/c. The coefficients of
`exp(A*v)/(1+v)` satisfy `q_0=1`, `q_r=A^r/r!-q_(r-1)`. A degree-2p panel is

\[
\frac{2h e^{c/k}}c\sum_{j=0}^p\frac{q_{2j}\rho^{2j}}{2j+1}.
\]

Panels satisfy h/c<=1/8 and h<=1/2. Cauchy's coefficient bound on a circle of
radius R_s around c, with h<R_s<c, gives a panel truncation bound

\[
\frac{2h e^{(c+R_s)/k}}{c-R_s}
\frac{(h/R_s)^{2p+2}}{(2p+3)(1-(h/R_s)^2)}.
\]

The implementation uses `R_s=min(3c/4,k*(2p+2))` and sums absolute panel bounds
with weights abs(mu(k))/k. This bounds numerical Taylor truncation in exact
arithmetic only. Roundoff and the model's discrepancy from pi(x) are separate.
Close endpoints use log1p and a log-span coordinate to avoid cancellation.

### Work budget and verification

Defaults, fixed before comparison: alpha=0.9, M=min(floor(x^alpha),10000),
K=floor(2*log2(x)), and numerical degree 12 (seven even terms per panel).
The boundary x=2 has M=x and is explicitly labelled exact-prefix-only.
Every default input x>2 has M<x. No exact-target fallback, long Mobius table,
zeta-zero sum or fitting to the target count is used. The short Mobius weights
are classical and are not attributed to MVDC. Explicit prefix overrides are
reported, including the case M=x.

```sh
python -X utf8 -B pi_mvdc_gui.py --x 10000 --order 12 --verify
python -X utf8 -B pi_mvdc_gui.py --x 1000000000000 --order 12
python -X utf8 -B experiments/compare_pi_mvdc.py --output-json reports/pi-mvdc-fast-tail-2026-10-19.json
```

`--max-cutoff` controls exact prefix work; `--cutoff` sets it explicitly.
`--numerical-tolerance` adapts only the evaluation order, not the error against
pi(x). The obsolete beta and scalar-denominator controls were removed.

The report separates three quantities: total error against exact pi(x),
classical-model error from a 70-digit evaluation, and the MVDC numerical
evaluation error. It includes a fixed order-convergence grid, fixed prefix
budgets, cold-call timings, and source hashes. Local exact counts through
10^7 come from an independent sieve after predictions are fixed. Larger
reference values are published exact counts, not values computed by MVDC.
Exact-prefix-only cases are excluded from approximation-success summaries.
No runtime superiority over optimized prime-counting libraries is claimed.

Sources:

- [NIST: Chebyshev psi](https://dlmf.nist.gov/25.16#E1)
- [NIST: Mobius inversion](https://dlmf.nist.gov/27.5)
- [Finite Riemann-type model notation](https://arxiv.org/abs/1410.1083)
- [Published exact prime-count references](https://t5k.org/howmany.html)

### Comparison baseline

The comparison script includes the previous factor-block approximation in
`experiments/archive/pi_mvdc_factor_block_20261019.py`. Its original tail
and denominator remain unproved model choices; it is a comparison baseline
and never computes the active approximation.

The excluded long-Mobius exact-count experiment is not part of this published
implementation and must not be cited as accuracy achieved by this approximation.

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
