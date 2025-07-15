# MVDC – Mean Value Decomposition by Centre

MVDC is a universal method for fast, high-accuracy asymptotics of *positive* finite products

$$
P = \prod_{i=1}^{m} a_i,\qquad a_i>0 .
$$

Instead of classical Taylor / Stirling / Bernoulli expansions, MVDC  
1. automatically chooses an **optimal centre** $k$ from the first two log-moments,  
2. returns the **main term** $H = k^m$ and a short polynomial correction  
   $C_0\,(1 + C_1/m + \dots + C_p/m^p)$,  
3. optionally adds a **log-cascade** layer that drives the error down to machine precision.

Key features  
* works entirely in log-domain → no overflow for $m\gtrsim10^6$,  
* reaches ≥ 10 extra digits compared to the same-length Stirling/Taylor series,  
* single algorithm for factorials, Gamma/Barnes functions, $q$-products, Euler products, …  

---

## Folder structure

| File / folder | Description |
| ------------- | ----------- |
| **Core Library** | |
| `mvdc_utils.py` | Core routine: centre selection + main term \(H\). |
| **Basic Examples** | |
| `wallis_mvdc.py` | Wallis product: MVDC (H, H+5) vs. classical expansion. |
| `binom_mvdc.py` | Central binomial coefficient $\binom{2n}{n}$. |
| `binom_analytic.py` | Analytic binomial coefficient approximation. |
| `gamma_ratio_mvdc.py` | Ratio $\Gamma(n+0.5)/\Gamma(n)$ – MVDC vs. Stirling. |
| `rising_mvdc_example.py` | Rising factorial $n^{\underline{k}}$ with $k\approx n/2$. |
| `qpoch_mvdc_example.py` | \(q\)-Pochhammer \((0.8,0.3)_N\). |
| `mobius_mvdc_example.py` | Truncated Euler product \(\prod_{p\le P}(1-p^{-2})\). |
| **Advanced Examples** | |
| `mvdc_factorial_analytic.py` | **New**: Fully analytic factorial approximation using Bernoulli numbers. |
| `faktorial.py` | Factorial computations and comparisons. |
| **Experiments** | |
| `experiments/mvdc_dirichlet_tail.py` | **New**: Dirichlet L-functions and character analysis. |
| `experiments/mvdc_euler_product.py` | **New**: Advanced Euler product convergence studies. |
| `experiments/mvdc_d_finiteness_test.py` | **New**: D-finiteness testing for sequences. |
| `experiments/pi_mvdc_gui.py` | **New**: Interactive GUI for π(x) prime counting approximations. |
| **Documentation** | |
| `mvdc_publication_en.pdf` | Full English paper (arXiv style). |
| `mvdc_publication_sk.pdf` | Slovak version of the paper. |
| `c_bernoulli_derivation.pdf` | **New**: Bernoulli derivation theory (English). |
| `c_bernoulli_derivation_sk.pdf` | **New**: Bernoulli derivation theory (Slovak). |

---

## Quick install

```bash
git clone https://github.com/robopol/MVDC.git
cd MVDC
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
# .\venv\Scripts\activate
pip install sympy mpmath
```

**Additional dependencies for GUI experiments:**
```bash
pip install tkinter  # Usually included with Python
```

---

## Run the examples

### Basic Examples
```bash
python wallis_mvdc.py
python binom_mvdc.py
python gamma_ratio_mvdc.py
python mvdc_factorial_analytic.py
```

### Interactive Experiments
```bash
# Launch GUI for prime counting function approximations
python experiments/pi_mvdc_gui.py

# Advanced mathematical experiments
python experiments/mvdc_dirichlet_tail.py
python experiments/mvdc_euler_product.py
```

Each script prints a table:

* **Exact** – reference value (`math` / high precision),
* **Stirling / Taylor** – classical expansion,
* **MVDC H, H+5** – main term + polynomial cascade,
* relative error.

---

## New Features

### 🆕 Analytic Factorial Approximation
The `mvdc_factorial_analytic.py` module provides a fully analytic MVDC approximation for factorials using exact Bernoulli number coefficients - no numeric fitting required!

### 🆕 Advanced Experiments
- **Dirichlet L-functions**: Analyze character modulo operations and convergence
- **Euler Products**: Advanced convergence studies for infinite products  
- **Prime Counting GUI**: Interactive tool for π(x) approximations with real-time visualization
- **D-finiteness Testing**: Sequence analysis for mathematical properties

### 🆕 Extended Documentation
- Additional Bernoulli derivation papers in both English and Slovak
- Enhanced theoretical background and mathematical proofs

---

## Using MVDC for *your* product

```python
from mvdc_utils import mvdc_generic_center
# build your list of positive factors
factors = [...]
_, ln_H = mvdc_generic_center(factors, order=2)   # main term (log)
# optional: fit C0..Cp once on a training range
```

Full pseudocode and fitting details are given in the PDF papers (Section *Algorithm + Cascade*).

---

## Examples by Application Domain

| Domain | Files | Description |
|--------|-------|-------------|
| **Combinatorics** | `binom_*.py` | Binomial coefficients, analytic and numeric |
| **Number Theory** | `experiments/mvdc_euler_product.py`<br>`experiments/mvdc_dirichlet_tail.py` | Prime products, L-functions |
| **Special Functions** | `gamma_ratio_mvdc.py`<br>`mvdc_factorial_analytic.py` | Gamma functions, factorials |
| **q-Analogs** | `qpoch_mvdc_example.py` | q-Pochhammer symbols |
| **Interactive Tools** | `experiments/pi_mvdc_gui.py` | GUI applications |

---

## Performance Notes

- Works efficiently for products with $m \gtrsim 10^6$ terms
- Memory usage scales linearly with number of factors
- GPU acceleration possible for large-scale applications (future work)

---

## License

MIT License © 2025 Ing. Robert Polak

---

## Citation

If you use MVDC in your research, please cite:

```bibtex
@article{polak2025mvdc,
  title={MVDC: Mean Value Decomposition by Centre for High-Accuracy Asymptotics},
  author={Polak, Robert},
  year={2025},
  journal={arXiv preprint},
  note={Available at: https://github.com/robopol/MVDC}
}
```