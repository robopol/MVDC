# MVDC – Mean Value Decomposition by Centre

MVDC is a universal method for fast, high-accuracy asymptotics of *positive* finite products

\[
P = \prod_{i=1}^{m} a_i, \qquad a_i>0 .
\]

Instead of classical Taylor / Stirling / Bernoulli expansions, MVDC  
1. automatically chooses an **optimal centre** \(k\) from the first two log-moments,  
2. returns the **main term** \(H=k^m\) and a short polynomial correction  
   \(C_0\,(1+C_1/m+\dots+C_p/m^p)\),  
3. optionally adds a **log-cascade** layer that drives the error down to machine precision.

Key features  
* works entirely in log-domain → no overflow for \(m\gtrsim10^6\),  
* reaches ≥ 10 extra digits compared to the same-length Stirling/Taylor series,  
* single algorithm for factorials, Gamma/Barnes functions, \(q\)-products, Euler products, …  

---

## Folder structure

| File / folder | Description |
| ------------- | ----------- |
| `mvdc_utils.py` | Core routine: centre selection + main term \(H\). |
| `wallis_mvdc.py` | Wallis product: MVDC (H, H+5) vs. classical expansion. |
| `binom_mvdc.py` | Central binomial coefficient \(\binom{2n}{n}\). |
| `gamma_ratio_mvdc.py` | **New** example: ratio \(\Gamma(n+0.5)/\Gamma(n)\) – MVDC vs. Stirling. |
| `rising_mvdc_example.py` | Rising factorial \(n^{\underline k}\) with \(k\approx n/2\). |
| `qpoch_mvdc_example.py` | \(q\)-Pochhammer \((0.8,0.3)_N\). |
| `mobius_mvdc_example.py` | Truncated Euler product \(\prod_{p\le P}(1-p^{-2})\). |
| `docs/mvdc_publication_en.pdf` | Full English paper (arXiv style). |
| `docs/mvdc_publication_sk.pdf` | Slovak paper. |

---

## Quick install

```bash
git clone https://github.com/<your-account>/<repo>.git
cd <repo>/MVDC
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
# .\venv\Scripts\activate
pip install sympy
```

---

## Run the examples

```bash
python wallis_mvdc.py
python binom_mvdc.py
python gamma_ratio_mvdc.py
```

Each script prints a table:

* **Exact** – reference value (`math` / high precision),
* **Stirling / Taylor** – classical expansion,
* **MVDC H, H+5** – main term + polynomial cascade,
* relative error.

---

## Using MVDC for *your* product

```python
from mvdc_utils import mvdc_generic_center
# build your list of positive factors
factors = [...]
_, ln_H = mvdc_generic_center(factors, order=2)   # main term (log)
# optional: fit C0..Cp once on a training range
```

Full pseudocode and fitting details are given in the PDF paper (Section *Algorithm + Cascade*).

---

## License

MIT License © 2025 Ing. Robert Polak