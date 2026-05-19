# Distributionally Robust Offline Pricing: OPE and OPL

Implementation of **Offline Policy Evaluation (OPE)** and **Offline Policy Learning (OPL)** for continuous pricing, with synthetic simulations and Expedia real-data experiments.

## Algorithms

| Name | Role | Reference |
|------|------|-----------|
| **LDR²PE-CP** | OPE: localized doubly robust DR policy evaluation | This repo (Algorithm 1) |
| **DRO-IPW** | OPE: DR-IPW style evaluator | Leung et al. (2025) |
| **CDR²O²PL-CP (DRO)** | OPL: continuum doubly robust DRO policy learning | `offline_pricing.tex` |
| **Leung2025** | OPL baseline | Leung et al. (2025) |
| **Ai2024** | OPL non-robust baseline (double debiasing) | Ai et al. (2026) |

Shared code lives under `src/` (DGP, learners, estimators, visualization). Experiment design (\(\delta\), sample sizes, bandwidth) is configured under `experiments/ope/` and `experiments/opl/`, not in `src/config.py`.

---

## Installation

**Requirements:** Python 3.8+ (3.7 may work; not tested recently).

```bash
git clone <repository-url>
cd Code_Delta_Github
pip install -r requirements.txt
```

Run scripts from the **repository root** so paths like `RealData/data/train.csv` resolve correctly:

```bash
python experiments/ope/run_main_OPE.py
python experiments/opl/run_main_OPL_synthetic.py
python experiments/opl/run_main_OPL_expedia.py
```

Outputs are written to the **current working directory** (typically the repo root unless you `cd` elsewhere).

---

## Project structure

```
Code_Delta_Github/
├── README.md
├── requirements.txt
├── src/                          # Core library
│   ├── config.py                 # Algorithm / DGP defaults (synced at runtime)
│   ├── dgp.py                    # Synthetic pricing environment
│   ├── policy.py                 # Linear / MLP pricing policies
│   ├── estimators.py             # Propensity, outcome, bandwidth helpers
│   ├── learner.py                # OPE / OPL learners
│   ├── visualization.py          # Figure 1 (OPE) and Figure 2 (OPL) plots
│   └── realdata_expedia.py       # Expedia CSV loading and splits
├── experiments/
│   ├── ope/                      # Synthetic OPE only
│   │   ├── config_ope.py
│   │   ├── ope_common.py
│   │   ├── ope_plotting.py
│   │   ├── run_main_OPE.py
│   │   └── run_bandwidth_sensitivity.py
│   └── opl/
│       ├── config_synthetic.py   # Synthetic OPL
│       ├── config_expedia.py     # Expedia OPL
│       ├── opl_common.py
│       ├── opl_plotting.py
│       ├── run_main_OPL_synthetic.py
│       ├── run_main_OPL_expedia.py
│       └── run_bandwidth_sensitivity.py
└── results/                     
```

---

## Robustness radius: train δ vs test δ

Both OPE and OPL separate:

- **\(\delta_{train}\)** — used when fitting nuisances / training robust learners (cross-fitting, dual problems).
- **\(\delta_{test}\)** — used when **evaluating** the distributionally robust objective (\(R_\delta\) or \(\hat R_{adv} \)).

This matches the paper design: train at a fixed ambiguity level, report performance at several test radii.

**Ai et al. (2026) (OPL only):** training is **non-robust**. For comparability on synthetic data, \(R_\delta\) at evaluation still uses `get_optimal_alpha(..., delta_test)` with the same \(\delta_{test}\) as DRO methods.

---

## Experiment 1: Synthetic OPE

**Goal:** Estimate the robust value \(R_\delta\) of a **fixed** target policy on synthetic offline data; compare **LDR²PE-CP** vs **DRO-IPW** by MSE against a Monte Carlo ground truth.

| Setting | Value |
|---------|--------|
| Data | `PricingEnvironment` (linear setting, nonlinear logging policy) |
| \(\delta_{train}\) | 0.2 (nuisance / cross-fit) |
| \(\delta_{test}\) | 0.1, 0.2, 0.3 |
| Offline sizes T | 500, 1000, 2000, 3000, 4000, 5000 |
| True \(R_\delta\) | LDR²PE-CP on **2500** held-out samples (`N_TEST`) |
| Repeats | 20 |
| Bandwidth (default) | Silverman: \(h = 1.06 \cdot \mathrm{std}(P) \cdot T^{-1/5}\) |

**Config:** `experiments/ope/config_ope.py`  
- Key fields: `DELTA_TRAIN`, `TEST_DELTAS`, `TRAIN_SIZES`, `N_TEST`, `BANDWIDTH_MODE`, `N_REPEATS`, `TARGET_POLICY_SEED`.

**Run:**

```bash
python experiments/ope/run_main_OPE.py
```

**Outputs (repo root by default):**

| File | Description |
|------|-------------|
| `OPE_Figure1_linear.pdf` | MSE vs T (Figure 1), three panels by \(\delta_{test}\) |

**Bandwidth sensitivity** (Silverman \(n^{-1/5.0}\) vs \(n^{-1/4.9}\) vs \(n^{-1/2.9}\) rules):

```bash
python experiments/ope/run_bandwidth_sensitivity.py
```

Edit `experiments/ope/config_bandwidth_sensitivity.py` (`BANDWIDTH_MODES`) or pass `bandwidth_mode=` in code. Main run bandwidth is set via `BANDWIDTH_MODE` in `config_ope.py` (or `run_ope_experiments(exp_cfg, bandwidth_mode=4.9)`).

---

## Experiment 2: Synthetic OPL

**Goal:** Learn pricing policies from synthetic offline data; compare **Ai2026**, **Leung2025 (\(\delta_{train}\))**, and **DRO / CDR²O²PL-CP (\(\delta_{train}\))**. Evaluation metric: **\(R_\delta\)** at each\(\delta_{test}\).

| Setting | Value |
|---------|--------|
| Data | Same DGP family as OPE (linear setting) |
| \(\delta_{train}\) | 0.2 |
| \(\delta_{test}\) | 0.1, 0.2, 0.3 |
| Train sizes N | 500, 1000, 1500, 2000, 2500 |
| Test set | 2500 samples (`N_TEST`) |
| Repeats | 20 |
| Policy | Linear (see `POLICY_TYPE` in config) |
| Price bounds | [0.5, 3.0] |

**Config:** `experiments/opl/config_synthetic.py`

**Run:**

```bash
python experiments/opl/run_main_OPL_synthetic.py
```

**Outputs:**

| File | Description |
|------|-------------|
| `Figure2_DROPL_vs_N_Q_DRO_linear.pdf` | Figure 2 style plot (via `src.visualization`) |

**Bandwidth sensitivity:**

```bash
python experiments/opl/run_bandwidth_sensitivity.py
```

Uses `experiments/opl/config_bandwidth_sensitivity.py`.

---

## Experiment 3: Expedia OPL

**Goal:** Policy learning on **real** Expedia hotel search logs; evaluate **\(\hat R_{adv} \)** via **KL-adversarial attack** resampling (not Hat Q_DRO). Compare the same three learners as synthetic OPL.

| Setting | Value |
|---------|--------|
| Data | [Expedia Personalized Sort](https://www.kaggle.com/competitions/expedia-personalized-sort) (`train.csv` → `RealData/data/train.csv`) |
| \(\delta_{train}\) | 0.1 |
| \(\delta_{test}\) | 0.2 |
| Train sizes N | 500 |
| Test / attack | `N_TEST=2500`, `N_TEST_ATTACK=1000`, `M_ATTACKS=10` |
| Split | `hotel_type` via `EXPEDIA_SPLIT_MODE` (prop_starrating 1–3 train, 4–5 test) |
| Reward | `EXPEDIA_REWARD_MODE` (default `booking_bool`) |
| Price bounds | Quantiles Q1–Q99 on training prices |

**Data prerequisite:** Download `train.csv` from the [Kaggle competition](https://www.kaggle.com/competitions/expedia-personalized-sort) and place it at `RealData/data/train.csv` (or set `EXPEDIA_TRAIN_CSV` in config). The file is **not** redistributed with this repository.

If you use the Expedia data, please cite:

```bibtex
@misc{expedia-personalized-sort,
    author = {Adam and Ben Hamner and Dan Friedman and SSA\_Expedia},
    title = {Personalize Expedia Hotel Searches - ICDM 2013},
    year = {2013},
    howpublished = {\url{https://kaggle.com/competitions/expedia-personalized-sort}},
    note = {Kaggle}
}
```

**Config:** `experiments/opl/config_expedia.py`

**Run:**

```bash
python experiments/opl/run_main_OPL_expedia.py
```

**Outputs:**

| File | Description |
|------|-------------|
| `Figure2_DROPL_vs_N_Q_min_linear_hotel_type.pdf` | Figure 2 style plot for \(\hat R_{adv} \) |

---

## Configuration reference

| Experiment | Config file | What to edit |
|------------|-------------|--------------|
| Synthetic OPE | `experiments/ope/config_ope.py` | \(\delta_{train}\) / \(\delta_{test}\), T, `N_TEST_TRUE`, `BANDWIDTH_MODE`, repeats |
| Synthetic OPL | `experiments/opl/config_synthetic.py` | \(\delta_{train}\) / \(\delta_{test}\), N grid, `N_TEST`, training epochs, price bounds |
| Expedia OPL | `experiments/opl/config_expedia.py` | CSV path, split, reward mode, \(\delta\), N, attack counts |
| `src/config.py` | — | **Do not** put experiment grids here; only shared learner/DGP defaults (LR, `DIM`, `ETA`, OPE fold count fallbacks, etc.). OPE/OPL entry points **sync** experiment values into `src.config` before running. |

### Bandwidth modes (`resolve_bandwidth` in `src/estimators.py`)

| `BANDWIDTH_MODE` | Formula |
|------------------|---------|
| `"silverman"` | \(h = 1.06 \cdot \mathrm{std}(P) \cdot T^{-1/5}\) |
| `4.9` | \(h = 1.06 \cdot \mathrm{std}(P) \cdot T^{-1/4.9}\) |
| `2.9` | \(h = 1.06 \cdot \mathrm{std}(P) \cdot T^{-1/2.9}\) |

---

## Regenerating figures from CSV

If statistics CSVs already exist:

```bash
python regenerate_figures.py
```

Expects default paths under `results/ope/` and `results/opl/`; adjust paths inside the script if your files live elsewhere.

---

## Results layout

Example outputs may appear under:

- `results/ope/` — OPE Figure 1, bandwidth sensitivity runs  
- `results/opl/` — OPL Figure 2, bandwidth sensitivity  

Fresh runs from the commands above write CSV/PDF to the **working directory** unless you change paths in the plotting helpers.

---

## Paper and Citation

To cite our work, please use the following citation.
