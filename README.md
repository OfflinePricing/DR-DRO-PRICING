# Distributionally Robust Offline Pricing: OPE and OPL

Implementation of **Offline Policy Evaluation (OPE)** and **Offline Policy Learning (OPL)** for continuous pricing, with synthetic simulations and Expedia real-data experiments.

## Algorithms

| Name | Role | Reference |
|------|------|-----------|
| **LDR²PE-CP** | OPE: localized doubly robust DR policy evaluation | This repo (Algorithm 1) |
| **DRO-IPW** | OPE/L: DRO-IPW baseline | Leung et al. (2025) |
| **CDR²O²PL-CP (DRO)** | OPL: continuum doubly robust DRO policy learning | This repo (Algorithm 2) |
| **Ai2024** | OPL non-robust baseline (double debiasing) | Ai et al. (2026) |

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
- **\(\delta_{test}\)** — used when **evaluating** the distributionally robust objective.

This matches the paper design: train at a fixed ambiguity level, report performance at several test radii.

**Ai et al. (2026) (OPL only):** training is **non-robust**. For comparability on synthetic data, \(R_\delta\) at evaluation still uses `get_optimal_alpha(..., delta_test)` with the same \(\delta_{test}\) as DRO methods.

---

## Experiment 1: Synthetic OPE

**Goal:** Estimate the robust value \(R_\delta\) of a **fixed** target policy on synthetic offline data; compare **LDR²PE-CP** vs **DRO-IPW** by MSE against a Monte Carlo ground truth.

**Config:** `experiments/ope/config_ope.py`  

**Run:**

```bash
python experiments/ope/run_main_OPE.py
```

**Outputs:**

File: `OPE_bandwidth.pdf` ( MSE vs T (Figure 1) )


**Bandwidth sensitivity** (Silverman \(n^{-1/5.0}\) vs \(n^{-1/4.9}\) vs \(n^{-1/2.9}\) rules):

```bash
python experiments/ope/run_bandwidth_sensitivity.py
```

---

## Experiment 2: Synthetic OPL

**Goal:** Learn pricing policies from synthetic offline data; compare **Ai2026**, **Leung2025 (\(\delta_{train}\))**, and **DRO / CDR²O²PL-CP (\(\delta_{train}\))**. Evaluation metric: **\(R_\delta\)** at each\(\delta_{test}\).

**Config:** `experiments/opl/config_synthetic.py`

**Run:**

```bash
python experiments/opl/run_main_OPL_synthetic.py
```

**Outputs:**

File: `OPL_bandwidth.pdf` (Figure 2 style plot (via `src.visualization`) 

**Bandwidth sensitivity:**

```bash
python experiments/opl/run_bandwidth_sensitivity.py
```

Uses `experiments/opl/config_bandwidth_sensitivity.py`.

---

## Experiment 3: Expedia OPL

**Goal:** Policy learning on **real** Expedia hotel search logs; evaluate **\(\hat R_{adv} \)** via **KL-adversarial attack** resampling (not Hat Q_DRO). Compare the same three learners as synthetic OPL.

**Data：** [Expedia Personalized Sort](https://www.kaggle.com/competitions/expedia-personalized-sort) (`train.csv` → `RealData/data/train.csv`) |

**Split：** `hotel_type` (prop_starrating 1–3 train, 4–5 test) |

**Reward：** `EXPEDIA_REWARD_MODE` (default `booking_bool`) |

**Price bounds：** Quantiles Q1–Q99 on training prices |

**Data prerequisite:** Download `train.csv` from the [Kaggle competition](https://www.kaggle.com/competitions/expedia-personalized-sort) and place it at `RealData/data/train.csv`. The file is **not** redistributed with this repository.

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

File：`Expedia_Statistics.csv` 



---

## Paper and Citation

To cite our work, please use the following citation.
