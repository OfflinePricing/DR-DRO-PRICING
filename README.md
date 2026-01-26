# Distributionally Robust Offline Pricing: OPE and OPL

This repository contains the implementation for Distributionally Robust Offline Policy Evaluation (OPE) and Offline Policy Learning (OPL) for continuous pricing problems.

## Overview

This project implements:
- **Offline Policy Evaluation (OPE)**: Evaluating a given target policy using distributionally robust methods
- **Offline Policy Learning (OPL)**: Learning robust pricing policies from offline data

The code implements several algorithms:
- **LDR²PE-CP** (Localized Doubly Robust Distributionally Robust Policy Evaluation for Continuous Pricing)
- **CDR²O²PL-CP** (Continuum Doubly Robust DRO OPL for Continuous Pricing)
- **DRO-IPW** (Distributionally Robust Inverse Propensity Weighting from Leung et al. 2025)
- **DR** (Double Robust method from Ai et al. 2024)

## Installation

### Requirements

- Python 3.7+
- See `requirements.txt` for detailed dependencies

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Code_Delta_Github
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
Code_Delta_Github/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── src/                         # Shared source code
│   ├── __init__.py
│   ├── config.py                # Configuration parameters
│   ├── dgp.py                   # Data generating process
│   ├── policy.py                # Policy definitions (Linear, MLP)
│   ├── estimators.py            # Estimators (Propensity, Outcome)
│   ├── learner.py               # Learning algorithms
│   └── visualization.py         # Visualization utilities
├── experiments/                 # Experiment scripts
│   ├── ope/                    # OPE experiments
│   │   ├── run_main_OPE.py     # Main OPE experiment script
│   │   └── theoretical_ope.py  # Theoretical OPE computation
│   └── opl/                    # OPL experiments
│       └── run_main_OPL.py     # Main OPL experiment script
└── results/                    # Experiment results
    ├── ope/                    # OPE results (figures, CSV files)
    └── opl/                    # OPL results (figures, CSV files)
```

## Usage

### Running OPE Experiments

To run Offline Policy Evaluation experiments:

```bash
cd experiments/ope
python run_main_OPE.py
```

This will:
- Evaluate target policies using LDR²PE-CP and DRO-IPW methods
- Generate results saved to `results/ope/`


### Running OPL Experiments

To run Offline Policy Learning experiments:

```bash
cd experiments/opl
python run_main_OPL.py
```

This will:
- Train robust pricing policies using CDR²O²PL-CP, DRO-IPW, and DR methods
- Generate results saved to `results/opl/`


### Configuration

Edit `src/config.py` to modify:
- Algorithm hyperparameters (learning rates, epochs, etc.)
- Experiment parameters (number of repeats, test set sizes, etc.)
- Environment settings (dimensions, price ranges, etc.)


## Results

Results from experiments are saved in the `results/` directory:
- **OPE results**: `results/ope/` contains MSE comparisons and statistics
- **OPL results**: `results/opl/` contains distributionally robust value comparisons


