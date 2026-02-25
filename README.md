# SOH and RUL Prediction Using Physics-Informed Feature Engineering and Stacked Ensembles

This repository implements an end-to-end machine learning pipeline for **State-of-Health (SoH)** prediction and **early-life Remaining Useful Life (RUL) forecasting** of lithium-ion batteries using high-resolution cycling data.

The framework combines:

* Physics-motivated feature engineering
* Robust per-cycle aggregation
* Stability-based feature selection
* Battery-aware cross-validation
* Regularized gradient boosting and stacked ensemble models

The goal is to produce **accurate, generalizable, and interpretable** health predictions that transfer across cells and cycling protocols.

---

## Key Capabilities

* Handles **constant-discharge** and **variable-discharge** cycling protocols
* Computes SoH via **current integration**
* Engineers **100+ per-cycle degradation features**, including:

  * Internal resistance statistics
  * dV/dt dynamics and resistance proxies
  * Charge/discharge voltage and current distributions
  * Coulombic and energy efficiency
  * CC and CV stability metrics
  * Incremental Capacity (IC) curve shape and peak features
* Performs **correlation pruning + permutation-importance stability selection**
* Uses **battery-holdout evaluation** (true generalization test)
* Trains:

  * Single well-regularized HistGradientBoosting model
  * Stacked ensemble (XGBoost + LightGBM + RandomForest → Ridge meta-learner)
* Generates **early-life forecasting plots** with uncertainty bands

---

## Methodology Overview

1. Load raw cycling data from Excel logs
2. Compute per-cycle SoH using current integration
3. Aggregate time-series measurements into cycle-level features
4. Filter low-quality cycles and extreme SoH spikes
5. Select stable features using grouped permutation importance
6. Train models using GroupKFold (battery-aware) cross-validation
7. Evaluate on unseen batteries
8. Forecast future SoH from early-life cycles

---

## Main Project Structure

```
SOH-and-RUL-predicition/
├─ README.md
└─ src/
   └─ SOH_prediction/
      ├─ battery_soh/
      │  ├─ config.py
      │  ├─ io/
      │  │  └─ io_utils.py
      │  ├─ targets/
      │  │  └─ soh.py
      │  ├─ fe/
      │  │  ├─ features.py
      │  │  └─ selection.py
      │  └─ modeling/
      │     ├─ models.py
      │     └─ early_life_core.py
      └─ scripts/
         ├─ main.py
         └─ early_life.py
```
ignore src/prev_work
---

## ⚙️ Installation

Create a virtual environment (recommended):

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS/Linux
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

---

#  Resulting workflow for anyone

```bash
git clone <repo>
cd SOH-and-RUL-predicition
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cd src\SOH_prediction
python -m scripts.main

Output:

```
HOLDOUT (by battery) -> MSE: <value>  MAE: <value>
HOLDOUT (single HGB) -> MSE: <value>  MAE: <value>
```

---

### Early-life forecasting + plots

```bash
python -m scripts.early_life
```

Produces SoH vs cycle plots for each battery, including:

* True cleaned SoH
* Predicted SoH
* ±1 standard-deviation uncertainty band
* Training cutoff (first 30% of life)

---

##  Features

Examples of feature categories:

* `Internal_Resistance_max`, `median`, `p90`
* `dVdt_mean`, `avg_abs_dvdt`
* `mean_abs_dVdt_over_I`
* `Charge_cumulativeCapacity`, `Discharge_duration`
* `Coulombic_Efficiency`, `Energy_Efficiency`
* `Charge_Step4_CV_tau10`
* `Discharge_Step7_IC_peakLocation`
* `cycle_frac` (normalized aging proxy)

Features are selected automatically via stability selection.

---

## 🔬 Modeling

### Base Models

* XGBoost
* LightGBM
* Random Forest

### Meta Learner

* Ridge Regression

### Baseline

* HistGradientBoostingRegressor (strong regularization)

---

## 🧪 Validation Strategy

* **GroupKFold by battery ID**
* Prevents leakage across cells
* Feature selection performed **only on training batteries**

---

## 📈 Early-Life Forecasting

For a target battery:

* Train on early cycles of all other batteries
* Predict future cycles of the target battery
* Enables RUL-style forecasting before significant degradation occurs

---

## 🚀 Applications

* Battery health monitoring
* Predictive maintenance
* Lifetime benchmarking
* Degradation mechanism analysis
* Data-driven battery management systems

---

## 📜 License

MIT License 

---

## ✉️ Contact

If you have questions, suggestions, or would like to collaborate:

**Princely O.**
