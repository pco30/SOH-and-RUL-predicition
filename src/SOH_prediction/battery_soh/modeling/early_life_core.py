# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

from battery_soh.config import (
    EARLY_TRAIN_FRAC, EARLY_RANDOM_STATE, EARLY_N_SPLITS,
    EARLY_SPIKE_WINDOW, EARLY_SPIKE_THRESH, EARLY_SPIKE_PASSES, EARLY_INTERP_METHOD,
    GROUP_COL, CYCLE_COL
)

FORCED_FEATURES = [
    "Cycle_Index",
    "Internal_Resistance_max",
    "Internal_Resistance_median",
    "Internal_Resistance_p90",
    "dVdt_mean",
    "avg_abs_dvdt",
    "mean_abs_dVdt_over_I",
    "median_abs_dVdt_over_I",
    "Charge_cumulativeCapacity",
    "Charge_cumulativeEnergy",
    "Charge_duration",
    "Charge_startVoltage",
    "Charge_endVoltage",
    "DeltaV_charge",
    "Charge_Voltage_max",
    "Charge_Voltage_min",
    "Charge_Voltage_mean",
    "Charge_Voltage_std",
    "Charge_Voltage_skewness",
    "Charge_Voltage_kurtosis",
    "Charge_Current_max",
    "Charge_Current_min",
    "Charge_Current_mean",
    "Charge_Current_std",
    "Charge_Current_skewness",
    "Charge_Current_kurtosis",
    "Discharge_cumulativeCapacity",
    "Discharge_cumulativeEnergy",
    "Discharge_duration",
    "Discharge_startVoltage",
    "Discharge_endVoltage",
    "DeltaV_discharge",
    "Discharge_Voltage_max",
    "Discharge_Voltage_min",
    "Discharge_Voltage_mean",
    "Discharge_Voltage_std",
    "Discharge_Voltage_skewness",
    "Discharge_Voltage_kurtosis",
    "Discharge_Current_max",
    "Discharge_Current_min",
    "Discharge_Current_mean",
    "Discharge_Current_std",
    "Discharge_Current_skewness",
    "Discharge_Current_kurtosis",
    "Coulombic_Efficiency",
    "Energy_Efficiency",
    "protocol",
]

AGE_FEATURES = ["cycle_frac"]
PROXY_FEATURES = ["deg_rate_proxy"]

def add_cycle_frac_if_missing(df):
    d = df.copy()
    if "cycle_frac" not in d.columns:
        d["cycle_frac"] = d.groupby(GROUP_COL)[CYCLE_COL].transform(
            lambda s: (s - s.min()) / max((s.max() - s.min()), 1)
        )
    return d

def add_degradation_proxy(df):
    d = df.copy()
    dis_cap = pd.to_numeric(d.get("Discharge_cumulativeCapacity"), errors="coerce")
    dis_dur = pd.to_numeric(d.get("Discharge_duration"), errors="coerce")
    d["deg_rate_proxy"] = dis_cap / np.maximum(dis_dur, 1e-9)
    if d["deg_rate_proxy"].notna().sum() > 10:
        lo, hi = d["deg_rate_proxy"].quantile([0.01, 0.99])
        d["deg_rate_proxy"] = d["deg_rate_proxy"].clip(lo, hi)
    return d

def clean_soh_series(y):
    y = pd.Series(y).astype(float)
    for _ in range(EARLY_SPIKE_PASSES):
        med = y.rolling(EARLY_SPIKE_WINDOW, center=True, min_periods=3).median()
        y = y.mask((y - med).abs() > EARLY_SPIKE_THRESH)
    return y.interpolate(method=EARLY_INTERP_METHOD, limit_direction="both").to_numpy()

def get_early_mask(df):
    rank = df.groupby(GROUP_COL)[CYCLE_COL].rank(method="first")
    n = df.groupby(GROUP_COL)[CYCLE_COL].transform("count")
    cutoff = np.maximum(1, np.ceil(EARLY_TRAIN_FRAC * n)).astype(int)
    return rank <= cutoff

def stacked_predict_with_fold_uncertainty(X_train, y_train, X_test, groups_train):
    pre = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

    X_train = pre.fit_transform(X_train)
    X_test = pre.transform(X_test)

    base_models = [
        XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, min_child_weight=2.0,
            random_state=101, n_jobs=-1
        ),
        LGBMRegressor(
            n_estimators=1200, learning_rate=0.02, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=101, n_jobs=-1
        ),
        RandomForestRegressor(
            n_estimators=600, max_depth=14, min_samples_leaf=2,
            random_state=101, n_jobs=-1
        )
    ]

    cv = GroupKFold(n_splits=EARLY_N_SPLITS)
    preds = np.zeros((EARLY_N_SPLITS, X_test.shape[0]))

    for i, (tr, _) in enumerate(cv.split(X_train, y_train, groups_train)):
        meta_tr, meta_te = [], []
        for m in base_models:
            m.fit(X_train[tr], y_train[tr])
            meta_tr.append(m.predict(X_train[tr]).reshape(-1, 1))
            meta_te.append(m.predict(X_test).reshape(-1, 1))

        meta = Ridge(alpha=5.0, random_state=EARLY_RANDOM_STATE)
        meta.fit(np.hstack(meta_tr), y_train[tr])
        preds[i] = meta.predict(np.hstack(meta_te))

    return preds.mean(axis=0), preds.std(axis=0, ddof=1)

def compute_battery_panel(cc_df, battery):
    is_target = (cc_df[GROUP_COL] == battery)
    is_early_all = get_early_mask(cc_df)

    train_df = cc_df[is_early_all & (~is_target)].copy()
    test_df = cc_df[is_target].copy()

    if len(train_df) < 20 or len(test_df) < 10:
        return None

    test_rank = test_df[CYCLE_COL].rank(method="first")
    test_n = len(test_df)
    test_cut_n = int(max(1, np.ceil(EARLY_TRAIN_FRAC * test_n)))

    cutoff_cycle_value = test_df.loc[test_rank == test_cut_n, CYCLE_COL].iloc[0] if test_cut_n <= test_n else test_df[CYCLE_COL].max()

    test_post = test_df.loc[test_rank > test_cut_n].copy()
    if len(test_post) < 5:
        return None

    features = [c for c in (FORCED_FEATURES + AGE_FEATURES + PROXY_FEATURES) if c in cc_df.columns]
    X_train = train_df[features]
    y_train = train_df["SoH_clean"]
    X_test = test_post[features]
    y_test = test_post["SoH_clean"]

    y_pred, y_std = stacked_predict_with_fold_uncertainty(
        X_train.values, y_train.values, X_test.values, train_df[GROUP_COL].values
    )
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    plot_df = test_df[[CYCLE_COL, "SoH_clean"]].copy()
    plot_df["pred"] = np.nan
    plot_df["std"] = np.nan
    plot_df.loc[test_post.index, "pred"] = y_pred
    plot_df.loc[test_post.index, "std"] = y_std

    return plot_df, cutoff_cycle_value, mae, rmse