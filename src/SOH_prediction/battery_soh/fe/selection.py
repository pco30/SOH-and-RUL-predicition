# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingRegressor

def starter_shortlist(include_cycle_index=False):
    feats = [
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
        "Discharge_cumulativeCapacity",
        "Discharge_cumulativeEnergy",
        "Discharge_duration",
        "Discharge_startVoltage",
        "Discharge_endVoltage",
        "DeltaV_discharge",
        "Coulombic_Efficiency",
        "Energy_Efficiency",
        "Charge_Voltage_mean",
        "Charge_Voltage_std",
        "Charge_Current_mean",
        "Charge_Current_std",
        "Discharge_Voltage_mean",
        "Discharge_Voltage_std",
        "Discharge_Current_mean",
        "Discharge_Current_std",
        "Charge_Step2_IC_areaVoltage",
        "Charge_Step2_IC_peakLocation",
        "Charge_Step2_IC_peakProminence",
        "Charge_Step2_IC_peakWidth",
        "Charge_Step2_IC_peakCount",
        "Charge_Step2_IC_peaksArea",
        "Charge_Step2_IC_peakToPeakDistance",
        "Charge_Step2_IC_peakLeftSlope",
        "Charge_Step2_IC_peakRightSlope",
        "Charge_Step2_CC_duration",
        "Charge_Step2_CC_currentMedian",
        "Charge_Step2_CC_energy",
        "Charge_Step2_CC_currentStd",
        "Charge_Step2_CC_voltageStd",
        "Charge_Step4_CV_duration",
        "Charge_Step4_CV_voltageMedian",
        "Charge_Step4_CV_slope",
        "Charge_Step4_CV_endCurrent",
        "Charge_Step4_CV_currentDrop",
        "Charge_Step4_CV_tau10",
        "Charge_Step4_CV_energy",
        "Discharge_Step7_IC_areaVoltage",
        "Discharge_Step7_IC_peakLocation",
        "Discharge_Step7_IC_peakProminence",
        "Discharge_Step7_IC_peakWidth",
        "Discharge_Step7_IC_peakCount",
        "Discharge_Step7_IC_peaksArea",
        "Discharge_Step7_IC_peakToPeakDistance",
        "Discharge_Step7_IC_peakLeftSlope",
        "Discharge_Step7_IC_peakRightSlope",
        "Discharge_Step7_CC_slope",
        "Discharge_Step7_CC_tInv",
        "Discharge_Step7_CC_duration",
        "Discharge_Step7_CC_energy",
        "Discharge_Step7_CC_currentStd",
        "Discharge_Step7_CC_voltageStd",
        "protocol",
        "cycle_frac",
    ]

    if include_cycle_index:
        feats = ["Cycle_Index"] + feats
    return feats

def correlation_prune(X, threshold=0.95, method="spearman"):
    if X.shape[1] <= 1:
        return list(X.columns)

    corr = X.corr(method=method).abs()

    keep = []
    for col in X.columns:
        if not keep:
            keep.append(col)
            continue

        if (corr.loc[col, keep] > threshold).any():
            continue

        keep.append(col)

    return keep

def stable_permutation_selection(X, y, groups=None, n_splits=5, top_n_each_fold=15, appear_frac_thresh=0.7, corr_threshold=0.95, corr_method="spearman", n_repeats=10, random_state=42, model=None):
    pruned_cols = correlation_prune(X, threshold=corr_threshold, method=corr_method)
    Xp = X[pruned_cols]

    if groups is not None:
        cv = GroupKFold(n_splits=n_splits)
        split_iter = cv.split(Xp, y, groups=groups)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = cv.split(Xp, y)

    if model is None:
        model = HistGradientBoostingRegressor(random_state=random_state)

    cols = np.array(Xp.columns)
    appear_counts = pd.Series(0, index=cols, dtype=int)
    mean_importance = pd.Series(0.0, index=cols, dtype=float)
    n_folds = 0

    for tr_idx, va_idx in split_iter:
        n_folds += 1

        X_tr, X_va = Xp.iloc[tr_idx], Xp.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model.fit(X_tr, y_tr)

        perm = permutation_importance(model, X_va, y_va, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
        imp = pd.Series(perm.importances_mean, index=cols)

        mean_importance += imp

        top_cols = imp.sort_values(ascending=False).head(min(top_n_each_fold, len(cols))).index
        appear_counts.loc[top_cols] += 1

    mean_importance /= max(n_folds, 1)
    appear_frac = appear_counts / max(n_folds, 1)

    ranking_df = pd.DataFrame(
        {"appear_frac": appear_frac, "appear_count": appear_counts, "mean_perm_importance": mean_importance}
    ).sort_values(["appear_frac", "mean_perm_importance"], ascending=False)

    selected = appear_frac[appear_frac >= appear_frac_thresh].index.tolist()

    if len(selected) == 0:
        selected = ranking_df["mean_perm_importance"].sort_values(ascending=False).head(min(10, len(cols))).index.tolist()

    return selected, ranking_df

def select_best_feature_df(combined, target_col="Calculated_SOH(%)", group_col="battery_id", include_cycle_index=False, dropna_target=True, corr_threshold=0.95, n_splits=5, top_n_each_fold=15, appear_frac_thresh=0.7, n_repeats=10, random_state=42, keep_target=False, keep_group=False, model=None):
    df = combined.copy()
    if dropna_target:
        df = df.loc[df[target_col].notna()].copy()

    shortlist = starter_shortlist(include_cycle_index=include_cycle_index)
    feat_cols = [c for c in shortlist if c in df.columns]

    X = df[feat_cols]
    y = df[target_col].astype(float)

    groups = None
    if group_col is not None and group_col in df.columns:
        groups = df[group_col]

    selected_features, _ranking = stable_permutation_selection(
        X, y, groups=groups,
        n_splits=n_splits, top_n_each_fold=top_n_each_fold,
        appear_frac_thresh=appear_frac_thresh,
        corr_threshold=corr_threshold,
        n_repeats=n_repeats,
        random_state=random_state,
        model=model
    )

    cols_to_return = []

    if keep_group and group_col is not None and group_col in df.columns:
        cols_to_return.append(group_col)

    if keep_target and target_col in df.columns:
        cols_to_return.append(target_col)

    cols_to_return += selected_features
    return df[cols_to_return].copy()