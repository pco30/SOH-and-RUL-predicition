# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from battery_soh.config import (
    FOLDER_PATHS_CONST, SHEET_NAMES_CONST,
    FOLDER_PATHS_VAR, SHEET_NAMES_VAR,
    TARGET_COL, GROUP_COL, CYCLE_COL, PROTOCOL_COL,
    EOL_CUTOFF_SOH, CHARGE_DURATION_MIN_H,
    TEST_BATTERIES_HOLDOUT, SPIKE_DIFF_THRESH
)

from battery_soh.io.io_utils import process_files, battery_name_from_path
from battery_soh.targets.soh import soh_from_current_integration, calculate_soh_variable_discharge
from battery_soh.fe.features import build_cycle_features
from battery_soh.fe.selection import select_best_feature_df
from battery_soh.modeling.models import fit_predict_stacked_ensemble, fit_predict_single_regularized_model

datasets = []
all_features = []

for i in range(len(FOLDER_PATHS_CONST)):
    battery_id = battery_name_from_path(FOLDER_PATHS_CONST[i])
    df = process_files(FOLDER_PATHS_CONST[i], SHEET_NAMES_CONST[i])
    df[GROUP_COL] = battery_id

    df_SOH = soh_from_current_integration(df)
    df_SOH = df_SOH[[CYCLE_COL, TARGET_COL]]
    df_SOH = df_SOH.groupby(CYCLE_COL, as_index=False).agg({TARGET_COL: "max"})

    features = build_cycle_features(
        df,
        cycle_col=CYCLE_COL, time_col="Test_Time(s)", volt_col="Voltage(V)", curr_col="Current(A)",
        step_col="Step_Index"
    )
    features[GROUP_COL] = battery_id
    features[PROTOCOL_COL] = 1
    features = features.merge(df_SOH, on=CYCLE_COL, how="left")
    features = features[features["Charge_duration"] >= CHARGE_DURATION_MIN_H]

    datasets.append(df)
    all_features.append(features)

for i in range(len(FOLDER_PATHS_VAR)):
    battery_id = battery_name_from_path(FOLDER_PATHS_VAR[i])
    df2 = process_files(FOLDER_PATHS_VAR[i], SHEET_NAMES_VAR[i])
    df2[GROUP_COL] = battery_id

    df_SOH = calculate_soh_variable_discharge(df2)
    df_SOH = df_SOH[[CYCLE_COL, TARGET_COL]]
    df_SOH = df_SOH.groupby(CYCLE_COL, as_index=False).agg({TARGET_COL: "max"})

    features = build_cycle_features(
        df2,
        cycle_col=CYCLE_COL, time_col="Test_Time(s)", volt_col="Voltage(V)", curr_col="Current(A)",
        step_col="Step_Index"
    )
    features[GROUP_COL] = battery_id
    features[PROTOCOL_COL] = 0
    features = features.merge(df_SOH, on=CYCLE_COL, how="left")

    soh = features[TARGET_COL]
    prev_diff = (soh - soh.shift(1)).abs()
    next_diff = (soh - soh.shift(-1)).abs()
    mask = (prev_diff > SPIKE_DIFF_THRESH) & (next_diff > SPIKE_DIFF_THRESH)
    bad_cycles = features.loc[mask, CYCLE_COL].unique()

    df2 = df2[~df2[CYCLE_COL].isin(bad_cycles)].reset_index(drop=True)
    features = features[~features[CYCLE_COL].isin(bad_cycles)].reset_index(drop=True)

    datasets.append(df2)
    all_features.append(features)

combined = pd.concat(all_features, ignore_index=True)

combined = combined[combined[TARGET_COL] >= EOL_CUTOFF_SOH].reset_index(drop=True)

combined["cycle_frac"] = combined.groupby(GROUP_COL)[CYCLE_COL].transform(
    lambda s: (s - s.min()) / max((s.max() - s.min()), 1)
).astype(float)

test_batteries_holdout = TEST_BATTERIES_HOLDOUT

is_test_holdout = combined[GROUP_COL].isin(test_batteries_holdout).to_numpy()
train_df_holdout = combined.loc[~is_test_holdout].reset_index(drop=True)
test_df_holdout  = combined.loc[ is_test_holdout].reset_index(drop=True)

best_X_train_df = select_best_feature_df(
    train_df_holdout,
    target_col=TARGET_COL,
    group_col=GROUP_COL,
    include_cycle_index=True,
    keep_target=False,
    keep_group=False
)

selected_cols = best_X_train_df.columns.tolist()
feature_names = selected_cols

y_train_holdout = train_df_holdout[TARGET_COL].astype(float).to_numpy()
groups_train_holdout = train_df_holdout[GROUP_COL].to_numpy()
X_train_holdout = best_X_train_df.to_numpy()

missing_in_test = [c for c in selected_cols if c not in test_df_holdout.columns]
for c in missing_in_test:
    test_df_holdout[c] = np.nan

X_test_holdout = test_df_holdout[selected_cols].to_numpy()
y_test_holdout = test_df_holdout[TARGET_COL].astype(float).to_numpy()

y_pred_holdout, mse_holdout, mae_holdout = fit_predict_stacked_ensemble(
    X_train_holdout, y_train_holdout, X_test_holdout, y_test_holdout,
    groups_train=groups_train_holdout, n_splits=5, random_state=42
)
print("HOLDOUT (by battery) -> MSE:", round(mse_holdout, 6), " MAE:", round(mae_holdout, 6))

y_pred_holdout_single, mse_holdout_single, mae_holdout_single = fit_predict_single_regularized_model(
    X_train_holdout, y_train_holdout, X_test_holdout, y_test_holdout, random_state=42
)
print("HOLDOUT (single HGB) -> MSE:", round(mse_holdout_single, 6), " MAE:", round(mae_holdout_single, 6))