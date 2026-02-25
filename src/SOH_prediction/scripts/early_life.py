# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from battery_soh.config import (
    FOLDER_PATHS_CONST, SHEET_NAMES_CONST,
    FOLDER_PATHS_VAR, SHEET_NAMES_VAR,
    TARGET_COL, GROUP_COL, CYCLE_COL, PROTOCOL_COL,
    EOL_CUTOFF_SOH, CHARGE_DURATION_MIN_H,
    SPIKE_DIFF_THRESH,
    EARLY_LIFE_BATTERIES, EARLY_UNC_ALPHA, EARLY_Y_LIM, EARLY_ALIGN_X_AXES
)

from battery_soh.io.io_utils import process_files, battery_name_from_path
from battery_soh.targets.soh import soh_from_current_integration, calculate_soh_variable_discharge
from battery_soh.fe.features import build_cycle_features
from battery_soh.modeling.early_life_core import (
    add_cycle_frac_if_missing, add_degradation_proxy, clean_soh_series, compute_battery_panel
)

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

cc = combined.copy()
cc = cc.loc[cc[PROTOCOL_COL] == 1].copy()
cc = add_cycle_frac_if_missing(cc)
cc = add_degradation_proxy(cc)

cc["SoH_raw"] = pd.to_numeric(cc[TARGET_COL], errors="coerce")
cc["SoH_clean"] = cc.groupby(GROUP_COL)["SoH_raw"].transform(clean_soh_series)

X_MAX = cc[CYCLE_COL].max() if EARLY_ALIGN_X_AXES else None

for b in EARLY_LIFE_BATTERIES:
    out = compute_battery_panel(cc, b)
    if out is None:
        continue

    plot_df, cutoff, mae, rmse = out

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(plot_df[CYCLE_COL], plot_df["SoH_clean"], label="True SoH (clean)", linewidth=2)
    plt.plot(plot_df[CYCLE_COL], plot_df["pred"], "--", label="Predicted SoH", linewidth=2)

    m = plot_df["pred"].notna()
    plt.fill_between(
        plot_df.loc[m, CYCLE_COL],
        plot_df.loc[m, "pred"] - plot_df.loc[m, "std"],
        plot_df.loc[m, "pred"] + plot_df.loc[m, "std"],
        alpha=EARLY_UNC_ALPHA,
        label="±1 SD (CV folds)"
    )

    plt.axvline(cutoff, linestyle=":", linewidth=1.5, label="Training cutoff (30%)")
    plt.axvspan(plot_df[CYCLE_COL].min(), cutoff, alpha=0.06)

    plt.title(f"{b} | MAE={mae:.2f}, RMSE={rmse:.2f}")
    plt.xlabel("Cycle")
    plt.ylabel("SoH (%)")
    plt.ylim(*EARLY_Y_LIM)
    if X_MAX is not None:
        plt.xlim(0, X_MAX)

    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()