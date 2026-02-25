# -*- coding: utf-8 -*-
import pandas as pd

def calculate_soh_variable_discharge(df):
    df['Test_Time(h)'] = df['Test_Time(s)'] / 3600
    df = df.sort_values(['Cycle_Index', 'Test_Time(h)']).reset_index(drop=True)
    df['delta_t(h)'] = df['Test_Time(h)'].diff()
    df.loc[df['Cycle_Index'] != df['Cycle_Index'].shift(), 'delta_t(h)'] = 0
    df['Discharge_Capacity(mAh)'] = 0.0
    discharge_mask = df['Current(A)'] < 0
    df.loc[discharge_mask, 'Discharge_Capacity(mAh)'] = (
        abs(df.loc[discharge_mask, 'Current(A)']) * df.loc[discharge_mask, 'delta_t(h)'] * 1000
    )
    df['Battery_Capacity(mAh)'] = df.groupby('Cycle_Index')['Discharge_Capacity(mAh)'].transform('sum')
    df['Battery_Capacity(mAh)'] = df['Battery_Capacity(mAh)']/5
    max_capacity = df['Battery_Capacity(mAh)'].max()
    df['Calculated_SOH(%)'] = (df['Battery_Capacity(mAh)'] / max_capacity) * 100
    df['Time_charged(h)'] = 0.0
    df['Time_discharged(h)'] = 0.0
    df.loc[df['Current(A)'] >= 0, 'Time_charged(h)'] = df['delta_t(h)']
    df.loc[df['Current(A)'] < 0,  'Time_discharged(h)'] = df['delta_t(h)']
    df['Test_Time_charged(h)'] = df.groupby('Cycle_Index')['Time_charged(h)'].transform('sum')
    df['Test_Time_Discharged(h)'] = df.groupby('Cycle_Index')['Time_discharged(h)'].transform('sum')
    return df

def soh_from_current_integration(df, cycle_col="Cycle_Index",time_col="Test_Time(s)", current_col="Current(A)",rated_capacity_Ah=None,reference="first"):
    d = df.copy().sort_values([cycle_col, time_col])
    d["dt_h"] = d.groupby(cycle_col)[time_col].diff().fillna(0) / 3600.0
    d["dt_h"] = d["dt_h"].clip(lower=0)
    d["I_dis_A"] = (-d[current_col]).clip(lower=0)
    d["dQ_Ah"] = d["I_dis_A"] * d["dt_h"]
    d["Qdis_Ah_cum"] = d.groupby(cycle_col)["dQ_Ah"].cumsum()
    cap_by_cycle = d.groupby(cycle_col)["Qdis_Ah_cum"].max()

    if rated_capacity_Ah is not None:
        ref = float(rated_capacity_Ah)
    elif reference == "first":
        ref = float(cap_by_cycle.iloc[0])
    else:
        ref = float(cap_by_cycle.max())

    d["Battery_Capacity(Ah)"] = d[cycle_col].map(cap_by_cycle)
    d["Discharge_Capacity_Cycle(Ah)"] = d["Battery_Capacity(Ah)"]
    d["Calculated_SOH(%)"] = d["Battery_Capacity(Ah)"] / ref * 100.0
    return d