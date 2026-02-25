# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks, peak_widths, peak_prominences
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _safe_stats(s: pd.Series, prefix: str):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return {
            f"{prefix}_max": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_skewness": np.nan,
            f"{prefix}_kurtosis": np.nan,
        }
    return {
        f"{prefix}_max": float(s.max()),
        f"{prefix}_min": float(s.min()),
        f"{prefix}_mean": float(s.mean()),
        f"{prefix}_std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        f"{prefix}_skewness": float(s.skew()) if len(s) > 2 else 0.0,
        f"{prefix}_kurtosis": float(s.kurt()) if len(s) > 3 else 0.0,
    }

def _linear_slope(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < 2:
        return np.nan
    x0 = x - x.mean()
    denom = np.dot(x0, x0)
    if denom == 0:
        return np.nan
    return float(np.dot(x0, y - y.mean()) / denom)

def _ic_features_from_segment(seg: pd.DataFrame, time_col: str, volt_col: str,curr_col: str, dt_h_col: str,peak_sign: str,prefix: str):
    out = {
        f"{prefix}_IC_area": np.nan,
        f"{prefix}_IC_areaVoltage": np.nan,
        f"{prefix}_IC_max": np.nan,
        f"{prefix}_IC_min": np.nan,
        f"{prefix}_IC_mean": np.nan,
        f"{prefix}_IC_std": np.nan,
        f"{prefix}_IC_skewness": np.nan,
        f"{prefix}_IC_kurtosis": np.nan,
        f"{prefix}_IC_peak": np.nan,
        f"{prefix}_IC_peakWidth": np.nan,
        f"{prefix}_IC_peakLocation": np.nan,
        f"{prefix}_IC_peakProminence": np.nan,
        f"{prefix}_IC_peaksArea": np.nan,
        f"{prefix}_IC_peakLeftSlope": np.nan,
        f"{prefix}_IC_peakRightSlope": np.nan,
        f"{prefix}_IC_peakCount": np.nan,
        f"{prefix}_IC_peakToPeakDistance": np.nan,
    }

    if seg is None or len(seg) < 5:
        return out

    seg = seg.copy().sort_values(time_col)

    I = pd.to_numeric(seg[curr_col], errors="coerce").to_numpy(dtype=float)
    V = pd.to_numeric(seg[volt_col], errors="coerce").to_numpy(dtype=float)
    dt_h = pd.to_numeric(seg[dt_h_col], errors="coerce").to_numpy(dtype=float)

    m = np.isfinite(I) & np.isfinite(V) & np.isfinite(dt_h)
    I = I[m]; V = V[m]; dt_h = dt_h[m]
    if len(I) < 5:
        return out

    dQ = np.abs(I) * dt_h
    Q = np.cumsum(dQ)

    dQ_dV = np.gradient(Q, V)

    s = pd.Series(dQ_dV).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) >= 1:
        y = s.to_numpy(dtype=float)

        out[f"{prefix}_IC_area"] = float(np.trapz(y, dx=1.0))
        out[f"{prefix}_IC_max"] = float(np.nanmax(y))
        out[f"{prefix}_IC_min"] = float(np.nanmin(y))
        out[f"{prefix}_IC_mean"] = float(np.nanmean(y))
        out[f"{prefix}_IC_std"] = float(np.nanstd(y, ddof=1)) if len(y) > 1 else 0.0
        out[f"{prefix}_IC_skewness"] = float(pd.Series(y).skew()) if len(y) > 2 else 0.0
        out[f"{prefix}_IC_kurtosis"] = float(pd.Series(y).kurt()) if len(y) > 3 else 0.0

        xV = V[:len(y)]
        ok = np.isfinite(xV) & np.isfinite(y)
        if ok.sum() >= 2:
            out[f"{prefix}_IC_areaVoltage"] = float(np.trapz(y[ok], x=xV[ok]))

    if not _HAS_SCIPY or len(s) < 5:
        return out

    y = s.to_numpy(dtype=float)
    xV = V[:len(y)]
    y_for_peaks = -y if peak_sign == "negative" else y

    peaks, _ = find_peaks(y_for_peaks)
    out[f"{prefix}_IC_peakCount"] = float(len(peaks))

    if len(peaks) == 0:
        return out

    prominences = peak_prominences(y_for_peaks, peaks)[0]
    k = int(np.argmax(prominences))
    p = int(peaks[k])

    out[f"{prefix}_IC_peak"] = float(y[p])
    out[f"{prefix}_IC_peakLocation"] = float(xV[p]) if np.isfinite(xV[p]) else np.nan
    out[f"{prefix}_IC_peakProminence"] = float(prominences[k])

    widths_res = peak_widths(y_for_peaks, [p], rel_height=0.5)
    out[f"{prefix}_IC_peakWidth"] = float(widths_res[0][0])

    left = int(np.floor(widths_res[2][0]))
    right = int(np.ceil(widths_res[3][0]))
    left = max(left, 0); right = min(right, len(y) - 1)
    if right > left:
        out[f"{prefix}_IC_peaksArea"] = float(np.trapz(y[left:right+1], dx=1.0))

    if p - 1 >= 0:
        out[f"{prefix}_IC_peakLeftSlope"] = float(y[p] - y[p-1])
    if p + 1 < len(y):
        out[f"{prefix}_IC_peakRightSlope"] = float(y[p+1] - y[p])

    if len(peaks) >= 2 and np.isfinite(xV).any():
        order = np.argsort(prominences)[::-1]
        p1 = int(peaks[order[0]])
        p2 = int(peaks[order[1]])
        if np.isfinite(xV[p1]) and np.isfinite(xV[p2]):
            out[f"{prefix}_IC_peakToPeakDistance"] = float(abs(xV[p1] - xV[p2]))

    return out

def build_cycle_features(df: pd.DataFrame,
                         cycle_col="Cycle_Index",
                         time_col="Test_Time(s)",
                         volt_col="Voltage(V)",
                         curr_col="Current(A)",
                         step_col=None,
                         charge_mask=None,
                         discharge_mask=None,
                         step2_value=2,
                         step4_value=4,
                         step7_value=7,
                         cc_tol_frac=0.05,
                         cv_vtol_frac=0.005,
                         eps=1e-12):
    d = df.copy()

    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d[volt_col] = pd.to_numeric(d[volt_col], errors="coerce")
    d[curr_col] = pd.to_numeric(d[curr_col], errors="coerce")
    d = d.sort_values([cycle_col, time_col])

    dt_s = d.groupby(cycle_col)[time_col].diff()
    d["dt_h"] = dt_s.fillna(0) / 3600.0
    d["dt_h"] = d["dt_h"].clip(lower=0)

    dV = d.groupby(cycle_col)[volt_col].diff()
    dT = dt_s
    d["dVdt_V_per_s"] = dV / dT
    d.loc[(dT <= 0) | (~np.isfinite(d["dVdt_V_per_s"])), "dVdt_V_per_s"] = np.nan

    absI = np.abs(d[curr_col].to_numpy(dtype=float))
    d["abs_dVdt_over_absI"] = np.abs(d["dVdt_V_per_s"].to_numpy(dtype=float)) / np.maximum(absI, eps)
    d.loc[~np.isfinite(d["abs_dVdt_over_absI"]), "abs_dVdt_over_absI"] = np.nan

    if charge_mask is None:
        charge_mask = lambda x: x[curr_col] > 0
    if discharge_mask is None:
        discharge_mask = lambda x: x[curr_col] < 0

    d["P_W"] = d[volt_col] * d[curr_col]
    d["dE_Wh"] = d["P_W"] * d["dt_h"]

    d["I_ch_A"] = d[curr_col].clip(lower=0)
    d["I_dis_A"] = (-d[curr_col]).clip(lower=0)
    d["dQ_ch_Ah"] = d["I_ch_A"] * d["dt_h"]
    d["dQ_dis_Ah"] = d["I_dis_A"] * d["dt_h"]

    rows = []

    for cyc, g in d.groupby(cycle_col, sort=True):
        out = {"Cycle_Index": float(cyc)}

        g = g.dropna(subset=[time_col, volt_col, curr_col, "dt_h"])
        if len(g) == 0:
            rows.append(out)
            continue

        if "Internal_Resistance(Ohm)" in g.columns:
            r = pd.to_numeric(g["Internal_Resistance(Ohm)"], errors="coerce").dropna()
            out["Internal_Resistance_max"] = float(r.max()) if len(r) else np.nan
            out["Internal_Resistance_median"] = float(r.median()) if len(r) else np.nan
            out["Internal_Resistance_p90"] = float(r.quantile(0.90)) if len(r) else np.nan
        else:
            out["Internal_Resistance_max"] = np.nan
            out["Internal_Resistance_median"] = np.nan
            out["Internal_Resistance_p90"] = np.nan

        dvdt = pd.to_numeric(g["dVdt_V_per_s"], errors="coerce").dropna()
        out["dVdt_mean"] = float(dvdt.mean()) if len(dvdt) else np.nan
        out["avg_abs_dvdt"] = float(np.mean(np.abs(dvdt))) if len(dvdt) else np.nan

        dyn = pd.to_numeric(g["abs_dVdt_over_absI"], errors="coerce").dropna()
        out["mean_abs_dVdt_over_I"] = float(dyn.mean()) if len(dyn) else np.nan
        out["median_abs_dVdt_over_I"] = float(dyn.median()) if len(dyn) else np.nan

        ch = g[charge_mask(g)].copy()
        dis = g[discharge_mask(g)].copy()

        out["Charge_cumulativeCapacity"] = float(ch["dQ_ch_Ah"].sum()) if len(ch) else np.nan
        out["Charge_cumulativeEnergy"] = float(ch["dE_Wh"].sum()) if len(ch) else np.nan
        out["Charge_duration"] = float(ch["dt_h"].sum()) if len(ch) else np.nan
        out["Charge_startVoltage"] = float(ch[volt_col].iloc[0]) if len(ch) else np.nan
        out["Charge_endVoltage"] = float(ch[volt_col].iloc[-1]) if len(ch) else np.nan
        out["DeltaV_charge"] = (out["Charge_endVoltage"] - out["Charge_startVoltage"]
                                if np.isfinite(out["Charge_endVoltage"]) and np.isfinite(out["Charge_startVoltage"])
                                else np.nan)
        out.update(_safe_stats(ch[volt_col], "Charge_Voltage"))
        out.update(_safe_stats(ch[curr_col], "Charge_Current"))

        out["Discharge_cumulativeCapacity"] = float(dis["dQ_dis_Ah"].sum()) if len(dis) else np.nan
        out["Discharge_cumulativeEnergy"] = float(dis["dE_Wh"].sum()) if len(dis) else np.nan
        out["Discharge_duration"] = float(dis["dt_h"].sum()) if len(dis) else np.nan
        out["Discharge_startVoltage"] = float(dis[volt_col].iloc[0]) if len(dis) else np.nan
        out["Discharge_endVoltage"] = float(dis[volt_col].iloc[-1]) if len(dis) else np.nan
        out["DeltaV_discharge"] = (out["Discharge_startVoltage"] - out["Discharge_endVoltage"]
                                   if np.isfinite(out["Discharge_startVoltage"]) and np.isfinite(out["Discharge_endVoltage"])
                                   else np.nan)
        out.update(_safe_stats(dis[volt_col], "Discharge_Voltage"))
        out.update(_safe_stats(dis[curr_col], "Discharge_Current"))

        if np.isfinite(out["Charge_cumulativeCapacity"]) and out["Charge_cumulativeCapacity"] > 0 and np.isfinite(out["Discharge_cumulativeCapacity"]):
            out["Coulombic_Efficiency"] = float(out["Discharge_cumulativeCapacity"] / out["Charge_cumulativeCapacity"])
        else:
            out["Coulombic_Efficiency"] = np.nan

        if np.isfinite(out["Charge_cumulativeEnergy"]) and out["Charge_cumulativeEnergy"] > 0 and np.isfinite(out["Discharge_cumulativeEnergy"]):
            out["Energy_Efficiency"] = float(abs(out["Discharge_cumulativeEnergy"]) / out["Charge_cumulativeEnergy"])
        else:
            out["Energy_Efficiency"] = np.nan

        if step_col is not None and step_col in g.columns:
            s2 = g[(g[step_col] == step2_value)].copy()
            s2_ch = s2[s2[curr_col] > 0].copy()

            out.update(_ic_features_from_segment(
                s2_ch, time_col, volt_col, curr_col, "dt_h",
                peak_sign="positive",
                prefix="Charge_Step2"
            ))

            if len(s2_ch) >= 5:
                I_med = float(np.nanmedian(s2_ch[curr_col]))
                tol = cc_tol_frac * abs(I_med) if np.isfinite(I_med) and I_med != 0 else np.nan
                s2_cc = s2_ch[np.abs(s2_ch[curr_col] - I_med) <= tol].copy() if np.isfinite(tol) else s2_ch.copy()

                out["Charge_Step2_CC_duration"] = float(s2_cc["dt_h"].sum()) if len(s2_cc) else np.nan
                out["Charge_Step2_CC_currentMedian"] = float(np.nanmedian(s2_cc[curr_col])) if len(s2_cc) else np.nan
                out["Charge_Step2_CC_energy"] = float((s2_cc["dE_Wh"]).sum()) if len(s2_cc) else np.nan

                out["Charge_Step2_CC_currentStd"] = float(pd.Series(s2_cc[curr_col]).dropna().std(ddof=1)) if len(s2_cc) > 1 else (0.0 if len(s2_cc) else np.nan)
                out["Charge_Step2_CC_voltageStd"] = float(pd.Series(s2_cc[volt_col]).dropna().std(ddof=1)) if len(s2_cc) > 1 else (0.0 if len(s2_cc) else np.nan)

                ccI = pd.Series(s2_cc[curr_col]).dropna()
                out["Charge_Step2_CC_skewness"] = float(ccI.skew()) if len(ccI) > 2 else (0.0 if len(ccI) else np.nan)
                out["Charge_Step2_CC_kurtosis"] = float(ccI.kurt()) if len(ccI) > 3 else (0.0 if len(ccI) else np.nan)
            else:
                out["Charge_Step2_CC_duration"] = np.nan
                out["Charge_Step2_CC_currentMedian"] = np.nan
                out["Charge_Step2_CC_energy"] = np.nan
                out["Charge_Step2_CC_currentStd"] = np.nan
                out["Charge_Step2_CC_voltageStd"] = np.nan
                out["Charge_Step2_CC_skewness"] = np.nan
                out["Charge_Step2_CC_kurtosis"] = np.nan

            s4 = g[(g[step_col] == step4_value)].copy()
            s4_ch = s4[s4[curr_col] > 0].copy()

            if len(s4_ch) >= 5:
                V_med = float(np.nanmedian(s4_ch[volt_col]))
                vtol = cv_vtol_frac * abs(V_med) if np.isfinite(V_med) and V_med != 0 else np.nan
                s4_cv = s4_ch[np.abs(s4_ch[volt_col] - V_med) <= vtol].copy() if np.isfinite(vtol) else s4_ch.copy()

                out["Charge_Step4_CV_duration"] = float(s4_cv["dt_h"].sum()) if len(s4_cv) else np.nan
                out["Charge_Step4_CV_voltageMedian"] = float(np.nanmedian(s4_cv[volt_col])) if len(s4_cv) else np.nan

                t_h = (s4_cv[time_col].to_numpy(dtype=float) / 3600.0)
                I = s4_cv[curr_col].to_numpy(dtype=float)
                out["Charge_Step4_CV_slope"] = _linear_slope(t_h, I)

                out["Charge_Step4_CV_energy"] = float((s4_cv["dE_Wh"]).sum()) if len(s4_cv) else np.nan
                cvI = pd.Series(s4_cv[curr_col]).dropna()
                out["Charge_Step4_CV_skewness"] = float(cvI.skew()) if len(cvI) > 2 else (0.0 if len(cvI) else np.nan)
                out["Charge_Step4_CV_kurtosis"] = float(cvI.kurt()) if len(cvI) > 3 else (0.0 if len(cvI) else np.nan)

                out["Charge_Step4_CV_endCurrent"] = float(s4_cv[curr_col].iloc[-1]) if len(s4_cv) else np.nan
                out["Charge_Step4_CV_startCurrent"] = float(s4_cv[curr_col].iloc[0]) if len(s4_cv) else np.nan
                if np.isfinite(out["Charge_Step4_CV_startCurrent"]) and np.isfinite(out["Charge_Step4_CV_endCurrent"]):
                    out["Charge_Step4_CV_currentDrop"] = float(out["Charge_Step4_CV_startCurrent"] - out["Charge_Step4_CV_endCurrent"])
                else:
                    out["Charge_Step4_CV_currentDrop"] = np.nan

                if len(s4_cv) >= 5 and np.isfinite(out["Charge_Step4_CV_startCurrent"]) and out["Charge_Step4_CV_startCurrent"] > 0:
                    I0 = out["Charge_Step4_CV_startCurrent"]
                    target = 0.10 * I0
                    idx = np.where(s4_cv[curr_col].to_numpy(dtype=float) <= target)[0]
                    if len(idx) > 0:
                        t0 = float(s4_cv[time_col].iloc[0]) / 3600.0
                        tt = float(s4_cv[time_col].iloc[int(idx[0])]) / 3600.0
                        out["Charge_Step4_CV_tau10"] = float(max(tt - t0, 0.0))
                    else:
                        out["Charge_Step4_CV_tau10"] = np.nan
                else:
                    out["Charge_Step4_CV_tau10"] = np.nan
            else:
                out["Charge_Step4_CV_duration"] = np.nan
                out["Charge_Step4_CV_voltageMedian"] = np.nan
                out["Charge_Step4_CV_slope"] = np.nan
                out["Charge_Step4_CV_energy"] = np.nan
                out["Charge_Step4_CV_skewness"] = np.nan
                out["Charge_Step4_CV_kurtosis"] = np.nan
                out["Charge_Step4_CV_endCurrent"] = np.nan
                out["Charge_Step4_CV_startCurrent"] = np.nan
                out["Charge_Step4_CV_currentDrop"] = np.nan
                out["Charge_Step4_CV_tau10"] = np.nan

            s7 = g[(g[step_col] == step7_value)].copy()
            s7_dis = s7[s7[curr_col] < 0].copy()

            out.update(_ic_features_from_segment(
                s7_dis, time_col, volt_col, curr_col, "dt_h",
                peak_sign="positive",
                prefix="Discharge_Step7"
            ))

            if len(s7_dis) >= 5:
                I_med = float(np.nanmedian(s7_dis[curr_col]))
                tol = cc_tol_frac * abs(I_med) if np.isfinite(I_med) and I_med != 0 else np.nan
                s7_cc = s7_dis[np.abs(s7_dis[curr_col] - I_med) <= tol].copy() if np.isfinite(tol) else s7_dis.copy()

                out["Discharge_Step7_CC_duration"] = float(s7_cc["dt_h"].sum()) if len(s7_cc) else np.nan
                out["Discharge_Step7_CC_currentMedian"] = float(np.nanmedian(s7_cc[curr_col])) if len(s7_cc) else np.nan

                t_h = (s7_cc[time_col].to_numpy(dtype=float) / 3600.0)
                V = s7_cc[volt_col].to_numpy(dtype=float)
                out["Discharge_Step7_CC_slope"] = _linear_slope(t_h, V)

                out["Discharge_Step7_CC_energy"] = float((s7_cc["dE_Wh"]).sum()) if len(s7_cc) else np.nan

                out["Discharge_Step7_CC_currentStd"] = float(pd.Series(s7_cc[curr_col]).dropna().std(ddof=1)) if len(s7_cc) > 1 else (0.0 if len(s7_cc) else np.nan)
                out["Discharge_Step7_CC_voltageStd"] = float(pd.Series(s7_cc[volt_col]).dropna().std(ddof=1)) if len(s7_cc) > 1 else (0.0 if len(s7_cc) else np.nan)

                ccV = pd.Series(s7_cc[volt_col]).dropna()
                out["Discharge_Step7_CC_skewness"] = float(ccV.skew()) if len(ccV) > 2 else (0.0 if len(ccV) else np.nan)
                out["Discharge_Step7_CC_kurtosis"] = float(ccV.kurt()) if len(ccV) > 3 else (0.0 if len(ccV) else np.nan)

                out["Discharge_Step7_CC_tInv"] = float(1.0 / out["Discharge_Step7_CC_duration"]) if np.isfinite(out["Discharge_Step7_CC_duration"]) and out["Discharge_Step7_CC_duration"] > 0 else np.nan
            else:
                out["Discharge_Step7_CC_duration"] = np.nan
                out["Discharge_Step7_CC_currentMedian"] = np.nan
                out["Discharge_Step7_CC_slope"] = np.nan
                out["Discharge_Step7_CC_energy"] = np.nan
                out["Discharge_Step7_CC_currentStd"] = np.nan
                out["Discharge_Step7_CC_voltageStd"] = np.nan
                out["Discharge_Step7_CC_skewness"] = np.nan
                out["Discharge_Step7_CC_kurtosis"] = np.nan
                out["Discharge_Step7_CC_tInv"] = np.nan

        rows.append(out)

    return pd.DataFrame(rows)