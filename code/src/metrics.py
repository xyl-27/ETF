import json
import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return super().default(obj)


def extract_drawdowns(vals, dates, max_periods=5):
    running_max = -np.inf
    peak_idx = 0
    in_dd = False
    dd_start = None
    dd_start_idx = None
    trough_date = None
    trough_idx = None
    trough_depth = 0
    periods = []

    for i, v in enumerate(vals):
        if v > running_max:
            running_max = v
            if in_dd:
                periods.append({
                    "start": str(dd_start),
                    "trough": str(trough_date),
                    "recovery": str(dates[i]),
                    "depth_pct": round(float(trough_depth), 2),
                    "duration_days": int(i - dd_start_idx),
                    "recovery_days": int(i - trough_idx),
                })
                in_dd = False
            peak_idx = i
        else:
            dd_pct = (running_max - v) / running_max * 100
            if not in_dd:
                in_dd = True
                dd_start = dates[peak_idx]
                dd_start_idx = peak_idx
                trough_date = dates[i]
                trough_idx = i
                trough_depth = dd_pct
            else:
                if dd_pct > trough_depth:
                    trough_date = dates[i]
                    trough_idx = i
                    trough_depth = dd_pct

    if in_dd:
        periods.append({
            "start": str(dd_start),
            "trough": str(trough_date),
            "recovery": None,
            "depth_pct": round(float(trough_depth), 2),
            "duration_days": int(len(vals) - dd_start_idx),
            "recovery_days": None,
        })

    periods.sort(key=lambda x: x["depth_pct"], reverse=True)
    return periods[:max_periods]


def compute_longterm_risk_metrics(daily_rets, cum, dates, dd_periods):
    if len(daily_rets) < 5:
        return {}
    daily_rets_arr = np.array(daily_rets)
    n = len(daily_rets_arr)

    sorted_rets = np.sort(daily_rets_arr)
    def _var_cvar(percentile):
        idx = max(1, int(np.ceil(n * (1 - percentile / 100))))
        var_val = float(sorted_rets[idx - 1])
        cvar_val = float(np.mean(sorted_rets[:idx])) if idx > 0 else var_val
        return var_val, cvar_val
    var_95, cvar_95 = _var_cvar(95)
    var_99, cvar_99 = _var_cvar(99)

    running_max = np.maximum.accumulate(cum)
    dd_series = (cum - running_max) / running_max * 100
    ulcer = float(np.sqrt(np.mean(dd_series ** 2)))

    gains = daily_rets_arr[daily_rets_arr > 0].sum()
    losses = abs(daily_rets_arr[daily_rets_arr < 0].sum())
    profit_factor = float(gains / losses) if losses > 0 else float("inf")

    recovery_days = max((dp.get("recovery_days") or 0 for dp in dd_periods), default=0)

    return {
        "var_95": round(float(var_95) * 100, 4),
        "cvar_95": round(float(cvar_95) * 100, 4),
        "var_99": round(float(var_99) * 100, 4),
        "cvar_99": round(float(cvar_99) * 100, 4),
        "ulcer_index": round(ulcer, 4),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else None,
        "max_recovery_days": recovery_days,
    }


def compute_window_metrics(ec_segment, init_cap):
    if len(ec_segment) < 2:
        return {}
    if isinstance(ec_segment, list):
        vals = [e["total_value"] for e in ec_segment]
        dates = [e["date"] for e in ec_segment]
    elif isinstance(ec_segment, pd.DataFrame):
        vals = ec_segment["total_value"].tolist()
        dates = ec_segment["date"].tolist()
    else:
        return {}
    total_ret = (vals[-1] / init_cap - 1) * 100
    cum = np.array(vals) / init_cap
    daily_rets = np.diff(vals) / np.array(vals[:-1])
    n_days = len(daily_rets)
    win_rate = float(np.mean(daily_rets > 0)) if n_days > 0 else 0.0
    daily_avg = float(np.mean(daily_rets)) * 100
    ann_ret = (1 + total_ret / 100) ** (252 / n_days) - 1 if n_days > 0 else 0.0
    annualized_ret_pct = ann_ret * 100
    daily_std = float(np.std(daily_rets)) if n_days > 0 else 0.0
    annualized_vol = daily_std * np.sqrt(252) * 100
    sharpe = float((np.mean(daily_rets) / daily_std) * np.sqrt(252)) if daily_std != 0 else 0.0
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max * 100
    max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0
    if max_dd > 0:
        dd_end_idx = np.argmin(dd)
        dd_series = dd[:dd_end_idx + 1]
        dd_start_idx = np.argmax(running_max[:dd_end_idx + 1])
        mdd_start = str(dates[int(dd_start_idx)])
        mdd_end = str(dates[int(dd_end_idx)])
        mdd_duration = int(dd_end_idx - dd_start_idx)
    else:
        mdd_start = mdd_end = ""
        mdd_duration = 0
    calmar = ann_ret / (max_dd / 100) if max_dd > 0 else 0.0
    downside = daily_rets[daily_rets < 0]
    downside_std = float(np.std(downside)) if len(downside) > 1 else daily_std
    sortino = float((np.mean(daily_rets) / downside_std) * np.sqrt(252)) if downside_std != 0 else 0.0
    dd_periods = extract_drawdowns(vals, dates)
    risk_metrics = compute_longterm_risk_metrics(daily_rets, cum, dates, dd_periods)
    return {
        "strategy_return_pct": round(total_ret, 4),
        "annualized_return_pct": round(annualized_ret_pct, 4),
        "daily_avg_return_pct": round(daily_avg, 4),
        "daily_win_rate": round(win_rate, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "max_drawdown_details": {
            "start_date": mdd_start,
            "end_date": mdd_end,
            "duration_days": mdd_duration,
        },
        "drawdown_periods": dd_periods,
        "sharpe_ratio": round(sharpe, 4),
        "calmar_ratio": round(calmar, 4),
        "sortino_ratio": round(sortino, 4),
        "annualized_volatility_pct": round(annualized_vol, 4),
        "total_days": n_days,
        "latest_value": round(vals[-1], 2),
        **risk_metrics,
    }
