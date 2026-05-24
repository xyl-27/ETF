"""Regenerate all historical daily reports from backtest_state.json using the current template."""
import json
import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("SMTP_USER", "")
os.environ.setdefault("SMTP_PASSWORD", "")

from send_report import (
    build_report_html, _build_pred_signals_table, _build_model_stats_table,
    _compute_health_score, _build_health_table, _add_window, _compute_max_drawdown,
    _load_etf_names, _xueqiu_url, PROJECT_ROOT,
)
from daily_eval import _compute_model_stats, _resolve_report_key, _extract_drawdowns, _compute_longterm_risk_metrics

STATE_PATH = PROJECT_ROOT / "output" / "backtest_state.json"
HISTORY_DIR = PROJECT_ROOT / "output" / "history_report"
DATA_PATH = PROJECT_ROOT / "etf_data" / "etf_74.csv"
HS300_CODE = "510300.XSHG"


def load_etf_data():
    df = pd.read_csv(DATA_PATH)
    df["日期"] = pd.to_datetime(df["日期"])
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    return df


def compute_ranks_at_date(raw_df, target_date, window=5):
    """Compute window-day return ranking at target_date."""
    target_dt = pd.Timestamp(target_date)
    sub = raw_df[raw_df["日期"] <= target_dt]
    dates = sorted(sub["日期"].unique())
    if len(dates) < window + 1:
        return {}
    end_date = dates[-1]
    start_date = dates[-(window + 1)]
    period = sub[sub["日期"].between(start_date, end_date)].copy()
    pivot = period.pivot_table(index="股票代码", columns="日期", values="收盘")
    if len(pivot.columns) < 2:
        return {}
    ret = (pivot.iloc[:, -1] / pivot.iloc[:, 0] - 1) * 100
    ret = ret.dropna().sort_values(ascending=False)
    return {code: i + 1 for i, code in enumerate(ret.index)}, ret


def build_market_monitor_section(raw_df, seq, target_date, holdings_at_date, etf_names):
    """Build regime table + ETF rankings for a historical date."""
    target_dt = pd.Timestamp(target_date)
    hs_raw = raw_df[raw_df["股票代码"] == HS300_CODE].sort_values("日期")
    hs_period = hs_raw[hs_raw["日期"] <= target_dt].copy()
    if len(hs_period) < 21:
        return ""

    # HS300 regime
    hs_period["return_pct"] = hs_period["收盘"].pct_change() * 100
    hs_period["rolling_20d"] = hs_period["收盘"].pct_change(20) * 100
    hs_period["rolling_vol"] = hs_period["return_pct"].rolling(20).std() * np.sqrt(252)
    last = hs_period.iloc[-1]
    r20 = last.get("rolling_20d", 0)
    vol = last.get("rolling_vol", 0)
    if r20 > 5 and vol < 30:
        regime = "bull"
        regime_cn = "牛市 ↑"
    elif r20 < -5:
        regime = "bear"
        regime_cn = "熊市 ↓"
    else:
        regime = "sideways"
        regime_cn = "震荡 →"

    # Market breadth
    all_codes = raw_df["股票代码"].unique()
    bull_count = sideways_count = bear_count = total_valid = 0
    for code in all_codes:
        sub = raw_df[raw_df["股票代码"] == code].sort_values("日期")
        sub = sub[sub["日期"] <= target_dt]
        if len(sub) < 21:
            continue
        total_valid += 1
        ret20 = (sub["收盘"].iloc[-1] / sub["收盘"].iloc[-21] - 1) * 100
        vol20 = sub["收盘"].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
        if ret20 > 5 and vol20 < 30:
            bull_count += 1
        elif ret20 < -5:
            bear_count += 1
        else:
            sideways_count += 1
    breadth_str = f"🔴牛市 {bull_count/total_valid*100:.0f}% / 🟡震荡 {sideways_count/total_valid*100:.0f}% / 🟢熊市 {bear_count/total_valid*100:.0f}% (共{total_valid}只ETF)" if total_valid > 0 else ""

    regime_labels_display = {"bull": "牛市 ↑", "bear": "熊市 ↓", "sideways": "震荡 →"}

    # ETF rankings
    rank_map, ret_series = compute_ranks_at_date(raw_df, target_date, 5)
    all_codes_list = list(ret_series.index) if ret_series is not None else []

    top10 = []
    if ret_series is not None:
        for code, r in ret_series.head(10).items():
            top10.append({"code": code, "name": etf_names.get(code, ""), "return": round(r, 2),
                          "held": code in holdings_at_date})

    bot10 = []
    if ret_series is not None:
        for code, r in ret_series.tail(10).items():
            bot10.append({"code": code, "name": etf_names.get(code, ""), "return": round(r, 2),
                          "held": code in holdings_at_date})

    current_rebalance_date = ""
    prev_rebalance_date = ""
    preds = seq.get("predictions_history", [])
    if preds:
        for p in reversed(preds):
            if p["date"] <= target_date:
                current_rebalance_date = p["date"]
                break
    if len(preds) >= 2:
        prev_idx = None
        for i, p in enumerate(preds):
            if p["date"] < current_rebalance_date:
                prev_idx = i
        if prev_idx is not None:
            prev_rebalance_date = preds[prev_idx]["date"]
        else:
            for i in range(len(preds) - 2, -1, -1):
                if preds[i]["date"] <= target_date:
                    prev_rebalance_date = preds[i]["date"]
                    break

    # 调仓日排名（用调仓当日的5日收益排名）
    rebalance_rank_map = {}
    if current_rebalance_date:
        rebalance_rank_map, _ = compute_ranks_at_date(raw_df, current_rebalance_date, 5)

    # Holdings data
    holdings_data = []
    if rank_map and holdings_at_date:
        for code in sorted(holdings_at_date):
            r = ret_series.get(code) if ret_series is not None else None
            if r is not None:
                rr = rebalance_rank_map.get(code, 0)
                holdings_data.append({
                    "code": code,
                    "name": etf_names.get(code, ""),
                    "return": round(r, 2),
                    "rank": rank_map.get(code, 0),
                    "rank_at_rebalance": rr,
                    "total": len(all_codes_list),
                    "rebalance_date": current_rebalance_date,
                })
        holdings_data.sort(key=lambda x: x["rank"])

    # Previous holdings
    prev_holdings_data = []
    if prev_rebalance_date and rank_map:
        prev_rank_map, _ = compute_ranks_at_date(raw_df, prev_rebalance_date, 5)
        prev_held = set()
        for t in seq.get("trades", []):
            if t["date"] > prev_rebalance_date:
                break
            if t["action"] == "买入":
                prev_held.add(t["stock"])
            elif t["action"] == "卖出":
                prev_held.discard(t["stock"])
        for code in sorted(prev_held):
            r = ret_series.get(code) if ret_series is not None else None
            if r is not None:
                pr = prev_rank_map.get(code, 0)
                prev_holdings_data.append({
                    "code": code,
                    "name": etf_names.get(code, ""),
                    "return": round(r, 2),
                    "rank": rank_map.get(code, 0),
                    "rank_at_rebalance": pr,
                    "total": len(all_codes_list),
                    "rebalance_date": prev_rebalance_date,
                    "days_ago": 0,
                })
        prev_holdings_data.sort(key=lambda x: x["rank"])

    # Compute per-regime stats from model equity curve
    ec = seq.get("equity_curve", [])
    ec_trunc = [e for e in ec if e["date"] <= target_date]
    stats = {}
    if len(ec_trunc) >= 3:
        hs_period["regime"] = ""
        hs_period.loc[hs_period["rolling_20d"] > 5, "regime"] = "bull"
        hs_period.loc[hs_period["rolling_20d"] < -5, "regime"] = "bear"
        hs_period.loc[(hs_period["rolling_20d"] <= 5) & (hs_period["rolling_20d"] >= -5), "regime"] = "sideways"
        regime_dict = dict(zip(hs_period["日期"].dt.strftime("%Y-%m-%d"), hs_period["regime"]))
        ec_dates = [e["date"] for e in ec_trunc]
        ec_values = [e["total_value"] for e in ec_trunc]
        records = []
        for i, d in enumerate(ec_dates):
            reg = regime_dict.get(d)
            if reg and reg in ("bull", "bear", "sideways"):
                model_ret = (ec_values[i] / ec_values[i - 1] - 1) * 100 if i > 0 else 0
                records.append({"date": d, "model_return": model_ret, "regime": reg, "model_value": ec_values[i]})
        if records:
            df = pd.DataFrame(records)
            for regime_key in ["bull", "bear", "sideways"]:
                sub = df[df["regime"] == regime_key]
                if len(sub) < 3:
                    continue
                model_total = (sub["model_value"].iloc[-1] / sub["model_value"].iloc[0] - 1) * 100
                beat_rate = (sub["model_return"] > 0).mean()
                stats[regime_key] = {
                    "days": len(sub),
                    "model_return": round(model_total, 2),
                    "hs300_return": 0,
                    "excess_return": round(model_total, 2),
                    "beat_rate": round(beat_rate, 4),
                    "model_win_rate": round(beat_rate, 4),
                }
            model_total_all = (ec_values[-1] / ec_values[0] - 1) * 100
            win_all = (df["model_return"] > 0).mean()
            stats["all"] = {
                "days": len(df),
                "model_return": round(model_total_all, 2),
                "hs300_return": 0,
                "excess_return": round(model_total_all, 2),
                "beat_rate": round(win_all, 4),
                "model_win_rate": round(win_all, 4),
            }

    # Build HTML
    from market_monitor import build_regime_table_html, build_etf_rankings_html
    regime_html = build_regime_table_html(stats, {
        "regime": regime,
        "rolling_20d_return": r20 if not pd.isna(r20) else 0,
        "rolling_vol": vol if not pd.isna(vol) else 0,
    }, {"bull_pct": bull_count/total_valid*100 if total_valid else 0,
        "sideways_pct": sideways_count/total_valid*100 if total_valid else 0,
        "bear_pct": bear_count/total_valid*100 if total_valid else 0,
        "total": total_valid} if total_valid else None)

    rank_start = ""
    rank_end = ""
    if ret_series is not None:
        rank_start = str(pd.Timestamp(target_date) - pd.Timedelta(days=5)).split(" ")[0]
        rank_end = target_date
    rank_date = f"{rank_start}~{rank_end}" if rank_start else ""

    rank_html = ""
    if top10:
        rank_html = build_etf_rankings_html(top10, bot10, holdings_data, prev_holdings_data, rank_date)

    return regime_html + "<br>" + rank_html if rank_html else regime_html


def simulate_state_at_date(seq, target_date, raw_df, initial_capital=100000):
    """Reconstruct positions, cash, and metrics as of target_date."""
    target_dt = pd.Timestamp(target_date)
    ec = seq.get("equity_curve", [])
    trades = seq.get("trades", [])

    # Truncate equity curve
    ec_trunc = [e for e in ec if e["date"] <= target_date]
    if not ec_trunc:
        return None
    ec_dates = [e["date"] for e in ec_trunc]
    ec_values = [e["total_value"] for e in ec_trunc]

    # Simulate positions and cash from trades up to target_date
    held = {}
    cash = initial_capital
    for t in trades:
        if t["date"] > target_date:
            break
        amt = t.get("amount", t["shares"] * t["price"])
        tc = t.get("trade_cost", 0)
        if t["action"] == "买入":
            held[t["stock"]] = held.get(t["stock"], 0) + t["shares"]
            cash -= amt + tc
        elif t["action"] == "卖出":
            held[t["stock"]] = held.get(t["stock"], 0) - t["shares"]
            if held[t["stock"]] <= 0:
                del held[t["stock"]]
            cash += amt - tc

    positions = {}
    for sid, shares in held.items():
        sub = raw_df[raw_df["股票代码"] == sid]
        sub = sub[sub["日期"] <= target_dt]
        if sub.empty:
            continue
        last_row = sub.iloc[-1]
        buy_trades = [t for t in trades if t["action"] == "买入" and t["stock"] == sid and t["date"] <= target_date]
        sell_trades = [t for t in trades if t["action"] == "卖出" and t["stock"] == sid and t["date"] <= target_date]
        total_cost = sum(t["amount"] + t.get("trade_cost", 0) for t in buy_trades)
        total_sold_shares = sum(t["shares"] for t in sell_trades)
        total_bought_shares = sum(t["shares"] for t in buy_trades)
        remaining = total_bought_shares - total_sold_shares
        if remaining > 0:
            cost_per_share = total_cost / total_bought_shares
            remaining_cost = cost_per_share * remaining
        else:
            remaining_cost = 0
        positions[sid] = {
            "shares": shares,
            "cost": round(remaining_cost, 2),
            "buy_price": buy_trades[-1]["price"] if buy_trades else 0,
            "buy_date": buy_trades[-1]["date"] if buy_trades else "",
        }

    # Compute today_pnl
    today_pnl_total = 0
    today_pnl_positions = []
    if len(ec_trunc) >= 2:
        prev_val = ec_trunc[-2]["total_value"]
        today_pnl_total = ec_trunc[-1]["total_value"] - prev_val
    for sid, pinfo in positions.items():
        sub = raw_df[raw_df["股票代码"] == sid]
        sub = sub[sub["日期"] <= target_dt]
        if len(sub) < 2:
            continue
        today_close = sub.iloc[-1]["收盘"]
        yesterday_close = sub.iloc[-2]["收盘"]
        pos_pnl = (today_close - yesterday_close) * pinfo["shares"] if yesterday_close else 0
        pnl_pct = ((today_close / yesterday_close) - 1) * 100 if yesterday_close else 0
        today_pnl_positions.append({
            "stock_id": sid,
            "shares": pinfo["shares"],
            "today_close": float(today_close),
            "yesterday_close": float(yesterday_close),
            "pnl": round(pos_pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
        })

    # Compute metrics
    total_val = ec_trunc[-1]["total_value"]
    strategy_ret = (total_val / initial_capital - 1) * 100
    n_days = len(ec_trunc)
    ann = ((1 + strategy_ret / 100) ** (252 / n_days) - 1) * 100 if n_days > 1 else 0
    wins = sum(1 for i in range(1, len(ec_values)) if ec_values[i] > ec_values[i - 1])
    win_rate = wins / (len(ec_values) - 1) if len(ec_values) > 1 else 0
    dd = _compute_max_drawdown(ec_values)

    # HS300 return (from backtest start date, not from HS300 data start)
    hs_raw = raw_df[raw_df["股票代码"] == HS300_CODE].sort_values("日期")
    hs_first_date = ec_trunc[0]["date"]
    hs_period = hs_raw[(hs_raw["日期"] >= pd.Timestamp(hs_first_date)) & (hs_raw["日期"] <= target_dt)]
    hs_ret = 0
    if len(hs_period) >= 2:
        hs_ret = (hs_period["收盘"].iloc[-1] / hs_period["收盘"].iloc[0] - 1) * 100

    # Sharpe, Calmar, Sortino
    daily_rets = []
    for i in range(1, len(ec_values)):
        daily_rets.append(ec_values[i] / ec_values[i - 1] - 1)
    sr = 0
    if len(daily_rets) > 1 and np.std(daily_rets) > 0:
        sr = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
    calmar = strategy_ret / dd if dd > 0 else 0
    downside = np.std([r for r in daily_rets if r < 0]) * np.sqrt(252) if any(r < 0 for r in daily_rets) else 1
    sortino = strategy_ret / (downside * 100) if downside > 0 else 0  # approximate

    # Build metrics dict
    metrics = {
        "strategy_return_pct": round(strategy_ret, 2),
        "annualized_return_pct": round(ann, 2),
        "daily_win_rate": round(win_rate, 4),
        "max_drawdown_pct": round(dd, 2),
        "latest_value": round(total_val, 2),
        "hs300_return_pct": round(hs_ret, 2),
        "excess_return_pct": round(strategy_ret - hs_ret, 2),
        "sharpe_ratio": round(sr, 2),
        "calmar_ratio": round(calmar, 2),
        "sortino_ratio": round(sortino, 2),
        "total_days": n_days,
    }

    dd_periods = _extract_drawdowns(ec_values, ec_dates)
    if dd_periods:
        metrics["drawdown_periods"] = dd_periods

    # 长期风险指标
    cum_arr = np.array(ec_values) / initial_capital
    risk_metrics = _compute_longterm_risk_metrics(np.array(daily_rets), cum_arr, ec_dates, dd_periods)
    metrics.update(risk_metrics)

    # Add window metrics
    if len(ec_values) >= 4:
        _add_window(metrics, "window_3d", ec_values, 3)
    if len(ec_values) >= 6:
        _add_window(metrics, "window_5d_real", ec_values, 5)
    if len(ec_values) >= 11:
        _add_window(metrics, "window_10d", ec_values, 10)
    if len(ec_values) >= 21:
        _add_window(metrics, "window_1m", ec_values, 20)

    # Next rebalance date
    preds = seq.get("predictions_history", [])
    next_rb = ""
    for p in preds:
        if p["date"] > target_date:
            next_rb = p["date"]
            break

    # Is rebalance day?
    pred_dates = [p["date"] for p in preds]
    is_rb = target_date in pred_dates

    # Today's trades
    today_trades = [t for t in trades if t["date"] == target_date]

    # HS300 curve
    hs300_curve = []
    hs_raw = raw_df[raw_df["股票代码"] == HS300_CODE].sort_values("日期")
    hs_first_date = ec_trunc[0]["date"]
    hs_period = hs_raw[(hs_raw["日期"] >= pd.Timestamp(hs_first_date)) & (hs_raw["日期"] <= target_dt)]
    if not hs_period.empty:
        hs300_start_price = float(hs_period["收盘"].iloc[0])
        for _, row in hs_period.iterrows():
            hs300_curve.append({
                "date": row["日期"].strftime("%Y-%m-%d"),
                "total_value": round(float(row["收盘"]) / hs300_start_price * initial_capital, 2),
            })

    holdings_at_date = set(sid for sid in positions.keys()
                           if positions[sid].get("shares", 0) > 0)
    mm_section = build_market_monitor_section(raw_df, seq, target_date, holdings_at_date, _load_etf_names())

    return {
        "date": target_date,
        "holdings": positions,
        "positions": positions,
        "equity_curve": ec_trunc,
        "metrics": metrics,
        "cash": round(cash, 2),
        "total_value": round(total_val, 2),
        "today_pnl": {
            "total_pnl": round(today_pnl_total, 2),
            "positions": today_pnl_positions,
        },
        "today_trades": today_trades,
        "all_today_trades": today_trades,
        "is_rebalance_day": is_rb,
        "next_rebalance_date": next_rb,
        "hs300_curve": hs300_curve,
        "predictions_history": [p for p in preds if p["date"] <= target_date],
        "trades": [t for t in trades if t["date"] <= target_date],
        "trade_mode": "open",
        "sequences_summary": {},
        "market_monitor_section": mm_section,
    }


def build_report(report_state, seq_key, all_sequences, raw_df, top_k=3, position_pct=0.95, weight_strategy="equal", strategy_params=None):
    """Generate HTML for one historical date."""
    seq_data = all_sequences.get(seq_key, {})
    metrics = report_state["metrics"]
    # Build price lookup from today_pnl positions
    close_prices = {p["stock_id"]: p["today_close"] for p in report_state.get("today_pnl", {}).get("positions", []) if p.get("today_close", 0) > 0}
    holdings_list = []
    etf_names = _load_etf_names()
    target_dt = pd.Timestamp(report_state["date"])
    for sid, pinfo in report_state["holdings"].items():
        price = close_prices.get(sid, 0)
        sub = raw_df[raw_df["股票代码"] == sid]
        if not price:
            tc_s = sub.loc[sub["日期"] == target_dt, "收盘"]
            price = float(tc_s.values[0]) if not tc_s.empty else 0
        hl_s = sub.loc[sub["日期"] == target_dt, "涨停价"]
        ll_s = sub.loc[sub["日期"] == target_dt, "跌停价"]
        buy_date = pinfo.get("buy_date", "")
        buy_factor = 1.0
        if buy_date:
            bf_s = sub.loc[sub["日期"] == pd.Timestamp(buy_date), "复权因子"]
            buy_factor = float(bf_s.values[0]) if not bf_s.empty else 1.0
        price_display_s = sub.loc[sub["日期"] == target_dt, "收盘_原始"]
        price_display = float(price_display_s.values[0]) if not price_display_s.empty else price
        buy_price_display = 0
        if buy_date:
            bpd_s = sub.loc[sub["日期"] == pd.Timestamp(buy_date), "收盘_原始"]
            buy_price_display = float(bpd_s.values[0]) if not bpd_s.empty else pinfo.get("buy_price", 0)
        holdings_list.append({
            "stock_id": sid,
            "name": etf_names.get(sid, ""),
            "price": price,
            "price_display": price_display,
            "buy_price": round(pinfo.get("buy_price", 0), 4),
            "buy_price_display": round(buy_price_display, 4),
            "buy_date": buy_date,
            "buy_factor": buy_factor,
            "shares": pinfo["shares"],
            "cost": pinfo.get("cost", 0),
            "high_limit": round(float(hl_s.values[0]), 4) if not hl_s.empty else 0,
            "low_limit": round(float(ll_s.values[0]), 4) if not ll_s.empty else 0,
        })

    # Build sequences_summary for this date
    sequences_summary = {}
    for key, seq in all_sequences.items():
        ec = [e for e in seq.get("equity_curve", []) if e["date"] <= report_state["date"]]
        trades = [t for t in seq.get("trades", []) if t["date"] <= report_state["date"]]
        today_pnl = {"total_pnl": 0, "positions": []}
        if len(ec) >= 2:
            today_pnl["total_pnl"] = round(ec[-1]["total_value"] - ec[-2]["total_value"], 2)
        seq_current_prices = {}
        for pos in report_state.get("today_pnl", {}).get("positions", []):
            if pos["today_close"] > 0:
                seq_current_prices[pos["stock_id"]] = pos["today_close"]
        model_stats = _compute_model_stats(trades, seq_current_prices, report_date=report_state["date"])
        sequences_summary[key] = {
            "metrics": seq.get("metrics", {}),
            "cash": 0,
            "positions_count": len(report_state["holdings"]),
            "trades_count": len(trades),
            "trades": trades,
            "today_pnl": today_pnl,
            "model_stats": model_stats,
            "equity_curve": ec,
            "skipped_trades": [],
            "predictions_history": [p for p in seq.get("predictions_history", []) if p["date"] <= report_state["date"]],
        }

    model_display = seq_key.replace("search_", "").replace("_exp_", " ")

    model_stats_section = _build_model_stats_table(sequences_summary)

    health_scores = {}
    for key, seq in sequences_summary.items():
        ec = seq.get("equity_curve", [])
        if ec:
            health_scores[key] = _compute_health_score(seq)
    health_section = _build_health_table(health_scores)

    equity_data = {}
    for key, seq in sequences_summary.items():
        ec = seq.get("equity_curve", [])
        if ec:
            equity_data[key.replace("search_", "").replace("_exp_", " ")] = ec
    hs300 = report_state.get("hs300_curve", [])
    if hs300:
        equity_data["沪深300"] = hs300

    pred_signals_section = _build_pred_signals_table(seq_data, report_state["date"], weight_strategy=weight_strategy, strategy_params=strategy_params, top_k=top_k, position_pct=position_pct)

    # Compute today_pnl
    today_pnl_total = report_state["today_pnl"]["total_pnl"]
    today_pnl_positions = report_state["today_pnl"]["positions"]

    # 调仓胜率
    master_stats = sequences_summary.get(seq_key, {}).get("model_stats", {})
    rebalance_win_rate = master_stats.get("total_win_rate_pct")

    html = build_report_html(
        date=report_state["date"],
        model_display=model_display,
        total_value=report_state["total_value"],
        cash=report_state["cash"],
        holdings=holdings_list,
        trades_list=report_state["all_today_trades"],
        metrics=metrics,
        next_rebalance=report_state["next_rebalance_date"],
        is_rebalance=report_state["is_rebalance_day"],
        today_pnl_total=today_pnl_total,
        today_pnl_positions=today_pnl_positions,
        model_stats_section=model_stats_section,
        equity_data=equity_data,
        health_section=health_section,
        trade_mode="open",
        pred_signals_section=pred_signals_section,
        market_monitor_section=report_state.get("market_monitor_section", ""),
        rebalance_win_rate=rebalance_win_rate,
        source="本地回测",
    )
    return html


def main():
    print("=" * 60)
    print("Regenerating historical reports")
    print("=" * 60)

    if not STATE_PATH.exists():
        print(f"Error: {STATE_PATH} not found")
        return

    state = json.load(open(STATE_PATH))
    sequences = state.get("sequences", {})
    if not sequences:
        print("Error: no sequences in backtest_state.json")
        return

    report_key = _resolve_report_key(sequences)
    if not report_key:
        report_key = next(k for k in sequences if k != "hs300")
    print(f"Primary sequence: {report_key}")

    seq = sequences[report_key]
    ec = seq.get("equity_curve", [])
    if len(ec) < 2:
        print("Error: equity curve too short")
        return

    raw_df = load_etf_data()

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    date_labels = {
        "2026-04-01": "调仓日",
        "2026-04-09": "调仓日",
        "2026-04-16": "调仓日",
        "2026-04-23": "调仓日",
        "2026-04-30": "调仓日",
        "2026-05-12": "调仓日",
    }

    for day in ec:
        date_str = day["date"]
        if date_str == ec[0]["date"]:
            continue  # skip first day (no PnL)

        label = date_labels.get(date_str, "")
        filename = f"{date_str}({label}).html" if label else f"{date_str}.html"
        filepath = HISTORY_DIR / filename

        print(f"  [{date_str}] generating...", end=" ")

        try:
            report_state = simulate_state_at_date(seq, date_str, raw_df)
            if report_state is None:
                print("skip (no data)")
                continue

            # 从 config.yaml 加载加权参数
            _cfg_path = PROJECT_ROOT / "config.yaml"
            try:
                import yaml
                with open(_cfg_path, encoding="utf-8") as _f:
                    _cfg = yaml.safe_load(_f) or {}
            except Exception:
                _cfg = {}
            _ws = _cfg.get("email", {}).get("weight_strategy", "equal")
            _sp = _cfg.get("strategy_params", {})
            _tk = _cfg.get("top_k", 3)
            _pp = _cfg.get("position_pct", 0.95)
            html = build_report(report_state, report_key, sequences, raw_df, top_k=_tk, position_pct=_pp, weight_strategy=_ws, strategy_params=_sp)
            filepath.write_text(html, encoding="utf-8")
            print(f"saved to {filename}")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. {len(ec) - 1} reports generated in {HISTORY_DIR}")


if __name__ == "__main__":
    main()
