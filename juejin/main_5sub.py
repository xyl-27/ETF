# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import json
import math
import os
from collections import defaultdict
from datetime import datetime

'''
掘金回测策略：5错峰子账户
- 每 REBALANCE_DAYS 个交易日调仓一次
- SUB_ACCOUNTS 个子账户，起始日依次错开1天
- 每天只有 1 个子账户调仓，其余不动
'''

PREDICTIONS_PATH = r"C:\Users\xyl\Desktop\ETF\output\predictions.json"
STATE_PATH = r"C:\Users\xyl\Desktop\ETF\output\backtest_state.json"
JUEJIN_STATE_PATH = r"C:\Users\xyl\Desktop\ETF\output\juejin_state.json"
RESULT_PATH = r"C:\Users\xyl\Desktop\ETF\output\juejin_result.json"
YAML_PATH = r"C:\Users\xyl\Desktop\ETF\config.yaml"

MODEL_KEY = None
TOP_K = 3
REBALANCE_DAYS = 5
SUB_ACCOUNTS = 5
START_DATE = "2026-04-01"
TRADE_MODE = "open"
POSITION_PCT = 0.95
WEIGHT_STRATEGY = "equal"
STRATEGY_PARAMS = {}
CACHED_CSV = None
ETF_ROOT = r"C:\Users\xyl\Desktop\ETF"
DATA_CSV_PATH = os.path.join(ETF_ROOT, "etf_data", "etf_74.csv")

def get_sub_single_pct():
    return (POSITION_PCT / SUB_ACCOUNTS) / TOP_K


def load_config_from_yaml():
    import yaml
    global MODEL_KEY, TOP_K, REBALANCE_DAYS, START_DATE, TRADE_MODE, POSITION_PCT, WEIGHT_STRATEGY, STRATEGY_PARAMS, CACHED_CSV
    try:
        with open(YAML_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        jcfg = data.get("juejin", {})
        TOP_K = int(jcfg.get("top_k", TOP_K))
        REBALANCE_DAYS = int(jcfg.get("rebalance_days", REBALANCE_DAYS))
        START_DATE = str(jcfg.get("start_date", START_DATE))
        TRADE_MODE = str(jcfg.get("trade_mode", TRADE_MODE))
        POSITION_PCT = float(jcfg.get("position_pct", POSITION_PCT))
        WEIGHT_STRATEGY = str(jcfg.get("weight_strategy", WEIGHT_STRATEGY))
        raw_params = jcfg.get("strategy_params", {})
        STRATEGY_PARAMS = dict(raw_params)
        if WEIGHT_STRATEGY == "softmax":
            STRATEGY_PARAMS.setdefault("temperature", float(jcfg.get("weight_temperature", 1.0)))
        if WEIGHT_STRATEGY in ("risk_parity", "score_risk", "score_risk_v1"):
            STRATEGY_PARAMS.setdefault("vol_window", int(raw_params.get("vol_window", 20)))
        if data.get("model_key"):
            MODEL_KEY = str(data["model_key"])
        else:
            for m in data.get("models", []):
                if m.get("enabled", True):
                    exp_dir = m["dir"]
                    import re
                    parent = os.path.basename(os.path.dirname(exp_dir))
                    parent = re.sub(r'_\d+_\d+', '', parent)
                    name = os.path.basename(exp_dir)
                    MODEL_KEY = f"{parent}_{name}"
                    break
        print(f"[配置] 从 {YAML_PATH} 读取: top_k={TOP_K} rebalance_days={REBALANCE_DAYS} "
              f"mode={TRADE_MODE} position_pct={POSITION_PCT} model_key={MODEL_KEY}")
    except Exception as e:
        print(f"[配置] 读取 yaml 失败，使用默认值: {e}")


load_config_from_yaml()

_BACKTEST_DATES = None


def get_backtest_end_date():
    if os.path.exists(PREDICTIONS_PATH):
        try:
            with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f).get("_meta", {})
            dates = meta.get("backtest_dates", [])
            if dates:
                return dates[-1]
        except Exception:
            pass
    import datetime
    today = datetime.date.today()
    while True:
        if today.weekday() < 5:
            return today.strftime("%Y-%m-%d")
        today -= datetime.timedelta(days=1)


def to_gm_symbol(stock_id):
    code, exchange = stock_id.split(".")
    exchange_map = {"XSHG": "SHSE", "XSHE": "SZSE"}
    return f"{exchange_map.get(exchange, exchange)}.{code}"


def gm_to_local(symbol):
    exchange_map = {"SHSE": "XSHG", "SZSE": "XSHE"}
    exchange, code = symbol.split(".")
    return f"{code}.{exchange_map.get(exchange, exchange)}"


def load_predictions_by_date():
    global _BACKTEST_DATES
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"[策略] 未找到 {PREDICTIONS_PATH}，请先运行 daily_eval --predictions-only")
        return {}
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        all_preds = json.load(f)
    meta = all_preds.pop("_meta", {})
    _BACKTEST_DATES = meta.get("backtest_dates", [])
    if _BACKTEST_DATES:
        print(f"[策略] 加载交易日历: {len(_BACKTEST_DATES)} 天 ({_BACKTEST_DATES[0]} ~ {_BACKTEST_DATES[-1]})")
    model_preds = all_preds.get(MODEL_KEY, {})
    if not model_preds:
        print(f"[策略] predictions.json 中未找到模型 {MODEL_KEY}")
        return {}
    print(f"[策略] 加载 {MODEL_KEY} 预测，共 {len(model_preds)} 个交易日")
    return model_preds


def _get_account_value(account, key, default=0):
    keys_to_try = [key]
    if key == "cash":
        keys_to_try.append("cash_balance")
    for k in keys_to_try:
        try:
            val = account[k]
            if val is not None:
                return float(val)
        except Exception:
            pass
        try:
            val = getattr(account, k, None)
            if val is not None:
                return float(val)
        except Exception:
            pass
    return default


def _get_pos_market_value(pos):
    try:
        return float(pos.get("market_value", 0))
    except Exception:
        try:
            return float(pos["market_value"])
        except Exception:
            return 0


def compute_sub_rebalance_dates(backtest_dates, start_date, rebalance_days, n_sub):
    """计算 n_sub 组错开1天的调仓日历"""
    sorted_dates = sorted(set(backtest_dates))
    start_idx = 0
    for i, d in enumerate(sorted_dates):
        if d >= start_date:
            start_idx = i
            break
    sub_dates = [set() for _ in range(n_sub)]
    for k in range(n_sub):
        for i in range(start_idx + k, len(sorted_dates), rebalance_days):
            sub_dates[k].add(sorted_dates[i])
    return sub_dates, start_idx


def init(context):
    context.predictions = load_predictions_by_date()
    if not context.predictions:
        raise ValueError("无预测数据，策略退出")

    context.end_date = _BACKTEST_DATES[-1] if _BACKTEST_DATES else get_backtest_end_date()
    cal = _BACKTEST_DATES or sorted(context.predictions.keys())
    context.calendar = sorted(cal)

    # 5错峰子账户调仓日历
    context.sub_rebalance_dates, context.start_idx = compute_sub_rebalance_dates(
        cal, START_DATE, REBALANCE_DAYS, SUB_ACCOUNTS)
    for k in range(SUB_ACCOUNTS):
        print(f"  [子账户{k}] 调仓日({len(context.sub_rebalance_dates[k])}天): "
              f"{sorted(context.sub_rebalance_dates[k])[:5]}...")

    # 每日调试：对比 CSV 日历 vs GM 日历
    try:
        gm_dates_raw = get_trading_dates(exchange='SHSE', start_date=START_DATE, end_date=context.end_date)
        gm_date_strs = [str(d) for d in gm_dates_raw]
        csv_only = set(cal) - set(gm_date_strs)
        gm_only = set(gm_date_strs) - set(cal)
        if csv_only:
            print(f"  [日历差异] 仅 CSV 有: {sorted(csv_only)}")
        if gm_only:
            print(f"  [日历差异] 仅 GM 有: {sorted(gm_only)}")
    except Exception as e:
        print(f"  [日历] GM日历查询失败: {e}")

    context.trade_mode = TRADE_MODE
    context.processed_dates = set()
    context.executed_trades = []
    context.daily_equity = [{"date": START_DATE, "total_value": 100000}]
    context.track_cash = 100000
    context.snapshot_positions = {}
    context.snapshot_cash = 0
    context.rebalance_snapshots = []
    context.daily_log = []

    # 子账户虚拟目标
    context.sub_targets = [{} for _ in range(SUB_ACCOUNTS)]
    # sub_targets[k] = { "513100.XSHG": to_gm_symbol, ... }

    trade_time = '09:31:00' if TRADE_MODE == 'open' else '14:55:00'
    print(f"[策略] 子账户={SUB_ACCOUNTS} 交易模式: {'开盘交易' if TRADE_MODE == 'open' else '收盘交易'} "
          f"执行时间: {trade_time}")
    schedule(schedule_func=algo, date_rule='1d', time_rule=trade_time)
    if TRADE_MODE == 'open':
        schedule(schedule_func=record_equity, date_rule='1d', time_rule='15:30:00')


def record_equity(context):
    now_str = context.now.strftime('%Y-%m-%d')
    try:
        account = context.account()
        cash = getattr(context, 'track_cash', 100000)
        pos_val = sum(_get_pos_market_value(p) for p in account.positions())
        total_val = round(cash + pos_val, 2)
        already_recorded = any(e["date"] == now_str for e in context.daily_equity)
        if total_val > 0 and not already_recorded:
            context.daily_equity.append({"date": now_str, "total_value": total_val})
        if len(context.daily_equity) % 5 == 2:
            print(f"  [净值] {now_str} 总值={total_val:.2f} 现金={cash:.2f} 持仓={pos_val:.2f}")
    except Exception as e:
        print(f"  [净值] 记录失败: {e}")


def algo(context):
    now_str = context.now.strftime('%Y-%m-%d')

    if now_str not in context.calendar:
        return
    cal_idx = context.calendar.index(now_str)
    sub_k = (cal_idx - context.start_idx) % SUB_ACCOUNTS

    if now_str not in context.sub_rebalance_dates[sub_k]:
        return
    if now_str in context.processed_dates:
        return
    context.processed_dates.add(now_str)

    # 预测日期
    if context.trade_mode == "open":
        cal = sorted(context.predictions.keys())
        idx = cal.index(now_str) if now_str in cal else -1
        pred_date = cal[idx - 1] if idx > 0 else now_str
    else:
        pred_date = now_str

    today_preds = context.predictions.get(pred_date, [])
    if not today_preds:
        print(f"[策略] {now_str} 子账户{sub_k} 无预测数据，跳过")
        return

    # 调仓前账户信息
    pre_account = context.account()
    pre_cash = _get_account_value(pre_account, "cash")
    pre_pos_val = sum(_get_pos_market_value(p) for p in pre_account.positions())
    pre_total = pre_cash + pre_pos_val
    print(f"\n{'='*50}")
    print(f"[调仓日] {now_str} 子账户{sub_k}")

    # 子账户 k 的新目标
    new_locals = {p["stock_id"] for p in today_preds[:TOP_K]}
    context.sub_targets[sub_k] = {local: to_gm_symbol(local) for local in new_locals}

    # 汇总全部子账户的全局目标权重
    global_w = defaultdict(float)
    for targets in context.sub_targets:
        for local, sym in targets.items():
            global_w[local] += get_sub_single_pct()

    top_k_stocks = {p["stock_id"] for p in today_preds[:TOP_K]}
    print(f"[策略] 子账户{sub_k} Top-{TOP_K}: {', '.join(top_k_stocks)}")

    # 卖: 实际持仓中不在任何子账户目标里的
    current_pos = context.account().positions()
    for pos in current_pos:
        local = gm_to_local(pos["symbol"])
        if local not in global_w:
            order_target_percent(symbol=pos["symbol"], percent=0,
                                 order_type=OrderType_Market,
                                 position_side=PositionSide_Long)
            print(f"[策略] 卖出 {local}({pos['symbol']})")

    # 买: 按全局权重同步每个symbol
    for local, w in global_w.items():
        if w <= 0:
            continue
        sym = to_gm_symbol(local)
        order_target_percent(symbol=sym, percent=w,
                             order_type=OrderType_Market,
                             position_side=PositionSide_Long)
        print(f"[策略] 调仓 {local}({sym}) → {w:.2%}")

    print(f"[策略] 子账户{sub_k} 调仓完成 (全局共{len(global_w)}只目标)")

    # 调仓快照
    try:
        post_account = context.account()
        post_cash = _get_account_value(post_account, "cash")
        post_positions = {}
        for pos in post_account.positions():
            local = gm_to_local(pos["symbol"])
            post_positions[local] = {
                "shares": pos["volume"],
                "vwap": round(float(pos["vwap"]), 4),
                "market_value": round(_get_pos_market_value(pos), 2),
            }
        context.rebalance_snapshots.append({
            "date": now_str,
            "sub_account": sub_k,
            "top_k": list(top_k_stocks),
            "global_weights": {k: round(v, 4) for k, v in global_w.items()},
            "pre_total": round(pre_total, 2),
            "post_cash": round(post_cash, 2),
            "post_positions": post_positions,
        })
    except Exception:
        pass

    # 持仓快照
    try:
        account = context.account()
        context.snapshot_cash = _get_account_value(account, "cash")
        pos_dict = {}
        for pos in account.positions():
            local = gm_to_local(pos["symbol"])
            pos_dict[local] = {"shares": pos["volume"], "cost": round(pos["vwap"] * pos["volume"], 2),
                               "buy_price": round(pos["vwap"], 4)}
        context.snapshot_positions = pos_dict
        post_total = _get_account_value(account, "cash") + sum(_get_pos_market_value(p) for p in account.positions())
        print(f"[账户] 调仓后: 总值={post_total:.2f} 现金={_get_account_value(account, 'cash'):.2f}")
    except Exception:
        pass

    # 每日快照
    try:
        account = context.account()
        cash = _get_account_value(account, "cash")
        pos_list = []
        for pos in account.positions():
            local = gm_to_local(pos["symbol"])
            pos_list.append({
                "stock": local,
                "shares": pos["volume"],
                "vwap": round(float(pos["vwap"]), 4),
                "market_value": round(_get_pos_market_value(pos), 2),
            })
        total = cash + sum(p["market_value"] for p in pos_list)
        context.daily_log.append({
            "date": now_str,
            "is_rebalance": now_str in context.sub_rebalance_dates[sub_k],
            "sub_account": sub_k,
            "cash": round(cash, 2),
            "positions": pos_list,
            "total_value": round(total, 2),
        })
    except Exception:
        pass


def on_order_status(context, order):
    if order["status"] == 3:
        side = "买入" if order["side"] == 1 else "卖出"
        local_stock = gm_to_local(order["symbol"])
        order_price = float(order["price"])
        fill_volume = int(order.get("filled_volume", order["volume"]))
        fill_vwap = order.get("filled_vwap", None)
        if fill_vwap is not None:
            fill_price = round(float(fill_vwap), 6)
        else:
            slippage = 0.001
            fill_price = round(order_price * (1 + slippage if order["side"] == 1 else (1 - slippage)), 6)
        trade_amount = round(fill_price * fill_volume, 2)
        fee = round(trade_amount * 0.0003, 2)
        trade = {
            "date": context.now.strftime('%Y-%m-%d'),
            "action": side,
            "stock": local_stock,
            "price": round(fill_price, 4),
            "shares": fill_volume,
            "amount": trade_amount,
        }
        context.executed_trades.append(trade)
        if order["side"] == 1:
            context.track_cash = round(context.track_cash - trade_amount - fee, 2)
        else:
            context.track_cash = round(context.track_cash + trade_amount - fee, 2)
        print(f"[成交] {order['symbol']}({local_stock}) {side} {fill_volume}股 "
              f"@ {order_price:.4f}(vwap={fill_price:.6f}) 现金→{context.track_cash:.2f}")


class _DatetimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        return super().default(obj)


def on_backtest_finished(context, indicator):
    print("\n" + "=" * 50)
    print("回测完成")

    ind_keys = []
    try:
        ind_keys = list(indicator.keys())
    except Exception:
        pass
    for k in ind_keys:
        try:
            print(f"  {k} = {indicator[k]}")
        except Exception:
            pass

    eq = context.daily_equity or []
    ret = 0.0
    max_dd = 0.0
    sharpe = 0.0
    annual_ret = 0.0
    annual_vol = 0.0
    daily_win_rate = 0.0
    calmar = 0.0

    if len(eq) >= 2:
        vals = [e["total_value"] for e in eq]
        first_val = vals[0]
        last_val = vals[-1]
        ret = (last_val - first_val) / first_val * 100
        n_days = len(vals)
        annual_ret = ((last_val / first_val) ** (252 / n_days) - 1) * 100
        daily_rets = [(vals[i] / vals[i-1] - 1) for i in range(1, len(vals))]
        avg_ret = sum(daily_rets) / len(daily_rets)
        var_ret = sum((r - avg_ret) ** 2 for r in daily_rets) / len(daily_rets)
        daily_std = math.sqrt(var_ret)
        annual_vol = daily_std * math.sqrt(252) * 100
        daily_win_rate = sum(1 for r in daily_rets if r > 0) / len(daily_rets)
        if daily_std > 0:
            sharpe = (avg_ret / daily_std) * math.sqrt(252)
        peak = vals[0]
        for v in vals:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100
            if dd > max_dd:
                max_dd = dd
        if max_dd > 0:
            calmar = annual_ret / max_dd if max_dd > 0 else 0

        print(f"\n  净值天数: {len(eq)}")
        print(f"  首日: {eq[0]['date']} = {first_val:.2f}")
        print(f"  末日: {eq[-1]['date']} = {last_val:.2f}")
        print(f"  累计收益: {ret:.2f}%")
        print(f"  年化收益: {annual_ret:.2f}%")
        print(f"  最大回撤: {max_dd:.2f}%")
        print(f"  夏普: {sharpe:.2f}")
        print(f"  日胜率: {daily_win_rate*100:.1f}%")
    else:
        print("  [警告] 净值曲线为空或数据不足")

    # 保存结果
    try:
        final_positions = getattr(context, 'snapshot_positions', {})
        final_cash = getattr(context, 'snapshot_cash', 0)

        snapshots = getattr(context, 'rebalance_snapshots', [])
        if snapshots:
            last_snap = snapshots[-1]
            post_positions = last_snap.get("post_positions", {})
            post_cash = last_snap.get("post_cash", final_cash)
            if post_positions and not any(v.get("shares", 0) <= 0 for v in post_positions.values()):
                final_positions = {k: {"shares": v["shares"], "cost": v.get("market_value", 0),
                                       "buy_price": v.get("vwap", 0)} for k, v in post_positions.items()}
                final_cash = post_cash

        sub_accounts_detail = []
        for k in range(SUB_ACCOUNTS):
            sub_accounts_detail.append({
                "sub_account": k,
                "rebalance_dates": sorted(context.sub_rebalance_dates[k]),
                "targets": context.sub_targets[k],
            })

        single_seq = {
            "equity_curve": context.daily_equity,
            "trades": context.executed_trades,
            "positions": final_positions,
            "cash": final_cash,
            "today_pnl": {},
            "skipped_trades": [],
            "metrics": {
                "strategy_return_pct": round(ret, 4),
                "annualized_return_pct": round(annual_ret, 4),
                "sharpe_ratio": round(sharpe, 4),
                "max_drawdown_pct": round(max_dd, 4),
                "daily_win_rate": round(daily_win_rate, 4),
                "calmar_ratio": round(calmar, 4),
                "annualized_volatility_pct": round(annual_vol, 4),
                "latest_value": round(last_val if len(eq) >= 2 else 100000, 2),
                "next_rebalance_date": "",
                "last_trade_prices": {},
            },
            "sub_accounts_detail": sub_accounts_detail,
        }

        state = {
            "sequences": {"juejin_5sub": single_seq},
            "rebalance_dates": sorted(set().union(*context.sub_rebalance_dates)),
            "start_date": START_DATE,
            "rebalance_days": REBALANCE_DAYS,
            "sub_accounts": SUB_ACCOUNTS,
            "position_pct": POSITION_PCT,
            "last_updated": str(datetime.now()),
            "trade_mode": getattr(context, 'trade_mode', 'open'),
        }

        with open(JUEJIN_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, cls=_DatetimeEncoder)
        print(f"\n[结果] {JUEJIN_STATE_PATH} 已保存")

        last_date = context.daily_equity[-1]["date"] if context.daily_equity else str(datetime.now().date())
        result = {
            "model_key": MODEL_KEY,
            "date": last_date,
            "trades": context.executed_trades,
            "equity_curve": context.daily_equity,
            "metrics": {
                "strategy_return_pct": round(ret, 4),
                "sharpe_ratio": round(indicator.get('sharp_ratio', 0), 4),
                "max_drawdown_pct": round(indicator.get('max_drawdown', 0) * 100, 4),
            },
            "config": {
                "sub_accounts": SUB_ACCOUNTS,
                "rebalance_days": REBALANCE_DAYS,
                "top_k": TOP_K,
            },
        }
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, cls=_DatetimeEncoder)
        print(f"[结果] 成交记录已保存: {RESULT_PATH}")
        print(f"  交易数: {len(context.executed_trades)}, 净值天数: {len(context.daily_equity)}")

    except Exception as e:
        print(f"\n[结果] 保存失败: {e}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="掘金回测策略 - 5错峰子账户")
    parser.add_argument("--trade-mode", type=str, default=None, choices=["open", "close"])
    args, _ = parser.parse_known_args()

    import sys
    this = sys.modules[__name__]
    if args.trade_mode:
        this.TRADE_MODE = args.trade_mode

    end_date = get_backtest_end_date()
    print(f"[策略] 回测区间: {START_DATE} ~ {end_date}")
    print(f"[策略] 子账户={SUB_ACCOUNTS} 每账户调仓间隔={REBALANCE_DAYS}天 Top-K={TOP_K}")

    run(strategy_id='strategy_id',
        filename=os.path.splitext(os.path.basename(__file__))[0],
        mode=MODE_BACKTEST,
        token='1b511135ca6034bc04c9f2eeb66b3a70cb08b831',
        backtest_start_time=f'{START_DATE} 08:00:00',
        backtest_end_time=f'{end_date} 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=100000,
        backtest_commission_ratio=0.0003,
        backtest_slippage_ratio=0.001,
        backtest_match_mode=1)
