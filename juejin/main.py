# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import json
import math
import os
from datetime import datetime

'''
掘金回测策略：从 predictions.json 读取模型预测分数，执行 Top-K 调仓
无需再依赖 backtest_state.json 的交易信号，策略自主决策。
'''

PREDICTIONS_PATH = r"C:\Users\xyl\Desktop\ETF\output\predictions.json"
STATE_PATH = r"C:\Users\xyl\Desktop\ETF\output\backtest_state.json"
JUEJIN_STATE_PATH = r"C:\Users\xyl\Desktop\ETF\output\juejin_state.json"
RESULT_PATH = r"C:\Users\xyl\Desktop\ETF\output\juejin_result.json"
YAML_PATH = r"C:\Users\xyl\Desktop\ETF\config.yaml"

MODEL_KEY = None
TOP_K = 3
REBALANCE_DAYS = 5
START_DATE = "2026-04-01"
TRADE_MODE = "open"
POSITION_PCT = 0.95
WEIGHT_STRATEGY = "equal"
STRATEGY_PARAMS = {}
CACHED_CSV = None
ETF_ROOT = r"C:\Users\xyl\Desktop\ETF"
DATA_CSV_PATH = os.path.join(ETF_ROOT, "etf_data", "etf_74.csv")


def load_config_from_yaml():
    """从 model_selection.yaml 读取掘金策略配置"""
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
        if data.get("master"):
            MODEL_KEY = str(data["master"])
        else:
            # 从第一个启用模型推导 model_key
            for m in data.get("models", []):
                if m.get("enabled", True):
                    exp_dir = m["dir"]
                    import re, os
                    parent = os.path.basename(os.path.dirname(exp_dir))
                    parent = re.sub(r'_\d+_\d+', '', parent)
                    name = os.path.basename(exp_dir)
                    MODEL_KEY = f"{parent}_{name}"
                    break
        print(f"[配置] 从 {YAML_PATH} 读取: top_k={TOP_K} rebalance_days={REBALANCE_DAYS} mode={TRADE_MODE} position_pct={POSITION_PCT} model_key={MODEL_KEY}")
    except Exception as e:
        print(f"[配置] 读取 yaml 失败，使用默认值: {e}")


# 模块级加载配置（GM 引擎以 import 方式导入本模块，不会执行 __main__）
load_config_from_yaml()

_BACKTEST_DATES = None  # 缓存预测文件中的全交易日历


def get_backtest_end_date():
    """从 predictions.json 读取最后一个回测日期，避免硬编码。"""
    if os.path.exists(PREDICTIONS_PATH):
        try:
            with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f).get("_meta", {})
            dates = meta.get("backtest_dates", [])
            if dates:
                return dates[-1]
        except Exception:
            pass
    # fallback: 取最近一个交易日
    import datetime
    today = datetime.date.today()
    while True:
        if today.weekday() < 5:
            return today.strftime("%Y-%m-%d")
        today -= datetime.timedelta(days=1)


def to_gm_symbol(stock_id):
    """510300.XSHG -> SHSE.510300"""
    code, exchange = stock_id.split(".")
    exchange_map = {"XSHG": "SHSE", "XSHE": "SZSE"}
    return f"{exchange_map.get(exchange, exchange)}.{code}"


def gm_to_local(symbol):
    """SHSE.510300 -> 510300.XSHG"""
    exchange_map = {"SHSE": "XSHG", "SZSE": "XSHE"}
    exchange, code = symbol.split(".")
    return f"{code}.{exchange_map.get(exchange, exchange)}"


def load_predictions_by_date():
    """从 predictions.json 加载模型预测分数和交易日历元信息。"""
    global _BACKTEST_DATES
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"[策略] 未找到 {PREDICTIONS_PATH}，请先运行 daily_eval --predictions-only")
        return {}
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        all_preds = json.load(f)

    # 提取元信息
    meta = all_preds.pop("_meta", {})
    _BACKTEST_DATES = meta.get("backtest_dates", [])
    if _BACKTEST_DATES:
        print(f"[策略] 加载交易日历: {len(_BACKTEST_DATES)} 天 ({_BACKTEST_DATES[0]} ~ {_BACKTEST_DATES[-1]})")

    model_preds = all_preds.get(MODEL_KEY, {})
    if not model_preds:
        print(f"[策略] predictions.json 中未找到模型 {MODEL_KEY}")
        return {}
    print(f"[策略] 加载 {MODEL_KEY} 预测，共 {len(model_preds)} 个交易日")
    return model_preds  # {date_str: [{rank, stock_id, score}, ...]}


def _get_account_value(account, key, default=0):
    """Helper: gm 3.0 DictLikeObject 需要 bracket 访问"""
    # 尝试指定 key，以及常见别名
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
    """Helper: 获取持仓市值"""
    try:
        return float(pos.get("market_value", 0))
    except Exception:
        try:
            return float(pos["market_value"])
        except Exception:
            return 0


def compute_rebalance_dates(backtest_dates, start_date, rebalance_days=5):
    """计算调仓日。与本地回测引擎一致：
    从 start_date 在交易日历中的位置起，每 rebalance_days 个交易日一次。
    """
    sorted_dates = sorted(set(backtest_dates))
    # 找到 start_date 在完整日历中的索引
    start_idx = 0
    for i, d in enumerate(sorted_dates):
        if d >= start_date:  # YYYY-MM-DD 字符串可直比较
            start_idx = i
            break
    rebalance_set = set()
    for i in range(start_idx, len(sorted_dates), rebalance_days):
        rebalance_set.add(sorted_dates[i])
    return rebalance_set


def init(context):
    context.predictions = load_predictions_by_date()
    if not context.predictions:
        raise ValueError("无预测数据，策略退出")

    context.end_date = _BACKTEST_DATES[-1] if _BACKTEST_DATES else get_backtest_end_date()

    # 计算调仓日（用全交易日历，与本地引擎对齐）
    cal = _BACKTEST_DATES or sorted(context.predictions.keys())
    context.rebalance_dates = compute_rebalance_dates(cal, START_DATE, REBALANCE_DAYS)
    print(f"[策略] 调仓日({len(context.rebalance_dates)}天): {sorted(context.rebalance_dates)}")

    # 调试：对比 CSV 日历 vs GM 日历
    try:
        from gm.api import get_trading_dates
        gm_dates_raw = get_trading_dates(exchange='SHSE', start_date=START_DATE, end_date=context.end_date)
        gm_date_strs = [str(d) for d in gm_dates_raw]
        csv_only = set(cal) - set(gm_date_strs)
        gm_only = set(gm_date_strs) - set(cal)
        if csv_only:
            print(f"  [日历差异] 仅 CSV 有: {sorted(csv_only)}")
        if gm_only:
            print(f"  [日历差异] 仅 GM 有: {sorted(gm_only)}")
        # 用 GM 日历算调仓日
        gm_rebalance = compute_rebalance_dates(gm_date_strs, START_DATE, REBALANCE_DAYS)
        print(f"  [GM日历] 调仓日({len(gm_rebalance)}天): {sorted(gm_rebalance)}")
        overlap = context.rebalance_dates & gm_rebalance
        diff1 = context.rebalance_dates - gm_rebalance
        diff2 = gm_rebalance - context.rebalance_dates
        print(f"  [对比] 重合: {len(overlap)}, CSV独有: {sorted(diff1)}, GM独有: {sorted(diff2)}")
    except Exception as e:
        print(f"  [日历] GM日历查询失败: {e}")

    context.trade_mode = TRADE_MODE
    context.calendar = sorted(cal)
    context.processed_dates = set()
    context.executed_trades = []
    context.daily_equity = [{"date": START_DATE, "total_value": 100000}]
    context.track_cash = 100000  # 手动追踪现金（account.cash 在此环境返回 0）
    context.snapshot_positions = {}
    context.snapshot_cash = 0
    context.rebalance_snapshots = []  # 每次调仓的快照
    context.daily_log = []  # 每日持仓+价格快照

    trade_time = '09:31:00' if TRADE_MODE == 'open' else '14:55:00'
    print(f"[策略] 交易模式: {'开盘交易' if TRADE_MODE == 'open' else '收盘交易'}, 执行时间: {trade_time}")
    schedule(schedule_func=algo, date_rule='1d', time_rule=trade_time)
    if TRADE_MODE == 'open':
        schedule(schedule_func=record_equity, date_rule='1d', time_rule='15:30:00')


def record_equity(context):
    """在盘中（15:30）记录每日总资产，用收盘价估值"""
    now_str = context.now.strftime('%Y-%m-%d')
    try:
        account = context.account()
        cash = getattr(context, 'track_cash', 100000)
        pos_val = sum(_get_pos_market_value(p) for p in account.positions())
        total_val = round(cash + pos_val, 2)
        already_recorded = any(e["date"] == now_str for e in context.daily_equity)
        if total_val > 0 and not already_recorded:
            context.daily_equity.append({
                "date": now_str,
                "total_value": total_val,
            })
        if len(context.daily_equity) % 5 == 2:
            print(f"  [净值] {now_str} 总值={total_val:.2f} 现金={cash:.2f} 持仓={pos_val:.2f}")
    except Exception as e:
        print(f"  [净值] 记录失败: {e}")


def algo(context):
    now_str = context.now.strftime('%Y-%m-%d')

    # 只在调仓日行动
    if now_str not in context.rebalance_dates:
        return
    if now_str in context.processed_dates:
        return
    context.processed_dates.add(now_str)

    # 根据交易模式确定预测日期
    if context.trade_mode == "open":
        # 用全量预测日期（含 seed date 2026-03-31）做前一交易日查找
        cal = sorted(context.predictions.keys())
        idx = cal.index(now_str) if now_str in cal else -1
        pred_date = cal[idx - 1] if idx > 0 else now_str
    else:
        pred_date = now_str
    if pred_date != now_str:
        print(f"[策略] {now_str}: 开盘交易模式，使用 {pred_date} 的预测信号")
    else:
        print(f"[策略] {now_str}: 使用当日 {pred_date} 的预测信号")

    today_preds = context.predictions.get(pred_date, [])
    if not today_preds:
        print(f"[策略] {pred_date} 无预测数据，跳过")
        return

    # 调仓前账户信息
    pre_account = context.account()
    pre_cash = _get_account_value(pre_account, "cash")
    pre_pos_val = sum(_get_pos_market_value(p) for p in pre_account.positions())
    pre_total = pre_cash + pre_pos_val
    print(f"\n{'='*50}")
    print(f"[调仓日] {now_str}")
    print(f"[账户] 调仓前: 总值={pre_total:.2f} 现金={pre_cash:.2f} 持仓={pre_pos_val:.2f}")

    # Top-K 目标持仓（用有序列表保持平台/本地一致性）
    top_k_list = [to_gm_symbol(p["stock_id"]) for p in today_preds[:TOP_K]]
    top_k_symbols = set(top_k_list)
    top_k_stocks = {p["stock_id"] for p in today_preds[:TOP_K]}
    print(f"[策略] Top-{TOP_K}: {', '.join(top_k_stocks)}")

    # 保存调仓前持仓快照（供日报收盘模式使用）
    try:
        pre_positions = {}
        for pos in context.account().positions():
            local = gm_to_local(pos["symbol"])
            pre_positions[local] = {
                "shares": pos["volume"],
                "cost": round(pos["vwap"] * pos["volume"], 2),
            }
        context.pre_rebalance_positions = pre_positions
    except Exception:
        context.pre_rebalance_positions = {}

    # 卖出现在持仓但不在 Top-K 的
    current_pos = context.account().positions()
    for pos in current_pos:
        local = gm_to_local(pos["symbol"])
        if local not in top_k_stocks:
            order_target_percent(symbol=pos["symbol"], percent=0,
                                 order_type=OrderType_Market,
                                 position_side=PositionSide_Long)
            print(f"[策略] 卖出 {local}({pos['symbol']})")

    # 买入 Top-K（按加权策略分配仓位）
    import sys
    sys.path.insert(0, os.path.join(ETF_ROOT, "code", "src"))
    from backtest import compute_weights, compute_volatility

    _params = dict(STRATEGY_PARAMS)
    if WEIGHT_STRATEGY in ("risk_parity", "score_risk", "score_risk_v1"):
        import pandas as pd
        global CACHED_CSV
        if CACHED_CSV is None and os.path.exists(DATA_CSV_PATH):
            CACHED_CSV = pd.read_csv(DATA_CSV_PATH, dtype={"股票代码": str})
        if CACHED_CSV is not None:
            top_ids = [p["stock_id"] for p in today_preds[:TOP_K]]
            _params["vol_dict"] = compute_volatility(
                CACHED_CSV, top_ids, str(context.now.date()),
                _params.get("vol_window", 20),
            )
    _weights = compute_weights(today_preds, TOP_K, WEIGHT_STRATEGY, _params) if top_k_list else {}
    if top_k_list:
        for sym in top_k_list:
            local = gm_to_local(sym)
            w = _weights.get(local, 0)
            percent = POSITION_PCT * w
            if percent <= 0:
                continue
            order_target_percent(symbol=sym, percent=percent,
                                 order_type=OrderType_Market,
                                 position_side=PositionSide_Long)
            print(f"[策略] 买入 {local}({sym}) 目标权重 {percent:.2%}")

    print(f"[策略] 调仓完成")

    # 记录调仓快照（供 debug 用）
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
            "top_k": list(top_k_stocks),
            "top_k_scores": [{"stock_id": p["stock_id"], "score": p["score"]} for p in today_preds[:TOP_K]],
            "all_scores": [{"stock_id": p["stock_id"], "score": p["score"], "rank": p["rank"]} for p in today_preds[:20]],
            "target_weights": {local: round(w, 4) for local, w in _weights.items()},
            "target_percents": {local: round(POSITION_PCT * _weights.get(local, 0), 4) for local in top_k_stocks},
            "pre_total": round(pre_total, 2),
            "post_cash": round(post_cash, 2),
            "post_positions": post_positions,
        })
    except Exception:
        pass

    # 保存持仓快照（供 on_backtest_finished 使用）
    try:
        account = context.account()
        context.snapshot_cash = _get_account_value(account, "cash")
        pos_dict = {}
        for pos in account.positions():
            local = gm_to_local(pos["symbol"])
            pos_dict[local] = {"shares": pos["volume"], "cost": round(pos["vwap"] * pos["volume"], 2), "buy_price": round(pos["vwap"], 4)}
        context.snapshot_positions = pos_dict
        post_total = _get_account_value(account, "cash") + sum(_get_pos_market_value(p) for p in account.positions())
        print(f"[账户] 调仓后: 总值={post_total:.2f} 现金={_get_account_value(account, 'cash'):.2f}")
    except Exception:
        pass

    # 每日持仓快照（含价格）
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
            "is_rebalance": now_str in context.rebalance_dates,
            "cash": round(cash, 2),
            "positions": pos_list,
            "total_value": round(total, 2),
        })
    except Exception:
        pass


def on_order_status(context, order):
    if order["status"] == 3:  # 全部成交
        side = "买入" if order["side"] == 1 else "卖出"
        local_stock = gm_to_local(order["symbol"])
        order_price = float(order["price"])
        fill_volume = int(order.get("filled_volume", order["volume"]))
        fill_vwap = order.get("filled_vwap", None)
        if fill_vwap is not None:
            fill_price = round(float(fill_vwap), 6)
        else:
            # order["price"] 是申报价，实际成交价 = 申报价 * (1 ± 滑点 0.1%)
            slippage = 0.001
            fill_price = round(order_price * (1 + slippage if order["side"] == 1 else (1 - slippage)), 6)
        trade_amount = round(fill_price * fill_volume, 2)
        fee = round(trade_amount * 0.0003, 2)  # 佣金 0.03%
        trade = {
            "date": context.now.strftime('%Y-%m-%d'),
            "action": side,
            "stock": local_stock,
            "price": round(fill_price, 4),
            "shares": fill_volume,
            "amount": trade_amount,
        }
        context.executed_trades.append(trade)
        # 追踪现金余额
        if order["side"] == 1:  # 买入
            context.track_cash = round(context.track_cash - trade_amount - fee, 2)
        else:  # 卖出
            context.track_cash = round(context.track_cash + trade_amount - fee, 2)
        vwap_src = "filled_vwap" if fill_vwap is not None else "slippage_formula"
        print(f"[成交] {order['symbol']}({local_stock}) {side} {fill_volume}股 @ {order_price:.4f}(vwap={fill_price:.6f}) 现金→{context.track_cash:.2f} [{vwap_src}]")
        # 调试：打印 order 所有 keys（仅首日）
        if len(context.executed_trades) <= 6:
            print(f"  [order keys] {list(order.keys())}")


class _DatetimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        return super().default(obj)


def on_backtest_finished(context, indicator):
    print("\n" + "=" * 50)
    print("回测完成")

    # 调试：打印所有 indicator 键
    ind_keys = []
    try:
        ind_keys = list(indicator.keys())
    except Exception:
        pass
    print(f"  indicator keys: {ind_keys}")
    for k in ind_keys:
        try:
            print(f"    {k} = {indicator[k]}")
        except Exception:
            pass

    # 从 equity curve 重算指标（GM indicator 数据不可靠）
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
        dates = [e["date"] for e in eq]
        first_val = vals[0]
        last_val = vals[-1]
        ret = (last_val - first_val) / first_val * 100

        # 年化
        n_days = len(vals)
        annual_ret = ((last_val / first_val) ** (252 / n_days) - 1) * 100

        # 日收益率序列
        daily_rets = [(vals[i] / vals[i-1] - 1) for i in range(1, len(vals))]
        avg_ret = sum(daily_rets) / len(daily_rets)
        var_ret = sum((r - avg_ret) ** 2 for r in daily_rets) / len(daily_rets)
        daily_std = math.sqrt(var_ret)
        annual_vol = daily_std * math.sqrt(252) * 100
        daily_win_rate = sum(1 for r in daily_rets if r > 0) / len(daily_rets)

        # 夏普 (假设无风险=0)
        if daily_std > 0:
            sharpe = (avg_ret / daily_std) * math.sqrt(252)

        # 最大回撤
        peak = vals[0]
        for v in vals:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # 卡玛
        if max_dd > 0:
            calmar = annual_ret / max_dd if max_dd > 0 else 0

        # 用 indicator 的 pnl_ratio 校准 equity curve
        # 手动现金追踪有累积舍入误差，用 indicator 的精确最终值做缩放校正
        # 只缩放增量收益部分，不缩放本金
        ind_ratio = float(indicator.get("pnl_ratio", 0))
        if ind_ratio > 0:
            base = 100000
            correct_final = base * (1 + ind_ratio)
            my_final = vals[-1]
            my_gain = my_final - base
            if my_gain > 0 and abs(correct_final - my_final) / correct_final > 0.001:
                gain_scale = (correct_final - base) / my_gain
                print(f"\n  [校准] indicator pnl_ratio={ind_ratio*100:.2f}% 目标终值={correct_final:.2f}")
                print(f"  [校准] 原始终值={my_final:.2f} gain_scale={gain_scale:.6f}")
                for e in context.daily_equity:
                    e["total_value"] = round(base + (e["total_value"] - base) * gain_scale, 2)
                # 重算指标
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
                max_dd = 0.0
                for v in vals:
                    if v > peak:
                        peak = v
                    dd = (peak - v) / peak * 100
                    if dd > max_dd:
                        max_dd = dd
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

    # 保存完整结果到 backtest_state.json 供日报使用
    try:
        # 从 algo 快照获取最终持仓（on_backtest_finished 中 account API 可能不可用）
        final_positions = getattr(context, 'snapshot_positions', {})
        final_cash = getattr(context, 'snapshot_cash', 0)

        equity_name = MODEL_KEY.replace("search_", "").replace("_exp_", " ") 

        # 从最新 rebalance snapshot 取准确持仓（若有）
        snapshots = getattr(context, 'rebalance_snapshots', [])
        if snapshots:
            last_snap = snapshots[-1]
            post_positions = last_snap.get("post_positions", {})
            post_cash = last_snap.get("post_cash", final_cash)
            # 用快照覆盖（快照是在 order 执行后记录的，更准确）
            if post_positions and not any(v.get("shares", 0) <= 0 for v in post_positions.values()):
                final_positions = {k: {"shares": v["shares"], "cost": v.get("market_value", 0), "buy_price": v.get("vwap", 0)} for k, v in post_positions.items()}
                final_cash = post_cash

        # 构建统一格式的 state
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
        }

        pre_rb_pos = getattr(context, 'pre_rebalance_positions', {})
        if pre_rb_pos:
            single_seq["pre_rebalance_positions"] = pre_rb_pos

        preds_by_date = getattr(context, 'predictions', {})
        if preds_by_date:
            ph = [{"date": d, "predictions": preds_by_date[d]} for d in sorted(preds_by_date.keys())]
            single_seq["predictions_history"] = ph

        state = {
            "sequences": {"juejin": single_seq},
            "rebalance_dates": sorted(context.rebalance_dates),
            "start_date": START_DATE,
            "rebalance_days": REBALANCE_DAYS,
            "position_pct": POSITION_PCT,
            "last_updated": str(datetime.now()),
            "trade_mode": getattr(context, 'trade_mode', 'open'),
        }

        with open(JUEJIN_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, cls=_DatetimeEncoder)
        print(f"\n[结果] {JUEJIN_STATE_PATH} 已保存")

        # 同时保存详细成交记录
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
        }
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, cls=_DatetimeEncoder)
        print(f"[结果] 成交记录已保存: {RESULT_PATH}")
        print(f"  交易数: {len(context.executed_trades)}, 净值天数: {len(context.daily_equity)}")

        # 保存详细 debug JSON（用于对比本地回测差异）
        debug_path = os.path.join(os.path.dirname(STATE_PATH), "juejin_debug.json")
        try:
            # 将每日净值转为 dict 方便按日期查找
            equity_by_date = {e["date"]: e["total_value"] for e in context.daily_equity}

            # 按调仓日分组计算每段收益
            rebalance_dates_sorted = sorted(context.rebalance_dates)
            period_returns = []
            prev_val = None
            prev_rd = None
            for rd in rebalance_dates_sorted:
                val = equity_by_date.get(rd)
                if val is None:
                    # 用次日净值
                    for e in context.daily_equity:
                        if e["date"] > rd and e["total_value"] > 0:
                            val = e["total_value"]
                            break
                if val is not None and prev_val is not None and prev_rd is not None:
                    period_returns.append({
                        "period": f"{prev_rd} → {rd}",
                        "start_value": round(prev_val, 2),
                        "end_value": round(val, 2),
                        "return_pct": round((val / prev_val - 1) * 100, 2),
                    })
                if val is not None:
                    prev_val = val
                    prev_rd = rd

            debug = {
                "model_key": MODEL_KEY,
                "config": {
                    "start_date": START_DATE,
                    "rebalance_days": REBALANCE_DAYS,
                    "top_k": TOP_K,
                    "initial_cash": 100000,
                    "slippage_ratio": 0.001,
                    "commission_ratio": 0.0003,
                    "backtest_end": context.end_date,
                    "weight_strategy": WEIGHT_STRATEGY,
                    "trade_mode": getattr(context, 'trade_mode', 'open'),
                    "position_pct": POSITION_PCT,
                },
                "rebalance_dates": rebalance_dates_sorted,
                "calendar_diff": {},
                "rebalance_snapshots": getattr(context, "rebalance_snapshots", []),
                "daily_log": getattr(context, "daily_log", []),
                "trades": context.executed_trades,
                "equity_curve": context.daily_equity,
                "period_returns": period_returns,
                "final_positions": final_positions,
                "final_cash": round(final_cash, 2),
                "final_value": round(last_val if len(eq) >= 2 else 100000, 2),
                "metrics": {
                    "pnl_ratio_pct": round(ret, 4),
                    "pnl_ratio_annual_pct": round(indicator.get('pnl_ratio_annual', 0) * 100, 4),
                    "sharp_ratio": round(indicator.get('sharp_ratio', 0), 4),
                    "max_drawdown_pct": round(max_dd, 4),
                    "win_ratio": round(indicator.get('win_ratio', 0), 4),
                    "calmar_ratio": round(indicator.get('calmar_ratio', 0), 4),
                },
                "gm_indicator_log": {},
            }
            # 保存 indicator 所有字段（平台vs本地关键对比数据）
            try:
                gm_log = {}
                for k in ind_keys:
                    try:
                        v = indicator[k]
                        if hasattr(v, 'strftime'):
                            v = str(v)
                        gm_log[k] = v
                    except Exception:
                        pass
                debug["gm_indicator_log"] = gm_log
            except Exception:
                pass
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(debug, f, ensure_ascii=False, indent=2, cls=_DatetimeEncoder)
            print(f"[结果] debug 详情已保存: {debug_path}")
        except Exception as e:
            print(f"[结果] debug 保存失败: {e}")

    except Exception as e:
        print(f"\n[结果] 保存失败: {e}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="掘金回测策略")
    parser.add_argument("--trade-mode", type=str, default=None, choices=["open", "close"],
                        help="交易模式: open（开盘交易）或 close（收盘交易），默认从 model_selection.yaml 读取")
    args, _ = parser.parse_known_args()

    # 命令行参数覆盖 yaml
    import sys
    this = sys.modules[__name__]
    if args.trade_mode:
        this.TRADE_MODE = args.trade_mode

    end_date = get_backtest_end_date()
    print(f"[策略] 回测区间: {START_DATE} ~ {end_date}")

    run(strategy_id='strategy_id',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='1b511135ca6034bc04c9f2eeb66b3a70cb08b831',
        backtest_start_time=f'{START_DATE} 08:00:00',
        backtest_end_time=f'{end_date} 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=100000,
        backtest_commission_ratio=0.0003,
        backtest_slippage_ratio=0.001,
        backtest_match_mode=1)
