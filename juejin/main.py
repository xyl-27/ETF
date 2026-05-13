# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import json
import os
from datetime import datetime

'''
掘金回测策略：从 predictions.json 读取模型预测分数，执行 Top-K 调仓
无需再依赖 backtest_state.json 的交易信号，策略自主决策。
'''

PREDICTIONS_PATH = r"C:\Users\xyl\Desktop\ETF\output\predictions.json"
STATE_PATH = r"C:\Users\xyl\Desktop\ETF\output\backtest_state.json"
RESULT_PATH = r"C:\Users\xyl\Desktop\ETF\output\juejin_result.json"

MODEL_KEY = "search_itransformer_exp_54"
TOP_K = 3
REBALANCE_DAYS = 5
START_DATE = "2026-04-01"
TRADE_MODE = "open"  # "open"（开盘交易，用前日特征）或 "close"（收盘交易，用当日特征）

_BACKTEST_DATES = None  # 缓存预测文件中的全交易日历


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
    try:
        val = account[key]
        return float(val) if val is not None else default
    except Exception:
        try:
            return float(getattr(account, key, default))
        except Exception:
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

    # 计算调仓日（用全交易日历，与本地引擎对齐）
    cal = _BACKTEST_DATES or sorted(context.predictions.keys())
    context.rebalance_dates = compute_rebalance_dates(cal, START_DATE, REBALANCE_DAYS)
    print(f"[策略] 调仓日({len(context.rebalance_dates)}天): {sorted(context.rebalance_dates)}")

    # 调试：对比 CSV 日历 vs GM 日历
    try:
        from gm.api import get_trading_dates
        gm_dates_raw = get_trading_dates(exchange='SHSE', start_date=START_DATE, end_date='2026-05-12')
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
    context.daily_equity = []
    context.snapshot_positions = {}
    context.snapshot_cash = 0
    context.rebalance_snapshots = []  # 每次调仓的快照

    trade_time = '09:31:00' if TRADE_MODE == 'open' else '14:55:00'
    print(f"[策略] 交易模式: {'开盘交易' if TRADE_MODE == 'open' else '收盘交易'}, 执行时间: {trade_time}")
    schedule(schedule_func=algo, date_rule='1d', time_rule=trade_time)


def algo(context):
    now_str = context.now.strftime('%Y-%m-%d')

    # 记录每日净值（跳过首日到账前的 0 值）
    try:
        account = context.account()
        cash = _get_account_value(account, "cash")
        pos_val = 0
        for pos in account.positions():
            pos_val += _get_pos_market_value(pos)
        total_val = round(cash + pos_val, 2)
        if total_val > 0 or len(context.daily_equity) > 0:
            # 有正净值才记录（跳过首日到账前的 0），或已有记录可保留 0
            context.daily_equity.append({
                "date": now_str,
                "total_value": total_val,
            })
        # 每日打印一次净值（非调仓日简要）
        if now_str in context.rebalance_dates:
            pass  # 调仓日下面会有详细输出
        elif len(context.daily_equity) % 5 == 1:
            print(f"  [净值] {now_str} 总值={total_val:.2f} 现金={cash:.2f} 持仓={pos_val:.2f}")
    except Exception as e:
        print(f"  [净值] 记录失败: {e}")

    # 只在调仓日行动
    if now_str not in context.rebalance_dates:
        return
    if now_str in context.processed_dates:
        return
    context.processed_dates.add(now_str)

    # 根据交易模式确定预测日期
    if context.trade_mode == "open":
        cal = context.calendar
        idx = cal.index(now_str) if now_str in cal else -1
        pred_date = cal[idx - 1] if idx > 0 else now_str
    else:
        pred_date = now_str
    if pred_date != now_str:
        print(f"[策略] {now_str}: 开盘交易模式，使用 {pred_date} 的预测信号")

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

    # Top-K 目标持仓
    top_k_symbols = {to_gm_symbol(p["stock_id"]) for p in today_preds[:TOP_K]}
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

    # 买入 Top-K（等权）
    if top_k_symbols:
        percent = 0.98 / len(top_k_symbols)
        for sym in top_k_symbols:
            order_target_percent(symbol=sym, percent=percent,
                                 order_type=OrderType_Market,
                                 position_side=PositionSide_Long)
            print(f"[策略] 买入 {gm_to_local(sym)}({sym}) 目标权重 {percent:.2%}")

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


def on_order_status(context, order):
    if order["status"] == 3:  # 全部成交
        side = "买入" if order["side"] == 1 else "卖出"
        local_stock = gm_to_local(order["symbol"])
        fill_price = order["price"]
        fill_volume = order["volume"]
        trade = {
            "date": context.now.strftime('%Y-%m-%d'),
            "action": side,
            "stock": local_stock,
            "price": round(fill_price, 4),
            "shares": fill_volume,
            "amount": round(fill_price * fill_volume, 2),
        }
        context.executed_trades.append(trade)
        print(f"[成交] {order['symbol']}({local_stock}) {side} {order['volume']}股 @ {order['price']:.4f}")


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

    # 优先用 indicator 的收益率（gm 引擎官方值）
    ret = indicator.get('pnl_ratio', 0) * 100
    max_dd = indicator.get('max_drawdown', 0) * 100
    sharpe = indicator.get('sharp_ratio', 0)

    # 净值曲线摘要
    if context.daily_equity:
        non_zero = [e for e in context.daily_equity if e["total_value"] > 0]
        print(f"  净值天数: {len(context.daily_equity)} (有效: {len(non_zero)})")
        if len(non_zero) >= 2:
            first_val = non_zero[0]["total_value"]
            last_val = non_zero[-1]["total_value"]
            ec_ret = (last_val / first_val - 1) * 100
            print(f"  首日有效净值: {first_val} ({non_zero[0]['date']})")
            print(f"  末日净值: {last_val} ({non_zero[-1]['date']})")
            print(f"  净值累计收益: {ec_ret:.2f}%")
    else:
        print(f"  [警告] 净值曲线为空")

    print(f"累计收益: {ret:.2f}%")
    print(f"最大回撤: {max_dd:.2f}%")
    print(f"夏普比率: {sharpe:.2f}")

    # 保存完整结果到 backtest_state.json 供日报使用
    try:
        # 从 algo 快照获取最终持仓（on_backtest_finished 中 account API 可能不可用）
        final_positions = getattr(context, 'snapshot_positions', {})
        final_cash = getattr(context, 'snapshot_cash', 0)

        latest_value = context.daily_equity[-1]["total_value"] if context.daily_equity else 100000
        equity_name = MODEL_KEY.replace("search_", "").replace("_exp_", " ") 

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
                "annualized_return_pct": round(indicator.get('pnl_ratio_annual', 0) * 100, 4),
                "sharpe_ratio": round(indicator.get('sharp_ratio', 0), 4),
                "max_drawdown_pct": round(indicator.get('max_drawdown', 0) * 100, 4),
                "latest_value": round(latest_value, 2),
                "next_rebalance_date": "",
                "last_trade_prices": {},
            },
        }

        pre_rb_pos = getattr(context, 'pre_rebalance_positions', {})
        if pre_rb_pos:
            single_seq["pre_rebalance_positions"] = pre_rb_pos

        state = {}
        if os.path.exists(STATE_PATH):
            try:
                with open(STATE_PATH, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    state = existing
            except Exception:
                pass
        state.setdefault("sequences", {})[MODEL_KEY] = single_seq
        state["last_updated"] = str(datetime.now())
        state["trade_mode"] = getattr(context, 'trade_mode', 'open')

        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, cls=_DatetimeEncoder)
        print(f"\n[结果] backtest_state.json 已保存 (序列: {list(state['sequences'].keys())})")

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
                    "backtest_end": "2026-05-12",
                },
                "rebalance_dates": rebalance_dates_sorted,
                "calendar_diff": {},
                "rebalance_snapshots": getattr(context, "rebalance_snapshots", []),
                "trades": context.executed_trades,
                "equity_curve": context.daily_equity,
                "period_returns": period_returns,
                "final_positions": final_positions,
                "final_cash": round(final_cash, 2),
                "final_value": round(latest_value, 2),
                "metrics": {
                    "pnl_ratio_pct": round(ret, 4),
                    "pnl_ratio_annual_pct": round(indicator.get('pnl_ratio_annual', 0) * 100, 4),
                    "sharp_ratio": round(indicator.get('sharp_ratio', 0), 4),
                    "max_drawdown_pct": round(max_dd, 4),
                    "win_ratio": round(indicator.get('win_ratio', 0), 4),
                    "calmar_ratio": round(indicator.get('calmar_ratio', 0), 4),
                },
            }
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
    parser.add_argument("--trade-mode", type=str, default=TRADE_MODE, choices=["open", "close"],
                        help="交易模式: open（开盘交易）或 close（收盘交易）")
    args, _ = parser.parse_known_args()

    # 更新全局 TRADE_MODE
    import sys
    this = sys.modules[__name__]
    this.TRADE_MODE = args.trade_mode

    run(strategy_id='strategy_id',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='{{token}}',
        backtest_start_time='2026-04-01 08:00:00',
        backtest_end_time='2026-05-12 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=100000,
        backtest_commission_ratio=0.0003,
        backtest_slippage_ratio=0.001,
        backtest_match_mode=1)
