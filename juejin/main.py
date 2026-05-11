# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import json
import os

'''
掘金回测策略：读取 backtest_state.json 的交易信号，执行调仓
使用 average 模型的交易记录
'''

STATE_PATH = r"C:\Users\xyl\Desktop\ETF\output\backtest_state.json"
MODEL_KEY = "search_itransformer_exp_54"

# ETF代码转换
def to_gm_symbol(stock_id):
    code, exchange = stock_id.split(".")
    exchange_map = {"XSHG": "SHSE", "XSHE": "SZSE"}
    return f"{exchange_map.get(exchange, exchange)}.{code}"

def load_trades_by_date():
    """从 backtest_state.json 加载调仓信号，按日期分组"""
    if not os.path.exists(STATE_PATH):
        print(f"[策略] 未找到 {STATE_PATH}")
        return {}
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)
    seqs = state.get("sequences", state)
    model = seqs.get(MODEL_KEY, {})
    trades = model.get("trades", [])

    by_date = {}
    for t in trades:
        d = t["date"]
        by_date.setdefault(d, []).append(t)
    return by_date


def init(context):
    context.trade_signals = load_trades_by_date()
    context.processed_dates = set()
    # 每天早上开盘执行
    schedule(schedule_func=algo, date_rule='1d', time_rule='09:31:00')


def algo(context):
    now_str = context.now.strftime('%Y-%m-%d')
    today_signals = context.trade_signals.get(now_str, [])

    if not today_signals:
        return

    if now_str in context.processed_dates:
        return
    context.processed_dates.add(now_str)

    print(f"\n{'='*50}")
    print(f"[调仓日] {now_str} 共 {len(today_signals)} 条信号")

    # 计算目标持仓
    buys = [t for t in today_signals if t["action"] == "买入"]
    sells = [t for t in today_signals if t["action"] == "卖出"]

    if buys:
        percent = 0.98 / len(buys)
    else:
        percent = 0

    # 1. 先卖出
    for t in sells:
        sym = to_gm_symbol(t["stock"])
        order_target_percent(symbol=sym, percent=0,
                             order_type=OrderType_Market,
                             position_side=PositionSide_Long)
        print(f"[策略] 卖出 {t['stock']}({sym})")

    # 2. 再买入
    for t in buys:
        sym = to_gm_symbol(t["stock"])
        order_target_percent(symbol=sym, percent=percent,
                             order_type=OrderType_Market,
                             position_side=PositionSide_Long)
        print(f"[策略] 买入 {t['stock']}({sym}) 目标权重 {percent:.2%}")

    print(f"[策略] 调仓完成")


def on_order_status(context, order):
    if order["status"] == 3:  # 全部成交
        side = "买入" if order["side"] == 1 else "卖出"
        print(f"[成交] {order['symbol']} {side} {order['volume']}股 @ {order['price']:.4f}")


def on_backtest_finished(context, indicator):
    print("\n" + "="*50)
    print("回测完成")
    print(f"累计收益: {indicator.get('cumulative_return_ratio', 0)*100:.2f}%")
    print(f"年化收益: {indicator.get('annual_return_ratio', 0)*100:.2f}%")
    print(f"最大回撤: {indicator.get('max_drawdown', 0)*100:.2f}%")
    print(f"夏普比率: {indicator.get('sharpe_ratio', 0):.2f}")


if __name__ == '__main__':
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
