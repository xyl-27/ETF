# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import json
import os
from pathlib import Path

'''
掘金模拟盘策略
读取日报系统输出的 portfolio.json，同步持仓到掘金模拟盘
ETF标的与日报系统完全一致
'''

# 日报系统输出路径（根据实际部署调整）
PORTFOLIO_PATH = r"C:\Users\xyl\Desktop\ETF\output\portfolio.json"
REPORT_PATH = r"C:\Users\xyl\Desktop\ETF\output\latest_report.json"

# ETF代码转换：XSHG/XSHE -> 掘金格式
def to_gm_symbol(stock_id):
    """515050.XSHG -> SHSE.515050, 159949.XSHE -> SZSE.159949"""
    code, exchange = stock_id.split(".")
    exchange_map = {"XSHG": "SHSE", "XSHE": "SZSE"}
    gm_exchange = exchange_map.get(exchange, exchange)
    return f"{gm_exchange}.{code}"

def to_local_symbol(gm_symbol):
    """SHSE.515050 -> 515050.XSHG"""
    exchange, code = gm_symbol.split(".")
    reverse_map = {"SHSE": "XSHG", "SZSE": "XSHE"}
    local_exchange = reverse_map.get(exchange, exchange)
    return f"{code}.{local_exchange}"

def read_portfolio():
    """读取日报系统最新的持仓信号"""
    if not os.path.exists(PORTFOLIO_PATH):
        print(f"[策略] 未找到 portfolio.json: {PORTFOLIO_PATH}")
        return None
    try:
        with open(PORTFOLIO_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"[策略] 读取 portfolio.json 失败: {e}")
        return None

def read_report_trades():
    """读取日报最新的调仓记录"""
    if not os.path.exists(REPORT_PATH):
        return None
    try:
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("all_today_trades", [])
    except Exception as e:
        print(f"[策略] 读取 latest_report.json 失败: {e}")
        return None


def init(context):
    context.pending_sell = []  # 待卖出列表
    context.pending_buy = []   # 待买入列表
    context.target_percent = {}  # {symbol: target_percent}

    # 每日 9:35 执行（给日报系统留出运行时间）
    schedule(schedule_func=algo, date_rule='1d', time_rule='09:35:00')


def algo(context):
    now_str = context.now.strftime('%Y-%m-%d')
    print(f"\n{'='*50}")
    print(f"[策略] {now_str} 开始执行")

    # 读取日报系统的持仓信号
    portfolio = read_portfolio()
    if portfolio is None:
        print("[策略] 无持仓信号，跳过")
        return

    predict_date = portfolio.get("predict_date", "")
    holdings = portfolio.get("holdings", [])
    total_value = portfolio.get("total_value", 0)

    print(f"[策略] 日报预测日期: {predict_date}")
    print(f"[策略] 目标持仓数: {len(holdings)}")

    # 转换目标持仓为掘金格式
    target_symbols = set()
    for h in holdings:
        sid = h.get("stock_id", "")
        if sid:
            gm_sym = to_gm_symbol(sid)
            target_symbols.add(gm_sym)

    if not target_symbols:
        print("[策略] 目标持仓为空，清空所有仓位")
        for pos in get_position():
            order_target_percent(symbol=pos["symbol"], percent=0,
                                 order_type=OrderType_Market,
                                 position_side=PositionSide_Long)
        return

    # 分配权重
    percent = 0.98 / len(target_symbols)

    # 获取当前持仓
    current_positions = {p["symbol"]: p for p in get_position()}
    current_symbols = set(current_positions.keys())

    # 需要卖出的：当前有持仓但不在目标中的
    to_sell = current_symbols - target_symbols
    # 需要买入的：目标中有但当前没有的
    to_buy = target_symbols - current_symbols
    # 需要调整的：两者都有但权重不同的
    to_adjust = current_symbols & target_symbols

    print(f"[策略] 当前持仓: {len(current_symbols)} 个")
    print(f"[策略] 需要卖出: {len(to_sell)} 个")
    print(f"[策略] 需要买入: {len(to_buy)} 个")

    # 1. 卖出不在目标中的
    for sym in to_sell:
        order_target_percent(symbol=sym, percent=0,
                             order_type=OrderType_Market,
                             position_side=PositionSide_Long)
        print(f"[策略] 卖出: {sym}")

    # 2. 买入目标中的
    for sym in to_buy:
        order_target_percent(symbol=sym, percent=percent,
                             order_type=OrderType_Market,
                             position_side=PositionSide_Long)
        print(f"[策略] 买入: {sym}")

    # 3. 调整已有持仓的权重
    for sym in to_adjust:
        cur = current_positions[sym]
        cur_pct = cur["target_percent"]
        if abs(cur_pct - percent) > 0.01:
            order_target_percent(symbol=sym, percent=percent,
                                 order_type=OrderType_Market,
                                 position_side=PositionSide_Long)

    print(f"[策略] 执行完成")


def on_order_status(context, order):
    if order["status"] == 3:  # 全部成交
        side = "买入" if order["side"] == 1 else "卖出"
        print(f"[成交] {context.now.strftime('%Y-%m-%d %H:%M:%S')} "
              f"{order['symbol']} {side} {order['volume']}股 @ {order['price']:.4f}")


def on_backtest_finished(context, indicator):
    print("\n" + "="*50)
    print("回测完成")


if __name__ == '__main__':
    run(strategy_id='strategy_id',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='{{token}}',
        backtest_start_time='2026-04-01 08:00:00',
        backtest_end_time='2026-05-11 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=100000,
        backtest_commission_ratio=0.0003,
        backtest_slippage_ratio=0.001,
        backtest_match_mode=1)
