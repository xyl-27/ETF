# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import numpy as np
import math
import csv
from datetime import timedelta

"""
五福闹新春 v3.3 动量多因子 — 掘金版
ETF池: etf_list_before_2022_74.csv (74只)
"""

# ==================== 路径 ====================
ETF_LIST_PATH = r"C:\Users\xyl\Desktop\ETF\etf_data\etf_list_before_2022_74.csv"

# ==================== 策略参数 ====================
HOLDINGS_NUM = 3
DEFENSIVE_ETF_CODE = "511880.XSHG"
SAFE_HAVEN_CODE = "511660.XSHG"
LOOKBACK_DAYS = 25
MIN_SCORE = 0
MAX_SCORE = 5
SCORE_THRESHOLD_RATIO = 0.9
ENABLE_R2 = True
R2_THRESHOLD = 0.4
ENABLE_VOLUME = True
VOLUME_LOOKBACK = 5
VOLUME_THRESHOLD = 1.0
ENABLE_LOSS = True
LOSS_RATIO = 0.97
USE_FIXED_STOP_LOSS = True
STOP_LOSS_PCT = 0.95
ENABLE_COOLDOWN = False       # 原JQ: sell_cooldown_enabled = False
COOLDOWN_DAYS = 3

# ==================== 全局 ====================
_etf_pool = []
_etf_local_pool = []
_etf_names = {}
_cooldown_end = None


def to_gm(sym):
    c, e = sym.split(".")
    return f"{'SHSE' if e == 'XSHG' else 'SZSE'}.{c}"


def to_local(sym):
    e, c = sym.split(".")
    return f"{c}.{'XSHG' if e == 'SHSE' else 'XSHE'}"


def load_pool():
    global _etf_pool, _etf_local_pool, _etf_names
    with open(ETF_LIST_PATH, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            c = r["代码"].strip()
            _etf_local_pool.append(c)
            _etf_pool.append(to_gm(c))
            _etf_names[c] = r["名称"].strip()
    print(f"[ETF] {len(_etf_pool)}只")


def get_data(sym, end_date, n):
    """用 history 取 n 天日线（end_date 为截止日期，不含当日）"""
    try:
        import pandas as pd
        end = pd.Timestamp(end_date)
        start = end - pd.Timedelta(days=n * 2)
        df = history(sym, "1d", start_time=start, end_time=end,
                     fields="close,volume", skip_suspended=True,
                     fill_missing="Last", adjust=0, df=True)
        if df is not None and not df.empty:
            v = df["close"].values[-n:]
            vv = df["volume"].values[-n:]
            return v, vv
    except Exception:
        pass
    return None


def calc_all(context):
    date_str = context.now.strftime("%Y-%m-%d")
    need = LOOKBACK_DAYS + 20
    out = []
    skip = 0
    for sym in _etf_pool:
        d = get_data(sym, date_str, need + 5)
        if d is None:
            skip += 1
            continue
        close, vol = d
        if len(close) < need + 1:
            skip += 1
            continue

        # 获取今日实时价格（JQ兼容: attribute_history + current_data.last_price）
        today_px = None
        try:
            bar = history_n(sym, "60m", 1, fields="close",
                            skip_suspended=True, fill_missing="Last",
                            adjust=0, df=True)
            if bar is not None and not bar.empty:
                today_px = bar["close"].values[-1]
        except Exception:
            pass

        px = np.append(close, today_px) if (today_px is not None and today_px > 0) else close
        cp = px[-1]
        local = to_local(sym)

        # 加权线性回归（用最后 LOOKBACK_DAYS+1 个数据点，含今日价格）
        y = np.log(px[-(LOOKBACK_DAYS + 1):])
        x = np.arange(len(y))
        w = np.linspace(1, 2, len(y))
        slope, _ = np.polyfit(x, y, 1, w=w)
        ann = math.exp(slope * 250) - 1
        ss_r = np.sum(w * (y - np.polyval([slope, _], x)) ** 2)
        ss_t = np.sum(w * (y - np.mean(y)) ** 2)
        r2 = 1 - ss_r / ss_t if ss_t > 0 else 0
        mom = ann * r2

        avg_v = np.mean(vol[-VOLUME_LOOKBACK:])
        vr = vol[-1] / avg_v if avg_v > 0 else None

        pl = True
        if len(close) >= 4:
            ds = [close[-1] / close[-2], close[-2] / close[-3], close[-3] / close[-4]]
            if min(ds) < LOSS_RATIO:
                pl = False

        out.append({
            "local": local, "symbol": sym, "name": _etf_names.get(local, local),
            "mom": mom, "r2": r2, "vr": vr, "pl": pl,
            "pm": MIN_SCORE <= mom <= MAX_SCORE,
            "pr": r2 > R2_THRESHOLD,
            "pv": vr is not None and vr < VOLUME_THRESHOLD,
            "_last_close": cp,
        })
    print(f"  {len(out)}有效 {skip}跳过", end="", flush=True)
    if out:
        print(f"  e.g. {out[0]['local']} close={out[0]['_last_close']:.3f} mom={out[0]['mom']:.3f}", flush=True)
    else:
        print(flush=True)
    out.sort(key=lambda x: x["mom"], reverse=True)
    return out


def algo(context):
    global _cooldown_end
    now = context.now.strftime("%Y-%m-%d")
    if ENABLE_COOLDOWN and _cooldown_end and context.now.date() <= _cooldown_end:
        print(f"[{now}] 冷却至{_cooldown_end} 跳过", flush=True)
        return

    all_m = calc_all(context)
    if not all_m:
        print(f"[{now}] 无数据 → 防御", flush=True)
        defensive(context)
        return

    # 过滤
    fil = [m for m in all_m if m["pm"] and (not ENABLE_R2 or m["pr"]) and (not ENABLE_VOLUME or m["pv"]) and (not ENABLE_LOSS or m["pl"])]
    fil.sort(key=lambda x: x["mom"], reverse=True)
    top = fil[:10]
    if not top:
        print(f"[{now}] 无合格 → 防御")
        defensive(context)
        return

    # 候选
    if len(top) >= HOLDINGS_NUM:
        th = top[HOLDINGS_NUM - 1]["mom"] * SCORE_THRESHOLD_RATIO
        cand = [m for m in top if m["mom"] >= th]
    else:
        cand = top[:]

    # 结合持仓
    cur = [to_local(p["symbol"]) for p in context.account().positions() if p["volume"] > 0]
    cm = {m["local"]: m for m in cand}
    ret = [cm[c] for c in cur if c in cm]
    if len(ret) >= HOLDINGS_NUM:
        ret.sort(key=lambda x: x["mom"], reverse=True)
        fin = ret[:HOLDINGS_NUM]
    else:
        need = HOLDINGS_NUM - len(ret)
        rem = [m for m in cand if m["local"] not in {r["local"] for r in ret}]
        fin = ret + rem[:need]

    tgt = [m["symbol"] for m in fin]
    tgt_str = ", ".join(f"{m['local']}({m['name']},{m['mom']:.3f})" for m in fin)
    print(f"[{now}] 目标: {tgt_str}", flush=True)

    cur_s = {p["symbol"] for p in context.account().positions() if p["volume"] > 0}
    print(f"[{now}] 持仓: {len(cur_s)}只 {', '.join(sorted(cur_s))}", flush=True)

    # 卖出现有持仓但不在目标中的 (参照 main.py 模式)
    for pos in context.account().positions():
        s = pos["symbol"]
        if s not in tgt and pos["volume"] > 0:
            print(f"[{now}] 卖出 {s}", flush=True)
            order_target_percent(symbol=s, percent=0,
                                 order_type=OrderType_Market,
                                 position_side=PositionSide_Long)

    # 对所有目标执行 order_target_percent (参照 main.py: 每个目标都调)
    pct = 0.95 / len(tgt) if tgt else 0
    for s in tgt:
        print(f"[{now}] 调仓 {s} → {pct:.1%}", flush=True)
        order_target_percent(symbol=s, percent=pct,
                             order_type=OrderType_Market,
                             position_side=PositionSide_Long)


def defensive(context):
    now = context.now.strftime("%Y-%m-%d")
    s = to_gm(DEFENSIVE_ETF_CODE)
    for p in context.account().positions():
        if p["volume"] > 0 and p["symbol"] != s:
            print(f"[{now}] 防御卖出 {p['symbol']}", flush=True)
            order_target_percent(p["symbol"], 0, OrderType_Market, PositionSide_Long)
    if s not in {p["symbol"] for p in context.account().positions() if p["volume"] > 0}:
        print(f"[{now}] 防御买入 {s}", flush=True)
        order_target_percent(s, 0.95, OrderType_Market, PositionSide_Long)


def get_current_price(sym):
    """获取今日实时价格（用于止损，JQ兼容: current_data.last_price）"""
    try:
        bar = history_n(sym, "60m", 1, fields="close",
                        skip_suspended=True, fill_missing="Last",
                        adjust=0, df=True)
        if bar is not None and not bar.empty:
            return float(bar["close"].values[-1])
    except Exception:
        pass
    return None


def stop_loss(context):
    if not USE_FIXED_STOP_LOSS:
        return
    if ENABLE_COOLDOWN and _cooldown_end and context.now.date() <= _cooldown_end:
        return
    now = context.now.strftime("%Y-%m-%d")
    for p in context.account().positions():
        if p["volume"] <= 0:
            continue
        try:
            cp = get_current_price(p["symbol"])
            if cp is None or cp <= 0:
                cp_bar = history_n(p["symbol"], "1d", 1, "close", skip_suspended=True, adjust=0)
                if cp_bar is None or len(cp_bar) == 0:
                    continue
                cp = cp_bar.iloc[-1]["close"]
            cost = p.get("vwap") or p.get("price", 0)
            if cp <= 0 or cost <= 0:
                continue
            if cp <= cost * STOP_LOSS_PCT:
                print(f"[{now}] 止损 {to_local(p['symbol'])} {(cp/cost-1)*100:.2f}%", flush=True)
                order_target_percent(p["symbol"], 0, OrderType_Market, PositionSide_Long)
                if ENABLE_COOLDOWN:
                    enter_cooldown(context)
                return
        except Exception:
            continue


def on_order_status(context, order):
    if order["status"] == 3:
        side = "买入" if order["side"] == 1 else "卖出"
        local = to_local(order["symbol"])
        fill_vwap = order.get("filled_vwap")
        fill_price = round(float(fill_vwap), 6) if fill_vwap else float(order["price"])
        fill_volume = int(order.get("filled_volume", order["volume"]))
        print(f"[成交] {local} {side} {fill_volume}股 @ {fill_price:.4f}", flush=True)


def enter_cooldown(context):
    global _cooldown_end
    _cooldown_end = context.now.date() + timedelta(days=COOLDOWN_DAYS)
    now = context.now.strftime("%Y-%m-%d")
    print(f"[{now}] 冷却至{_cooldown_end}", flush=True)
    hv = to_gm(SAFE_HAVEN_CODE)
    for p in context.account().positions():
        if p["symbol"] != hv and p["volume"] > 0:
            order_target_percent(p["symbol"], 0, OrderType_Market, PositionSide_Long)
    order_target_percent(hv, 0.95, OrderType_Market, PositionSide_Long)


# ==================== 初始化 ====================
def init(context):
    load_pool()
    print(f"[参数] 持仓{HOLDINGS_NUM} R²{'ON' if ENABLE_R2 else 'OFF'} "
          f"量{'ON' if ENABLE_VOLUME else 'OFF'} 风控{'ON' if ENABLE_LOSS else 'OFF'}", flush=True)
    # JQ: 13:09:59计算 + 13:10卖出 + 13:11买入，GM合并为一次执行
    schedule(schedule_func=algo, date_rule="1d", time_rule="13:09:00")
    if USE_FIXED_STOP_LOSS:
        # JQ: 09:30, 10:30, 14:00, 14:57
        for t in ["09:30:00", "10:30:00", "14:00:00", "14:57:00"]:
            schedule(schedule_func=stop_loss, date_rule="1d", time_rule=t)


if __name__ == '__main__':
    run(strategy_id='f2a9379b-5a86-11f1-b563-0a002700000c',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='1b511135ca6034bc04c9f2eeb66b3a70cb08b831',
        backtest_start_time='2026-01-01 08:00:00',
        backtest_end_time='2026-05-27 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001,
        backtest_match_mode=1)
