"""
用掘金量化(GM) API 下载全部 ETF 日线数据，输出为 etf_74.csv 兼容格式

支持:
  python juejin/download_etf_data.py            # 全量下载
  python juejin/download_etf_data.py --update    # 增量更新（仅追加最新数据）
  python juejin/download_etf_data.py --threads 8  # 多线程加速
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

TOKEN = "1b511135ca6034bc04c9f2eeb66b3a70cb08b831"
ETF_LIST_PATH = Path(__file__).resolve().parent.parent / "etf_data" / "etf_list_before_2022_74.csv"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "etf_data" / "etf_74.csv"

GM_FUNC_CACHE = None


def setup():
    from gm.api import set_token
    set_token(TOKEN)


def _get_func():
    global GM_FUNC_CACHE
    if GM_FUNC_CACHE is not None:
        return GM_FUNC_CACHE
    for name in ["history_n", "history", "get_history_bars", "history_bars", "get_bars"]:
        try:
            mod = __import__("gm.api", fromlist=[name])
            fn = getattr(mod, name, None)
            if fn is not None:
                GM_FUNC_CACHE = (fn, name)
                return fn, name
        except Exception:
            continue
    GM_FUNC_CACHE = (None, None)
    return None, None


def _to_gm_symbol(stock_id):
    code, exchange = stock_id.replace(".SH", ".XSHG").replace(".SZ", ".XSHE").split(".")
    exchange_map = {"XSHG": "SHSE", "XSHE": "SZSE"}
    return f"{exchange_map.get(exchange, exchange)}.{code}"


def fetch_daily_bars(symbol, end_date, start_date=None):
    fn, fn_name = _get_func()
    if fn is None:
        return None
    # 根据起始日期估算所需的 bars 数量（年均 ~250 交易日，+50% buffer）
    count = 1500
    if start_date:
        try:
            sd = datetime.strptime(start_date, "%Y-%m-%d") if isinstance(start_date, str) else start_date
            ed = datetime.strptime(end_date, "%Y-%m-%d") if isinstance(end_date, str) else end_date
            trading_days = int((ed - sd).days * 250 / 365 * 1.5) + 100
            count = trading_days
        except Exception:
            pass
    kwargs = dict(frequency="1d", df=True)
    if fn_name == "history_n":
        kwargs.update(count=count, end_time=end_date)
    elif fn_name == "history":
        kwargs.update(start_time=start_date or "2019-01-01", end_time=end_date)
    else:
        kwargs.update(count=count, end_time=end_date)
    try:
        result = fn(symbol, **kwargs)
    except TypeError as e:
        print(f"    [debug] {symbol} TypeError: {e}")
        fallback = {k: v for k, v in kwargs.items() if k in ("frequency", "count", "end_time", "start_time", "df")}
        try:
            result = fn(symbol, **fallback)
        except Exception as e2:
            print(f"    [debug] {symbol} 降级也失败: {e2}")
            return None
    except Exception as e:
        print(f"    [debug] {symbol} 调用失败: {e}")
        return None
    if result is None:
        return None
    if hasattr(result, "__len__") and len(result) == 0:
        return None
    if hasattr(result, "columns"):
        for date_col in ("eob", "bob", "trade_date"):
            if date_col in result.columns:
                result[date_col] = pd.to_datetime(result[date_col])
                result = result.set_index(date_col)
                break
        # 去时区，按起始日期截断
        if result.index.tz is not None:
            result.index = result.index.tz_localize(None)
        if start_date:
            result = result[result.index >= pd.Timestamp(start_date)]
        return result
    # list[dict] 降级转换
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        df = pd.DataFrame(result)
        for dc in ("eob", "bob", "trade_date"):
            if dc in df.columns:
                df[dc] = pd.to_datetime(df[dc])
                df = df.set_index(dc)
                break
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        return df
    return None


def fetch_history_instruments(symbol, start_date, end_date):
    """获取涨跌停价、停牌、复权因子等"""
    from gm.api import get_history_instruments
    try:
        raw = get_history_instruments(
            symbol,
            fields="trade_date,pre_close,upper_limit,lower_limit,is_suspended,adj_factor,turn_rate",
            start_date=start_date,
            end_date=end_date,
            df=True,
        )
        if raw is not None and len(raw) > 0:
            raw["trade_date"] = pd.to_datetime(raw["trade_date"])
            return raw.set_index("trade_date")
    except Exception:
        pass
    return None


def fetch_single_etf(code, end_date, start_date="2019-01-01", download_instruments=True):
    """获取单只 ETF 的完整数据"""
    symbol = _to_gm_symbol(code)
    bars = fetch_daily_bars(symbol, end_date, start_date)
    if bars is None or len(bars) == 0:
        return None

    bars = bars.rename(columns={
        "open": "开盘", "high": "最高", "low": "最低", "close": "收盘",
        "volume": "成交量", "amount": "成交额", "pre_close": "前收盘",
    })
    bars["股票代码"] = code
    bars["日期"] = bars.index

    if "前收盘" not in bars.columns:
        bars["前收盘"] = bars["收盘"].shift(1)
    bars["涨跌额"] = bars["收盘"] - bars["前收盘"]
    bars["涨跌幅"] = (bars["涨跌额"] / bars["前收盘"] * 100).round(2)
    bars["振幅"] = ((bars["最高"] - bars["最低"]) / bars["前收盘"] * 100).round(2)

    if download_instruments:
        inst = fetch_history_instruments(symbol, start_date, end_date)
        if inst is not None and len(inst) > 0:
            inst.index = inst.index.tz_localize(None)
            for col in ("upper_limit", "lower_limit", "is_suspended", "adj_factor", "turn_rate"):
                if col in inst.columns:
                    bars[col] = inst[col]
        for jq_col, gm_col in [("涨停价", "upper_limit"), ("跌停价", "lower_limit"),
                                ("换手率", "turn_rate")]:
            bars[jq_col] = bars.get(gm_col) if gm_col in bars.columns else None
        bars["复权因子"] = bars.get("adj_factor") if "adj_factor" in bars.columns else 1.0
        bars["停牌"] = bars.get("is_suspended", 0)
        if "停牌" in bars.columns:
            bars["停牌"] = bars["停牌"].fillna(0).astype(int)

    return bars


def download_all(codes, end_date, start_date="2019-01-01", threads=1):
    """下载全部 ETF，支持多线程"""
    all_data = []
    errors = []

    if threads > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=threads) as pool:
            futures = {
                pool.submit(fetch_single_etf, code, end_date, start_date): code
                for code in codes
            }
            for i, future in enumerate(as_completed(futures), 1):
                code = futures[future]
                t1 = time.time()
                try:
                    df = future.result()
                except Exception:
                    df = None
                elapsed = time.time() - t1
                if df is not None and len(df) > 0:
                    all_data.append(df)
                    print(f"  [{i}/{len(codes)}] {code} OK ({elapsed:.1f}s, {len(df)} rows)")
                else:
                    errors.append(code)
                    print(f"  [{i}/{len(codes)}] {code} FAIL ({elapsed:.1f}s)")
    else:
        for i, code in enumerate(codes, 1):
            t1 = time.time()
            df = fetch_single_etf(code, end_date, start_date)
            elapsed = time.time() - t1
            if df is not None and len(df) > 0:
                all_data.append(df)
                print(f"  [{i}/{len(codes)}] {code} OK ({elapsed:.1f}s, {len(df)} rows)")
            else:
                errors.append(code)
                print(f"  [{i}/{len(codes)}] {code} FAIL ({elapsed:.1f}s)")

    if not all_data:
        return None

    combined = pd.concat(all_data, ignore_index=True)
    return combined, errors


def save_csv(df, path):
    """保存为与 JQ 兼容的 CSV 格式"""
    cols_out = [
        "股票代码", "日期",
        "开盘", "收盘", "最高", "最低", "前收盘",
        "涨跌额", "涨跌幅", "振幅",
        "成交量", "成交额",
        "涨停价", "跌停价", "停牌", "换手率", "复权因子",
        "开盘_原始", "收盘_原始", "最高_原始", "最低_原始", "前收盘_原始",
    ]
    df["日期"] = df["日期"].dt.tz_localize(None)

    for raw_col in ("开盘_原始", "收盘_原始", "最高_原始", "最低_原始", "前收盘_原始"):
        base = raw_col.replace("_原始", "")
        df[raw_col] = df[base] if base in df.columns else None

    missing_cols = [c for c in cols_out if c not in df.columns]
    for c in missing_cols:
        df[c] = None

    df[cols_out].to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"\n已保存 {len(df)} 行 → {path}")


def merge_existing(new_df):
    """增量更新: 合并已有的 etf_74.csv，覆盖已有日期"""
    if not OUTPUT_PATH.exists():
        return new_df
    old = pd.read_csv(OUTPUT_PATH)
    old["日期"] = pd.to_datetime(old["日期"])
    old_key = old.set_index(["股票代码", "日期"])
    new_key = new_df.set_index(["股票代码", "日期"])
    old_key.update(new_key)
    merged = pd.concat([old_key, new_key[~new_key.index.isin(old_key.index)]])
    merged = merged.reset_index().sort_values(["股票代码", "日期"]).reset_index(drop=True)
    return merged


def main():
    parser = argparse.ArgumentParser(description="用 GM API 下载 ETF 日线数据")
    parser.add_argument("--update", action="store_true", help="增量更新（仅追加最新数据）")
    parser.add_argument("--threads", type=int, default=4, help="并行线程数 (默认 4)")
    parser.add_argument("--start-date", type=str, default="2022-01-01",
                        help="起始日期 (默认 2022-01-01)")
    args = parser.parse_args()

    print("=" * 60)
    print("掘金量化 GM API — ETF 数据下载")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    setup()

    if not ETF_LIST_PATH.exists():
        print(f"❌ 未找到 ETF 列表: {ETF_LIST_PATH}")
        sys.exit(1)

    codes = pd.read_csv(ETF_LIST_PATH)["代码"].tolist()
    print(f"\nETF 列表: {len(codes)} 只")

    end_date = datetime.now().strftime("%Y-%m-%d")

    if args.update and OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        existing["日期"] = pd.to_datetime(existing["日期"])
        last_date = existing["日期"].max()
        start_date = (last_date - timedelta(days=5)).strftime("%Y-%m-%d")
        print(f"增量更新模式: 现有数据最新 {last_date.date()}, 从 {start_date} 开始")
    else:
        start_date = args.start_date
        print(f"全量下载模式, 起始: {start_date}")

    print(f"下载区间: {start_date} ~ {end_date}")
    print(f"线程数: {args.threads}\n")

    t0 = time.time()
    result = download_all(codes, end_date, start_date, threads=args.threads)

    if result is None:
        print("\n❌ 全部失败")
        sys.exit(1)

    combined, errors = result
    total_time = time.time() - t0

    print(f"\n总计: {len(codes) - len(errors)}/{len(codes)} 成功, {len(errors)} 失败, 耗时 {total_time:.0f}s")
    print(f"总行数: {len(combined)}")

    if args.update:
        combined = merge_existing(combined)
        print(f"合并后总行数: {len(combined)}")

    save_csv(combined, OUTPUT_PATH)

    if errors:
        print(f"\n⚠️ 失败的 ETF: {errors}")


if __name__ == "__main__":
    main()
