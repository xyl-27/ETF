"""
探索从 掘金量化(GM) API 获取 ETF 行情数据
对比 JoinQuant 现有数据，评估接入可行性

运行: python juejin/fetch_gm_data.py
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================
# 配置
# ============================================================
TOKEN = "1b511135ca6034bc04c9f2eeb66b3a70cb08b831"
ETF_LIST_PATH = Path(__file__).resolve().parent.parent / "etf_data" / "etf_list_before_2022_74.csv"
JQ_DATA_PATH = Path(__file__).resolve().parent.parent / "etf_data" / "etf_74.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def step1_setup():
    """初始化 GM SDK 并设置 token"""
    print("=" * 60)
    print("Step 1: 初始化 GM SDK")
    print("=" * 60)
    try:
        from gm.api import set_token
        set_token(TOKEN)
        print("✅ token 设置成功")
        return True
    except ImportError:
        print("❌ gm SDK 未安装")
        print("   安装: pip install gm-server-sdk -i https://pypi.myquant.cn/simple")
        return False
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return False


def step2_list_etfs():
    """读取现有 ETF 列表"""
    print("\n" + "=" * 60)
    print("Step 2: ETF 列表")
    print("=" * 60)
    import pandas as pd
    if not ETF_LIST_PATH.exists():
        print(f"❌ 未找到 ETF 列表: {ETF_LIST_PATH}")
        return None
    df = pd.read_csv(ETF_LIST_PATH)
    print(f"ETF 数量: {len(df)}")
    print(df.head(10).to_string(index=False))
    print("...")
    return df


_GM_FUNC_CACHE = None


def _get_func():
    """自动探测 GM API 中可用的日线获取函数"""
    global _GM_FUNC_CACHE
    if _GM_FUNC_CACHE is not None:
        return _GM_FUNC_CACHE

    for name in ["history_n", "history", "get_history_bars", "history_bars", "get_bars"]:
        try:
            mod = __import__("gm.api", fromlist=[name])
            fn = getattr(mod, name, None)
            if fn is not None:
                _GM_FUNC_CACHE = (fn, name)
                return fn, name
        except Exception:
            continue
    _GM_FUNC_CACHE = (None, None)
    return None, None


def _to_gm_symbol(stock_id):
    """510300.XSHG -> SHSE.510300  或  510300.SH -> SHSE.510300"""
    code, exchange = stock_id.replace(".SH", ".XSHG").replace(".SZ", ".XSHE").split(".")
    exchange_map = {"XSHG": "SHSE", "XSHE": "SZSE"}
    return f"{exchange_map.get(exchange, exchange)}.{code}"


def _fetch_daily_bars(symbol, start_date, end_date, count=1000):
    """统一调用 GM 日线 API，兼容 history_n / history 等不同接口"""
    import pandas as pd
    fn, fn_name = _get_func()
    if fn is None:
        return None

    # 尝试多种 symbol 格式
    symbol_variants = [symbol, _to_gm_symbol(symbol)]
    # 去重
    seen = set()
    unique_variants = []
    for s in symbol_variants:
        if s not in seen:
            seen.add(s)
            unique_variants.append(s)

    for sym in unique_variants:
        kwargs = dict(
            frequency="1d",
            skip_suspended=True,
            fill_missing=None,
            df=True,
        )

        if fn_name == "history_n":
            kwargs["count"] = count
            kwargs["end_time"] = end_date
        elif fn_name == "history":
            kwargs["start_time"] = start_date
            kwargs["end_time"] = end_date
        else:
            kwargs["count"] = count
            kwargs["end_time"] = end_date

        try:
            result = fn(sym, **kwargs)
        except TypeError as te:
            # 不支持 df=True 则降级
            fallback_kw = {k: v for k, v in kwargs.items() if k not in ("df", "skip_suspended", "fill_missing")}
            try:
                result = fn(sym, **fallback_kw)
            except Exception as e2:
                print(f"    [debug] {sym} 降级也失败: {e2}")
                continue
        except Exception as e:
            print(f"    [debug] {sym} 调用失败: {e}")
            continue

        if result is not None:
            # df=True → DataFrame
            if hasattr(result, "columns"):
                if len(result) > 0:
                    # 尝试设置日期索引
                    for date_col in ("eob", "bob", "trade_date"):
                        if date_col in result.columns:
                            result[date_col] = pd.to_datetime(result[date_col])
                            result = result.set_index(date_col)
                            break
                    return result
            # list[dict] → DataFrame
            elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                df = pd.DataFrame(result)
                for drop_col in ("symbol", "frequency"):
                    if drop_col in df.columns:
                        df = df.drop(columns=[drop_col])
                for date_col in ("eob", "bob", "trade_date"):
                    if date_col in df.columns:
                        df[date_col] = pd.to_datetime(df[date_col])
                        df = df.set_index(date_col)
                        break
                return df
            elif hasattr(result, "__len__") and len(result) > 0:
                return result
        else:
            print(f"    [debug] {sym} 返回空")

    return None


def step3_fetch_daily(symbol, start_date="2026-01-01", end_date=None):
    """获取单只 ETF 日线行情"""
    fn, fn_name = _get_func()
    if fn is None:
        tried = "history_n / history / get_history_bars / history_bars / get_bars"
        print(f"❌ 未找到 GM 日线 API (尝试了 {tried})")
        return None

    end_date = end_date or datetime.now().strftime("%Y-%m-%d")
    print(f"\n  获取 {symbol} {start_date} → {end_date} (API: {fn_name}) ...")
    try:
        df = _fetch_daily_bars(symbol, start_date, end_date)
        if df is not None and len(df) > 0:
            idx0, idx1 = df.index[0], df.index[-1]
            print(f"    ✅ {len(df)} 条记录, {idx0} ~ {idx1}")
            return df
        else:
            print(f"    ⚠️ 空数据")
            return None
    except Exception as e:
        print(f"    ❌ 错误: {e}")
        return None


def step4_fetch_multiple():
    """获取全部 74 只 ETF 并统计"""
    print("\n" + "=" * 60)
    print("Step 3: 批量获取全部 ETF 日线")
    print("=" * 60)
    import pandas as pd
    import time

    if not ETF_LIST_PATH.exists():
        return None

    codes = pd.read_csv(ETF_LIST_PATH)["代码"].tolist()
    end = datetime.now().strftime("%Y-%m-%d")
    start = "2026-01-01"

    all_data = []
    errors = []
    t0 = time.time()

    for i, code in enumerate(codes):
        symbol = _to_gm_symbol(code)
        t1 = time.time()
        df = step3_fetch_daily(symbol, start, end)
        elapsed = time.time() - t1
        if df is not None:
            df["股票代码"] = code
            df["日期"] = df.index
            all_data.append(df)
            status = "OK"
        else:
            errors.append(code)
            status = "FAIL"
        print(f"    [{i+1}/{len(codes)}] {code} → {symbol} {elapsed:.1f}s {status}")

    total_time = time.time() - t0
    print(f"\n总计: {len(all_data)}/{len(codes)} 成功, {len(errors)} 失败, 耗时 {total_time:.0f}s")

    if not all_data:
        return None

    combined = pd.concat(all_data, ignore_index=True)
    print(f"合并后: {len(combined)} 条, {combined['股票代码'].nunique()} 只 ETF")
    return combined


def step5_compare_jq(gm_data):
    """对比 GM 和 JoinQuant 数据"""
    print("\n" + "=" * 60)
    print("Step 4: 对比 GM vs JoinQuant 数据")
    print("=" * 60)
    import pandas as pd

    if gm_data is None or not JQ_DATA_PATH.exists():
        print("跳过对比")
        return

    jq = pd.read_csv(JQ_DATA_PATH)
    jq["日期"] = pd.to_datetime(jq["日期"])

    # 补齐日期归一化: GM 有时区(+08:00)，JQ 无时区
    gm_dates = gm_data["日期"].dt.tz_localize(None)
    jq_dates = jq["日期"]

    gm_latest = gm_dates.max()
    jq_latest = jq_dates.max()
    print(f"GM 最新日期: {gm_latest}")
    print(f"JQ 最新日期: {jq_latest}")

    common_date = min(gm_latest, jq_latest)
    print(f"对比日期: {common_date}")

    gm_sub = gm_data[gm_dates == common_date].copy()
    jq_sub = jq[jq_dates == common_date].copy()

    if len(gm_sub) == 0 or len(jq_sub) == 0:
        print(f"无共同日期数据: GM有{len(gm_sub)}条, JQ有{len(jq_sub)}条")
        return

    print(f"共有 ETF: GM={len(gm_sub)}, JQ={len(jq_sub)}")

    # --- 字段结构对比 ---
    print(f"\nGM 字段 ({len(gm_data.columns)}): {list(gm_data.columns)}")
    print(f"JQ 字段 ({len(jq.columns)}): {list(jq.columns)}")
    print(f"共同字段: {set(gm_data.columns) & set(jq.columns)}")
    print(f"GM 独有: {set(gm_data.columns) - set(jq.columns)}")
    print(f"JQ 独有: {set(jq.columns) - set(gm_data.columns)}")

    # 字段映射对比（同日期、同股票代码对齐）
    field_map = {
        "open": "开盘", "close": "收盘", "high": "最高", "low": "最低",
        "volume": "成交量", "amount": "成交额", "adj_factor": "复权因子",
    }
    print("\n字段映射对比 (GM→JQ):")
    gm_common = gm_sub.set_index("股票代码")
    jq_common = jq_sub.set_index("股票代码")
    for gm_col, jq_col in field_map.items():
        if gm_col in gm_common.columns and jq_col in jq_common.columns:
            aligned = gm_common[[gm_col]].join(jq_common[[jq_col]], how="inner")
            aligned["diff"] = (aligned[gm_col] - aligned[jq_col]).abs()
            max_diff = aligned["diff"].max()
            mean_diff = aligned["diff"].mean()
            print(f"  {gm_col:12s} → {jq_col:8s}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    # --- 收盘价逐只对比 ---
    gm_close = gm_sub.set_index("股票代码")["close"].rename("gm_close")
    jq_close = jq_sub.set_index("股票代码")["收盘"].rename("jq_close")
    compare = pd.concat([gm_close, jq_close], axis=1).dropna()
    compare["diff_pct"] = (compare["gm_close"] - compare["jq_close"]) / compare["jq_close"] * 100

    print(f"\n收盘价对比 ({common_date.date()}, 样本数: {len(compare)}):")
    print(compare.describe())

    max_diff = compare["diff_pct"].abs().max()
    mean_abs_diff = compare["diff_pct"].abs().mean()
    print(f"\n平均差异: {mean_abs_diff:.4f}%")
    print(f"最大差异: {max_diff:.4f}%")
    if max_diff < 0.1:
        print("✅ 两数据源高度一致")
    elif max_diff < 1:
        print("⚠️ 轻微差异，可能因复权方式不同")
    else:
        print("❌ 差异较大，需排查原因")

    # 打印差异最大的前 10 只
    top_diff = compare.reindex(compare["diff_pct"].abs().sort_values(ascending=False).index).head(10)
    print(f"\n差异最大的 10 只 ETF:")
    print(top_diff.to_string())

    (OUTPUT_DIR / "gm_vs_jq_compare.csv").write_text(
        compare.to_csv(encoding="utf-8-sig"), encoding="utf-8"
    )
    print(f"\n对比结果已保存: gm_vs_jq_compare.csv")


def step6_benchmark():
    """压测: 单线程 vs 多线程下载速度"""
    print("\n" + "=" * 60)
    print("Step 5: 下载速度对比 (可选)")
    print("=" * 60)
    # 用户可根据需要实现多线程版本
    print("当前单线程 ~250ms/只, 74只 ≈ 18s")
    print("可考虑 concurrent.futures.ThreadPoolExecutor 加速")


def step7_inspect_gm_fields():
    """查看 GM 返回的字段结构，探索更多可用字段"""
    print("\n" + "=" * 60)
    print("Step 6: GM 数据字段探索")
    print("=" * 60)
    fn, fn_name = _get_func()
    if fn is None:
        print("❌ 无法获取数据，跳过")
        return

    sym = "SZSE.159605"
    end = datetime.now().strftime("%Y-%m-%d")

    # 尝试更多的 fields 组合，看看哪些字段可用
    field_sets = [
        "symbol,frequency,open,high,low,close,volume,amount",
        "symbol,frequency,open,high,low,close,volume,amount,adj_factor,eob",
        "symbol,frequency,open,high,low,close,volume,amount,adj_factor,eob,limit_up,limit_down",
        "symbol,frequency,open,high,low,close,volume,amount,adj_factor,eob,limit_up,limit_down,suspended,is_suspended",
        "symbol,frequency,open,high,low,close,volume,amount,adj_factor,eob,limit_up,limit_down,suspended,turnover",
    ]
    print(f"测试 {sym}:")
    for fields in field_sets:
        try:
            if fn_name == "history_n":
                raw = fn(sym, frequency="1d", count=3, end_time=end,
                         fields=fields, skip_suspended=True, fill_missing=None)
            else:
                raw = fn(sym, frequency="1d", count=3, end_time=end, fields=fields)
            if raw and len(raw) > 0:
                present = [k for k in raw[0].keys() if k not in ("symbol", "frequency")]
                missing = [k for k in fields.split(",") if k not in raw[0] and k not in ("symbol", "frequency")]
                print(f"  ✅ fields=[{','.join(present)}]")
                if missing:
                    print(f"     无: {missing}")
            else:
                print(f"  ❌ 空返回: {fields[:60]}")
        except Exception as e:
            print(f"  ❌ 错误: {e}")

    # 尝试 stk_get_daily_basic
    print("\n试试 stk_get_daily_basic:")
    try:
        from gm.api import stk_get_daily_basic
        raw = stk_get_daily_basic(sym,
            fields="limit_up,limit_down,suspended,turnover,adj_factor,pe,pb,market_value",
            start_date="2026-05-10", end_date=end, df=True)
        if raw is not None:
            print(f"  行数: {len(raw)}, 列: {list(raw.columns)}")
            if len(raw) > 0:
                print(f"  示例: {raw.iloc[0].to_dict()}")
        else:
            print(f"  返回 None")
    except Exception as e:
        print(f"  ❌ {e}")

    # get_history_instruments: 含涨停价/跌停价/停牌/复权因子
    print("\n试试 get_history_instruments:")
    try:
        from gm.api import get_history_instruments
        fields = "symbol,trade_date,pre_close,upper_limit,lower_limit,is_suspended,adj_factor,turn_rate"
        raw = get_history_instruments(sym, fields=fields, start_date="2026-05-10", end_date=end, df=True)
        if raw is not None and len(raw) > 0:
            print(f"  ✅ 行数={len(raw)}, 列={list(raw.columns)}")
            print(f"  示例: {raw.iloc[0].to_dict()}")
        else:
            print(f"  ⚠️ 空返回")
    except Exception as e:
        print(f"  ❌ {e}")

    # 无fields 版本
    try:
        from gm.api import get_history_instruments
        raw = get_history_instruments(sym, start_date="2026-05-10", end_date=end, df=True)
        if raw is not None and len(raw) > 0:
            print(f"  无fields版本: 列={list(raw.columns)}")
    except Exception as e:
        print(f"  ❌ {e}")


def step8_check_real_time():
    """测试盘中实时数据获取"""
    print("\n" + "=" * 60)
    print("Step 7: 实时行情测试 (可选)")
    print("=" * 60)
    try:
        from gm.api import get_realtime_ticks
        ticks = get_realtime_ticks("159605.SZ", count=5)
        print(f"实时 tick: {ticks}")
    except ImportError:
        print("get_realtime_ticks 不可用")
    except Exception as e:
        print(f"实时行情接口不可用: {e}")


def step0_debug():
    """调试: 列出 gm.api 所有公开函数"""
    print("\n" + "=" * 60)
    print("Step 0: gm.api 可用的数据接口")
    print("=" * 60)
    import gm.api as gm
    funcs = [n for n in dir(gm) if not n.startswith("_") and ("history" in n.lower() or "get_" in n.lower() or "bar" in n.lower())]
    for f in sorted(funcs):
        obj = getattr(gm, f)
        if callable(obj):
            import inspect
            sig = str(inspect.signature(obj))[:120]
            print(f"  {f}{sig}")


if __name__ == "__main__":
    print("=" * 60)
    print("掘金量化(GM) API 数据获取探索")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not step1_setup():
        sys.exit(1)

    etf_list = step2_list_etfs()
    if etf_list is None:
        sys.exit(1)

    step0_debug()

    step7_inspect_gm_fields()

    step8_check_real_time()

    gm_data = step4_fetch_multiple()

    step5_compare_jq(gm_data)

    step6_benchmark()
