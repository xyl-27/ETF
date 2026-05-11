"""
接入掘金银河实盘交易数据，同步更新日报 latest_report.json

用法:
  # 从掘金交易结果 JSON 文件同步
  python juejin/sync_report.py --from-juejin juejin_result.json

  # 手动指定更新字段
  python juejin/sync_report.py --override-trades trades.json --override-holdings holdings.json

  # 更新后重新发送邮件
  python juejin/sync_report.py --from-juejin result.json --send-email

API 用法:
  from juejin.sync_report import apply_juejin_result, sync_and_send
  apply_juejin_result("juejin_result.json")
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

REPORT_PATH = PROJECT_ROOT / "output" / "latest_report.json"
STATE_PATH = PROJECT_ROOT / "output" / "backtest_state.json"


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.ndarray,)): return obj.tolist()
        return super().default(obj)


def load_report(path=None):
    path = path or REPORT_PATH
    if not path.exists():
        print(f"错误: 未找到日报文件 {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_report(report, path=None):
    path = path or REPORT_PATH
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)
    print(f"[同步] 日报已保存: {path}")


def apply_juejin_result(juejin_json_path, report=None, model_key="search_itransformer_exp_54", dry_run=False):
    """从掘金回测结果 JSON 更新日报。

    期望的掘金结果格式:
    {
        "model_key": "search_itransformer_exp_54",   # 可选，默认使用 i54
        "equity_curve": [{"date": "2026-05-11", "total_value": 134094.95}, ...],
        "trades": [{"date": "2026-05-11", "action": "买入", "stock": "510300.XSHG",
                    "price": 4.1234, "shares": 10000, "amount": 41234.0}, ...],
        "holdings": [{"stock_id": "510300.XSHG", "shares": 10000,
                      "price": 4.1234, "cost": 41234.0}, ...],
        "cash": 50000.0,
        "total_value": 134094.95,
        "metrics": {                              # 可选，覆盖默认计算
            "strategy_return_pct": 34.095,
            "daily_win_rate": 0.7083,
            "max_drawdown_pct": 2.35,
            "sharpe_ratio": 10.95
        }
    }

    参数:
        juejin_json_path: JSON 文件路径 或 dict
        report: 日报 dict（None 则自动加载 latest_report.json）
        model_key: 要更新的模型标识
        dry_run: 为 True 时不保存到文件，仅返回修改后的 dict
    """
    if report is None:
        report = load_report()
        if report is None:
            return None

    if isinstance(juejin_json_path, (str, Path)):
        with open(juejin_json_path, "r", encoding="utf-8") as f:
            juejin_data = json.load(f)
    else:
        juejin_data = juejin_json_path

    mk = juejin_data.get("model_key", model_key)
    updated_keys = []

    # 1. 覆盖 equity_curve → 更新到 sequences[model_key]
    if "equity_curve" in juejin_data:
        ec = juejin_data["equity_curve"]
        report.setdefault("sequences", {}).setdefault(mk, {})["equity_curve"] = ec
        updated_keys.append("equity_curve")
        # 同步到 backtest_state.json
        if not dry_run and STATE_PATH.exists():
            try:
                with open(STATE_PATH, "r", encoding="utf-8") as f:
                    state = json.load(f)
                for seq_key in state.get("sequences", {}):
                    state["sequences"][seq_key]["equity_curve"] = ec
                with open(STATE_PATH, "w", encoding="utf-8") as f:
                    json.dump(state, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)
                print(f"[同步] backtest_state.json equity_curve 已更新")
            except Exception as e:
                print(f"[同步] 更新 backtest_state.json 失败: {e}")

    # 2. 覆盖 trades → 更新到 sequences[model_key]
    if "trades" in juejin_data:
        trades = juejin_data["trades"]
        report.setdefault("sequences", {}).setdefault(mk, {})["trades"] = trades
        # 同步到主序列
        if mk in report.get("sequences", {}):
            report["sequences"][mk]["trades_count"] = len(trades)
        updated_keys.append("trades")

    # 3. 覆盖 holdings
    if "holdings" in juejin_data:
        report["holdings"] = juejin_data["holdings"]
        updated_keys.append("holdings")

    # 4. 覆盖 cash
    if "cash" in juejin_data:
        cash = juejin_data["cash"]
        report["cash"] = cash
        report.setdefault("sequences", {}).setdefault(mk, {})["cash"] = cash
        updated_keys.append("cash")

    # 5. 覆盖 total_value 和 metrics
    if "total_value" in juejin_data:
        report["total_value"] = juejin_data["total_value"]
        report.setdefault("sequences", {}).setdefault(mk, {}).setdefault("metrics", {})["latest_value"] = juejin_data["total_value"]
        updated_keys.append("total_value")

    if "metrics" in juejin_data:
        for k, v in juejin_data["metrics"].items():
            report.setdefault("sequences", {}).setdefault(mk, {}).setdefault("metrics", {})[k] = v
        updated_keys.append("metrics")

    # 6. 覆盖今日调仓
    if "today_trades" in juejin_data:
        report["today_trades"] = juejin_data["today_trades"]
        report["all_today_trades"] = juejin_data.get("all_today_trades", juejin_data["today_trades"])
        updated_keys.append("today_trades")
        if any(t.get("action") in ("买入", "卖出") for t in juejin_data["today_trades"]):
            report["is_rebalance_day"] = True

    # 7. 更新日期
    if "date" in juejin_data:
        report["date"] = juejin_data["date"]

    print(f"[同步] 已更新字段: {', '.join(updated_keys)}")
    if not dry_run:
        save_report(report)
    else:
        print("[同步] dry_run 模式，未保存到文件")
    return report


def sync_and_send(juejin_json_path, model_key="search_itransformer_exp_54"):
    """同步掘金数据并重新发送邮件报告。"""
    report = apply_juejin_result(juejin_json_path, model_key=model_key)
    if report is None:
        return False

    try:
        from send_report import send_report
        print("[同步] 重新发送邮件报告...")
        success = send_report(model_key=model_key)
        if success:
            print("[同步] 邮件发送成功")
        return success
    except Exception as e:
        print(f"[同步] 邮件发送失败: {e}")
        return False


def override_from_cli():
    """CLI 入口：从命令行参数覆盖日报字段。"""
    import argparse

    parser = argparse.ArgumentParser(description="同步掘金实盘交易数据到日报")
    parser.add_argument("--from-juejin", type=str, default=None,
                        help="掘金回测结果 JSON 文件路径")
    parser.add_argument("--model-key", type=str, default="search_itransformer_exp_54",
                        help="模型标识 (默认 i54)")
    parser.add_argument("--send-email", action="store_true",
                        help="同步后重新发送邮件")
    parser.add_argument("--override-trades", type=str, default=None,
                        help="覆盖交易列表的 JSON 文件")
    parser.add_argument("--override-holdings", type=str, default=None,
                        help="覆盖持仓列表的 JSON 文件")
    parser.add_argument("--override-equity", type=str, default=None,
                        help="覆盖净值曲线的 JSON 文件")
    parser.add_argument("--override-cash", type=float, default=None,
                        help="覆盖现金余额")
    parser.add_argument("--override-total-value", type=float, default=None,
                        help="覆盖账户总值")
    parser.add_argument("--override-metrics", type=str, default=None,
                        help="覆盖指标字典的 JSON 文件")
    parser.add_argument("--date", type=str, default=None,
                        help="覆盖报告日期")
    parser.add_argument("--save-only", action="store_true",
                        help="仅保存，不发送邮件")

    args = parser.parse_args()

    report = load_report()
    if report is None:
        return 1

    model_key = args.model_key
    updated = False

    def _load_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    if args.from_juejin:
        report = apply_juejin_result(args.from_juejin, report, model_key=model_key)
        if report:
            updated = True

    if args.override_trades:
        trades = _load_json(args.override_trades)
        report.setdefault("sequences", {}).setdefault(model_key, {})["trades"] = trades
        print(f"[CLI] 覆盖 trades ({len(trades)} 条)")
        updated = True

    if args.override_holdings:
        holdings = _load_json(args.override_holdings)
        report["holdings"] = holdings
        print(f"[CLI] 覆盖 holdings ({len(holdings)} 条)")
        updated = True

    if args.override_equity:
        ec = _load_json(args.override_equity)
        report.setdefault("sequences", {}).setdefault(model_key, {})["equity_curve"] = ec
        # 也同步到 backtest_state
        if STATE_PATH.exists():
            try:
                with open(STATE_PATH, "r", encoding="utf-8") as f:
                    state = json.load(f)
                for seq_key in state.get("sequences", {}):
                    state["sequences"][seq_key]["equity_curve"] = ec
                with open(STATE_PATH, "w", encoding="utf-8") as f:
                    json.dump(state, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[CLI] 同步 backtest_state 失败: {e}")
        print(f"[CLI] 覆盖 equity_curve ({len(ec)} 条)")
        updated = True

    if args.override_cash is not None:
        report["cash"] = args.override_cash
        report.setdefault("sequences", {}).setdefault(model_key, {})["cash"] = args.override_cash
        print(f"[CLI] 覆盖 cash: {args.override_cash}")
        updated = True

    if args.override_total_value is not None:
        report["total_value"] = args.override_total_value
        report.setdefault("sequences", {}).setdefault(model_key, {}).setdefault("metrics", {})["latest_value"] = args.override_total_value
        print(f"[CLI] 覆盖 total_value: {args.override_total_value}")
        updated = True

    if args.override_metrics:
        metrics = _load_json(args.override_metrics)
        for k, v in metrics.items():
            report.setdefault("sequences", {}).setdefault(model_key, {}).setdefault("metrics", {})[k] = v
        print(f"[CLI] 覆盖 metrics: {list(metrics.keys())}")
        updated = True

    if args.date:
        report["date"] = args.date
        print(f"[CLI] 覆盖 date: {args.date}")
        updated = True

    if updated:
        save_report(report)

    if args.send_email and updated:
        try:
            from send_report import send_report
            print("[CLI] 重新发送邮件...")
            send_report(model_key=model_key)
        except Exception as e:
            print(f"[CLI] 邮件发送失败: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(override_from_cli())
