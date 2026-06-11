"""
ETF AI预测策略 - 回测模块
"""

import pandas as pd
import numpy as np
import json
import joblib
import torch
import multiprocessing as mp
import os
import tempfile
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
try:
    from .risk_strategies import RiskStrategy
except ImportError:
    from risk_strategies import RiskStrategy

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class BacktestResult:
    """回测结果数据类"""

    start_date: str
    end_date: str
    strategy_return: float
    hs300_return: float
    excess_return: float
    max_drawdown: float
    drawdown_days: int
    recovered: bool
    recovery_days: Optional[int]
    log_file: Optional[str] = None
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    hs300_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    rebalance_stats: Dict[str, Any] = field(default_factory=dict)
    trades: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "回测期间": f"{self.start_date} ~ {self.end_date}",
            "策略累计收益": f"{self.strategy_return:.2f}%",
            "HS300累计收益": f"{self.hs300_return:.2f}%",
            "超额收益": f"{self.excess_return:.2f}%",
            "最大回撤": f"{self.max_drawdown:.2f}%",
            "回撤持续天数": self.drawdown_days,
            "是否恢复": self.recovered,
            "恢复天数": self.recovery_days,
        }
        if self.rebalance_stats:
            result["调仓次数"] = self.rebalance_stats.get("total", 0)
            result["胜率"] = f"{self.rebalance_stats.get('win_rate', 0):.1f}%"
            result["平均每次调仓收益"] = f"{self.rebalance_stats.get('avg_return', 0):.2f}%"
        if self.log_file:
            result["日志文件"] = self.log_file
        return result

    def print_summary(self):
        print("=" * 50)
        print("回测结果汇总")
        print("=" * 50)
        for k, v in self.to_dict().items():
            print(f"{k}: {v}")
        if self.rebalance_stats.get("returns"):
            rets = self.rebalance_stats["returns"]
            print("-" * 50)
            print(f"调仓次数: {self.rebalance_stats['total']}")
            print(f"盈利次数: {self.rebalance_stats['wins']}")
            print(f"胜率: {self.rebalance_stats['win_rate']:.1f}%")
            print(f"平均每次调仓收益: {self.rebalance_stats['avg_return']:.2f}%")
            print(f"最好: {max(rets):+.2f}%")
            print(f"最差: {min(rets):+.2f}%")
        print("=" * 50)

    def plot(self, save_path: str = None):
        """绘制回测结果图表"""
        if self.equity_curve.empty:
            print("没有回测数据可绘制")
            return

        equity = self.equity_curve.copy()
        equity["date"] = pd.to_datetime(equity["date"])
        initial_capital = equity["total_value"].iloc[0]
        equity["策略收益率"] = (equity["total_value"] / initial_capital - 1) * 100

        if not self.hs300_data.empty:
            hs300 = self.hs300_data.copy()
            hs300["date"] = pd.to_datetime(hs300["date"])
            initial_hs300 = hs300["close"].iloc[0]
            hs300["基准收益率"] = (hs300["close"] / initial_hs300 - 1) * 100

            merged = pd.merge(
                equity[["date", "策略收益率"]],
                hs300[["date", "基准收益率"]],
                on="date",
                how="inner",
            )
        else:
            merged = equity[["date", "策略收益率"]].copy()
            merged["基准收益率"] = 0.0

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_facecolor("#f5f5f5")
        fig.patch.set_facecolor("white")

        ax.plot(
            merged["date"],
            merged["策略收益率"],
            label="策略收益率",
            color="#2ecc71",
            linewidth=2.5,
            alpha=0.9,
        )

        if not self.hs300_data.empty:
            ax.plot(
                merged["date"],
                merged["基准收益率"],
                label="沪深300收益率",
                color="#e74c3c",
                linewidth=2,
                alpha=0.8,
                linestyle="--",
            )

        ax.fill_between(
            merged["date"],
            0,
            merged["策略收益率"],
            where=merged["策略收益率"] >= 0,
            color="#2ecc71",
            alpha=0.15,
            interpolate=True,
        )
        ax.fill_between(
            merged["date"],
            0,
            merged["策略收益率"],
            where=merged["策略收益率"] < 0,
            color="#e74c3c",
            alpha=0.15,
            interpolate=True,
        )

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
        ax.set_title(
            "策略 vs 沪深300 累计收益率", fontsize=16, fontweight="bold", pad=20
        )
        ax.set_xlabel("日期", fontsize=12)
        ax.set_ylabel("累计收益率 (%)", fontsize=12)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45, ha="right")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
        ax.legend(loc="upper left", fontsize=11, framealpha=0.9, edgecolor="gray")

        color = "#2ecc71" if self.excess_return >= 0 else "#e74c3c"
        ax.text(
            0.02,
            0.75,
            f"最终策略收益: {self.strategy_return:.2f}%",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="#2ecc71", alpha=0.3),
        )
        ax.text(
            0.02,
            0.68,
            f"最终基准收益: {self.hs300_return:.2f}%",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="#e74c3c", alpha=0.3),
        )
        ax.text(
            0.02,
            0.61,
            f"超额收益: {self.excess_return:+.2f}%",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor=color, alpha=0.3),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"图表已保存到: {save_path}")
        else:
            plt.show()

        plt.close()


def compute_volatility(price_data, stock_ids, current_date, window=20):
    """compute annualized volatility for each stock_id from price_data up to current_date.

    returns dict[stock_id, annualized_vol] (floor 0.001 to avoid div-by-zero).
    """
    if price_data is None:
        return {}
    hist = price_data[pd.to_datetime(price_data["日期"]) < pd.to_datetime(current_date)]
    vol_dict = {}
    for sid in stock_ids:
        closes = hist.loc[hist["股票代码"] == sid, "收盘"].values
        if len(closes) >= window + 1:
            rets = np.diff(closes[-(window + 1):]) / closes[-(window + 1):-1]
            vol = float(np.nanstd(rets, ddof=1) * np.sqrt(252))
            vol_dict[sid] = max(vol, 0.001)
        else:
            vol_dict[sid] = 1.0
    return vol_dict


def compute_liquidity(price_data, stock_ids, current_date, window=20):
    """compute average turnover amount for each stock_id up to current_date.

    returns dict[stock_id, avg_turnover_amount] (floor 1.0 to avoid div-by-zero).
    """
    if price_data is None:
        return {}
    hist = price_data[pd.to_datetime(price_data["日期"]) < pd.to_datetime(current_date)]
    liq_dict = {}
    for sid in stock_ids:
        amounts = hist.loc[hist["股票代码"] == sid, "成交额"].values
        if len(amounts) >= window:
            liq_dict[sid] = max(float(np.nanmean(amounts[-window:])), 1.0)
        elif len(amounts) > 0:
            liq_dict[sid] = max(float(np.nanmean(amounts)), 1.0)
        else:
            liq_dict[sid] = 1.0
    return liq_dict


def compute_weights(predictions, top_k, weight_strategy="equal", strategy_params=None):
    """standalone: compute per-stock allocation weights for top_k stocks.

    strategy_params dict carries strategy-specific fields:
      - "temperature" (softmax)
      - "vol_dict"    (risk_parity, score_risk, score_risk_v1)

    returns dict[stock_id, weight] summing to 1.
    """
    sp = strategy_params or {}
    top_preds = predictions[:top_k]
    n = len(top_preds)
    if n == 0:
        return {}

    if weight_strategy == "equal":
        w = 1.0 / n
        return {p["stock_id"]: w for p in top_preds}

    if weight_strategy == "rank_linear":
        total = n * (n + 1) / 2
        weights = {}
        for p in top_preds:
            rank = p.get("rank", 1)
            w = (n + 1 - rank) / total
            weights[p["stock_id"]] = w
        return weights

    if weight_strategy == "softmax":
        T = sp.get("temperature", 1.0)
        scores = np.array([p["score"] for p in top_preds], dtype=np.float64)
        scores -= scores.max()
        exp_s = np.exp(scores / T)
        total = exp_s.sum()
        if total <= 0:
            w = 1.0 / n
            return {p["stock_id"]: w for p in top_preds}
        return {p["stock_id"]: float(exp_s[i] / total) for i, p in enumerate(top_preds)}

    if weight_strategy == "risk_parity":
        vol = sp.get("vol_dict", {})
        inv = {}
        for p in top_preds:
            iv = 1.0 / vol.get(p["stock_id"], 1.0)
            inv[p["stock_id"]] = iv
        total = sum(inv.values())
        if total <= 0:
            w = 1.0 / n
            return {p["stock_id"]: w for p in top_preds}
        return {sid: v / total for sid, v in inv.items()}

    if weight_strategy == "score_risk":
        vol = sp.get("vol_dict", {})
        scores = np.array([p["score"] for p in top_preds], dtype=np.float64)
        scores = scores - scores.min() + 1e-8
        vols = np.array([vol.get(p["stock_id"], 1.0) for p in top_preds], dtype=np.float64)
        risk_adj = scores / (vols * vols)
        total = risk_adj.sum()
        if total <= 0:
            w = 1.0 / n
            return {p["stock_id"]: w for p in top_preds}
        return {p["stock_id"]: float(risk_adj[i] / total) for i, p in enumerate(top_preds)}

    if weight_strategy == "score_risk_v1":
        vol = sp.get("vol_dict", {})
        scores = np.array([p["score"] for p in top_preds], dtype=np.float64)
        scores = scores - scores.min() + 1e-8
        vols = np.array([vol.get(p["stock_id"], 1.0) for p in top_preds], dtype=np.float64)
        risk_adj = scores / vols
        total = risk_adj.sum()
        if total <= 0:
            w = 1.0 / n
            return {p["stock_id"]: w for p in top_preds}
        return {p["stock_id"]: float(risk_adj[i] / total) for i, p in enumerate(top_preds)}

    if weight_strategy == "kelly":
        vol = sp.get("vol_dict", {})
        scores = np.array([p["score"] for p in top_preds], dtype=np.float64)
        smin, smax = scores.min(), scores.max()
        if smax - smin < 1e-12:
            w = 1.0 / n
            return {p["stock_id"]: w for p in top_preds}
        score_norm = (scores - smin) / (smax - smin)
        kelly = np.array([
            max(0, score_norm[i]) / (max(vol.get(p["stock_id"], 1.0), 0.001) ** 2 + 1e-12)
            for i, p in enumerate(top_preds)
        ])
        total = kelly.sum()
        if total <= 0:
            w = 1.0 / n
            return {p["stock_id"]: w for p in top_preds}
        return {p["stock_id"]: float(kelly[i] / total) for i, p in enumerate(top_preds)}

    if weight_strategy == "liquidity":
        liq = sp.get("liq_dict", {})
        vals = np.array([max(liq.get(p["stock_id"], 1.0), 1e-8) for p in top_preds], dtype=np.float64)
        total = vals.sum()
        if total <= 0:
            w = 1.0 / n
            return {p["stock_id"]: w for p in top_preds}
        return {p["stock_id"]: float(vals[i] / total) for i, p in enumerate(top_preds)}

    if weight_strategy == "score_power_2":
        scores = np.array([p["score"] for p in top_preds], dtype=np.float64)
        scores = scores - scores.min() + 1e-8
        scores = scores ** 2
        total = scores.sum()
        if total <= 0:
            w = 1.0 / n
            return {p["stock_id"]: w for p in top_preds}
        return {p["stock_id"]: float(scores[i] / total) for i, p in enumerate(top_preds)}

    if weight_strategy == "score_power_3":
        scores = np.array([p["score"] for p in top_preds], dtype=np.float64)
        scores = scores - scores.min() + 1e-8
        scores = scores ** 3
        total = scores.sum()
        if total <= 0:
            w = 1.0 / n
            return {p["stock_id"]: w for p in top_preds}
        return {p["stock_id"]: float(scores[i] / total) for i, p in enumerate(top_preds)}

    if weight_strategy == "dynamic_gap":
        scores = np.array([p["score"] for p in top_preds], dtype=np.float64)
        gaps = np.zeros_like(scores)
        for i in range(n - 1):
            gaps[i] = max(scores[i] - scores[i + 1], 0)
        gaps[-1] = max(scores[-1] - 0, 0)
        total_gap = gaps.sum()
        if total_gap <= 0:
            w = 1.0 / n
            return {p["stock_id"]: w for p in top_preds}
        return {p["stock_id"]: float(gaps[i] / total_gap) for i, p in enumerate(top_preds)}

    if weight_strategy == "concentrated":
        weights = {}
        top_weight = sp.get("top_weight", 0.5)
        remaining = 1.0 - top_weight
        rest = max(n - 1, 1)
        for i, p in enumerate(top_preds):
            if i == 0:
                weights[p["stock_id"]] = top_weight
            else:
                weights[p["stock_id"]] = remaining / rest
        return weights

    raise ValueError(f"unknown weight_strategy: {weight_strategy}")


class BacktestEngine:
    """回测引擎"""

    def __init__(
        self,
        initial_capital=1000000,
        commission=0.0003,
        slippage=0.001,
        top_k=5,
        position_pct=0.95,
        weight_strategy="equal",
        strategy_params=None,
        log=False,
        log_file=None,
        risk_manager_config=None,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.top_k = top_k
        self.position_pct = position_pct
        self.weight_strategy = weight_strategy
        self.strategy_params = strategy_params or {}
        self.log = log
        self.log_file = log_file

        self.cash = initial_capital
        self._risk_strategy = RiskStrategy.from_config(risk_manager_config or {})
        self._risk_enabled = (risk_manager_config or {}).get("enabled", False)
        self.positions = {}
        self.positions_prev = {}
        self.pre_rebalance_positions = {}
        self.equity_curve = []
        self.trades = []
        self.predictions_history = []
        self.skipped_trades = []
        self._prev_total_value = initial_capital

        self._log_fh = None
        if self.log_file:
            self._log_fh = open(self.log_file, "w", encoding="utf-8")

    def _write_log(self, msg: str):
        print(msg)
        if self._log_fh:
            self._log_fh.write(msg + "\n")
            self._log_fh.flush()

    def close_log(self):
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None

    def buy(self, stock, price, value, date):
        exec_price = price * (1 + self.slippage)
        shares = int(value / exec_price / 100) * 100
        if shares == 0:
            return
        cost = shares * exec_price * (1 + self.commission)
        if cost > self.cash:
            shares = int(self.cash / exec_price / (1 + self.commission) / 100) * 100
            if shares == 0:
                return
            cost = shares * exec_price * (1 + self.commission)
        self.cash -= cost

        if stock not in self.positions:
            self.positions[stock] = {"shares": 0, "cost": 0}

        old_shares = self.positions[stock]["shares"]
        old_cost = self.positions[stock]["cost"]
        new_shares = old_shares + shares
        new_cost = old_cost + cost

        self.positions[stock] = {"shares": new_shares, "cost": new_cost}

        trade_cost = round(cost - shares * price, 2)
        self.trades.append(
            {
                "date": date,
                "action": "买入",
                "stock": stock,
                "price": round(exec_price, 4),
                "shares": shares,
                "amount": cost,
                "trade_cost": trade_cost,
            }
        )

    def sell(self, stock, price, percent=1.0, date=None):
        if stock not in self.positions:
            return
        shares = int(self.positions[stock]["shares"] * percent / 100) * 100
        if shares == 0:
            return
        exec_price = price * (1 - self.slippage)
        revenue = shares * exec_price * (1 - self.commission)
        self.cash += revenue

        old_shares = self.positions[stock]["shares"]
        old_cost = self.positions[stock]["cost"]
        cost_of_sold = old_cost * (shares / old_shares) if old_shares > 0 else 0
        pnl = revenue - cost_of_sold

        trade_cost = round(shares * price - revenue, 2)
        self.trades.append(
            {
                "date": date,
                "action": "卖出",
                "stock": stock,
                "price": round(exec_price, 4),
                "shares": shares,
                "amount": revenue,
                "trade_cost": trade_cost,
                "pnl": round(pnl, 2),
            }
        )

        self.positions[stock]["shares"] -= shares
        self.positions[stock]["cost"] -= old_cost * (
            shares / old_shares
        )

        if self.positions[stock]["shares"] == 0:
            del self.positions[stock]

    def _can_buy(self, stock, price, high_limit_dict, paused_dict):
        if paused_dict.get(stock, 0) == 1:
            return False
        if price >= high_limit_dict.get(stock, float("inf")):
            return False
        return True

    def _can_sell(self, stock, price, low_limit_dict, paused_dict):
        if paused_dict.get(stock, 0) == 1:
            return False
        if price <= low_limit_dict.get(stock, 0):
            return False
        return True

    def get_total_value(self, price_dict):
        pos_value = sum(
            self.positions.get(st, {}).get("shares", 0) * price_dict.get(st, 0)
            for st in self.positions
        )
        return self.cash + pos_value

    def _compute_weights(self, predictions, top_k):
        return compute_weights(
            predictions, top_k,
            self.weight_strategy, self.strategy_params,
        )

    def run(
        self,
        dates: List,
        price_data: pd.DataFrame,
        predictions_func,
        rebalance_days: int = 5,
        first_rebalance_date=None,
        trade_mode: str = "open",
    ) -> List[Dict]:
        """运行回测
        trade_mode: "open"（开盘交易，用前日特征）或 "close"（收盘交易，用当日特征）
        """
        if first_rebalance_date is None:
            first_rebalance_date = dates[0]

        start_idx = 0
        for i, d in enumerate(dates):
            if d >= first_rebalance_date:
                start_idx = i
                break

        if self.log:
            self._write_log(f"\n{'=' * 50}")
            self._write_log(
                f"回测开始，第一个调仓日: {first_rebalance_date.strftime('%Y-%m-%d')}"
            )
            self._write_log(f"{'=' * 50}\n")

        _date_data_map = {d: grp for d, grp in price_data.groupby("日期")}

        for i in range(start_idx, len(dates)):
            current_date = dates[i]
            date_data = _date_data_map.get(current_date)
            price_dict = dict(zip(date_data["股票代码"], date_data["收盘"]))
            high_limit_dict = dict(zip(date_data["股票代码"], date_data["涨停价"]))
            low_limit_dict = dict(zip(date_data["股票代码"], date_data["跌停价"]))
            paused_dict = dict(zip(date_data["股票代码"], date_data["停牌"]))
            total_value = self.get_total_value(price_dict)
            risk_mult = 1.0
            self.equity_curve.append({"date": current_date, "total_value": total_value, "risk_multiplier": risk_mult, "stock_exposure": (total_value - self.cash) / total_value if total_value > 0 else 0})
            if self.log:
                position_pct = (
                    (total_value - self.cash) / total_value * 100
                    if total_value > 0
                    else 0
                )
                daily_return = (total_value / self._prev_total_value - 1) * 100
                total_return = (total_value / self.initial_capital - 1) * 100

                self._write_log(f"\n{'=' * 50}")
                self._write_log(f"{current_date.strftime('%Y-%m-%d')}")
                self._write_log(
                    f"账户总资产: {total_value:.2f} ({(total_value - self._prev_total_value):+.2f}, {daily_return:+.2f}%)"
                )
                self._write_log(f"持有现金: {self.cash:.2f}")
                self._write_log(f"仓位比例: {position_pct:.2f}%")
                self._write_log(f"累计收益率: {total_return:+.2f}%")

                self._write_log("持仓:")
                winning = 0
                total = 0
                for stock, pos_info in self.positions.items():
                    shares = pos_info["shares"]
                    cost = pos_info["cost"]
                    price = price_dict.get(stock, 0)
                    pos_value = shares * price
                    profit = pos_value - cost
                    profit_pct = (profit / cost * 100) if cost > 0 else 0
                    profit_str = (
                        f"盈{profit:+.2f}({profit_pct:+.2f}%)"
                        if profit >= 0
                        else f"亏{profit:+.2f}({profit_pct:+.2f}%)"
                    )

                    prev_shares = self.positions_prev.get(stock, {}).get("shares", 0)
                    prev_value = prev_shares * price_dict.get(stock, 0)
                    change = pos_value - prev_value
                    change_str = f" ({change:+.2f})" if change != 0 else ""

                    self._write_log(
                        f"  {stock}: {shares}股 @ {price:.4f} | {profit_str} | 市值:{pos_value:.2f}{change_str}"
                    )

                    if profit > 0:
                        winning += 1
                    total += 1

                if total > 0:
                    win_rate = winning / total * 100
                    self._write_log(f"持仓胜率: {winning}/{total} ({win_rate:.1f}%)")

                self._write_log(f"{'=' * 50}\n")

            self._prev_total_value = total_value
            self.positions_prev = {
                s: {"shares": p["shares"], "cost": p["cost"]}
                for s, p in self.positions.items()
            }

            if (i - start_idx) % rebalance_days == 0:
                self.pre_rebalance_positions = {
                    s: {"shares": p["shares"], "cost": p["cost"]}
                    for s, p in self.positions.items()
                }

                if self.log:
                    self._write_log(f"\n{'=' * 50}")
                    self._write_log(f"调仓日: {current_date.strftime('%Y-%m-%d')}")

                if trade_mode == "open":
                    if i > start_idx:
                        pred_date = dates[i - 1]
                    else:
                        all_dates_ts = sorted(price_data["日期"].unique())
                        try:
                            idx = all_dates_ts.index(current_date)
                            pred_date = all_dates_ts[idx - 1] if idx > 0 else current_date
                        except ValueError:
                            pred_date = current_date
                else:
                    pred_date = current_date

                predictions = predictions_func(pred_date)
                if predictions is None:
                    if self.log:
                        self._write_log("预测失败，跳过调仓")
                    continue

                _sp_snapshot = dict(self.strategy_params) if self.strategy_params else {}
                self.predictions_history.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "pred_date": pred_date.strftime("%Y-%m-%d"),
                    "predictions": predictions,
                    "strategy_params": _sp_snapshot,
                })

                pred_scores = [p["score"] for p in predictions]
                score_std = float(np.std(pred_scores)) if len(pred_scores) > 1 and float(np.std(pred_scores)) > 0 else 1.0
                cutoff_idx = min(self.top_k, len(predictions) - 1)
                score_cutoff = predictions[cutoff_idx]["score"]
                score_map = {p["stock_id"]: p["score"] for p in predictions}

                if self.log:
                    self._write_log(f"目标持仓 (Top {self.top_k}):")
                    for p in predictions[: self.top_k]:
                        self._write_log(
                            f"  {p['rank']}. {p['stock_id']} (score: {p['score']:.4f})"
                        )

                if self.log:
                    prev_holdings = set(self.positions.keys())
                    target_holdings = set(
                        [p["stock_id"] for p in predictions[: self.top_k]]
                    )
                    kept = prev_holdings & target_holdings
                    new_added = target_holdings - prev_holdings
                    exited = prev_holdings - target_holdings
                    self._write_log(
                        f"调仓变化: 新增{len(new_added)}只, 保留{len(kept)}只, 剔除{len(exited)}只"
                    )
                    if kept:
                        self._write_log(f"  保留: {list(kept)}")
                    if new_added:
                        self._write_log(f"  新增: {list(new_added)}")
                    if exited:
                        self._write_log(f"  剔除: {list(exited)}")

                for stock in list(self.positions.keys()):
                    if stock not in [p["stock_id"] for p in predictions[: self.top_k]]:
                        if not self._can_sell(stock, price_dict.get(stock, 0), low_limit_dict, paused_dict):
                            reason = "停牌" if paused_dict.get(stock, 0) == 1 else "跌停"
                            self.skipped_trades.append({"date": current_date, "action": "卖出", "stock": stock, "reason": reason})
                            if self.log:
                                self._write_log(f"  {stock}: {reason}，暂不可卖出，保留")
                            continue
                        self.sell(stock, price_dict.get(stock, 0), 1.0, current_date)
                        if self.log:
                            self._write_log(f"卖出: {stock}")

                # 按加权策略分配权重
                if self.weight_strategy in ("risk_parity", "score_risk", "score_risk_v1", "kelly"):
                    top_ids = [p["stock_id"] for p in predictions[: self.top_k]]
                    vol_dict = compute_volatility(
                        price_data, top_ids, current_date,
                        self.strategy_params.get("vol_window", 20),
                    )
                    self.strategy_params["vol_dict"] = vol_dict
                if self.weight_strategy == "liquidity":
                    top_ids = [p["stock_id"] for p in predictions[: self.top_k]]
                    liq_dict = compute_liquidity(
                        price_data, top_ids, current_date,
                        self.strategy_params.get("liq_window", 20),
                    )
                    self.strategy_params["liq_dict"] = liq_dict
                weights = self._compute_weights(predictions, self.top_k)

                # 风控: 计算仓位乘数
                if self._risk_enabled:
                    risk_mult = self._risk_strategy.get_multiplier(
                        current_date, price_data, self.positions, self.equity_curve,
                        date_data_map=_date_data_map,
                    )
                    if risk_mult < 1.0 and self.log:
                        self._write_log(
                            f"风控触发: multiplier={risk_mult:.2f} (策略={type(self._risk_strategy).__name__})"
                        )
                else:
                    risk_mult = 1.0
                self.equity_curve[-1]["risk_multiplier"] = risk_mult
                effective_pct = self.position_pct * risk_mult

                for pred in predictions[: self.top_k]:
                    stock = pred["stock_id"]
                    price = price_dict.get(stock, 0)
                    if price <= 0:
                        continue

                    exec_price_buy = price * (1 + self.slippage)
                    exec_price_sell = price * (1 - self.slippage)

                    target_value = total_value * effective_pct * weights.get(stock, 0)
                    target_shares = int(target_value / exec_price_buy / 100) * 100
                    if target_shares == 0:
                        continue

                    current_shares = self.positions.get(stock, {}).get("shares", 0)
                    diff_shares = target_shares - current_shares

                    if diff_shares > 0:
                        if not self._can_buy(stock, price, high_limit_dict, paused_dict):
                            reason = "停牌" if paused_dict.get(stock, 0) == 1 else "涨停"
                            self.skipped_trades.append({"date": current_date, "action": "买入", "stock": stock, "reason": reason})
                            if self.log:
                                self._write_log(f"  {stock}: {reason}，暂不可买入，跳过")
                            continue
                        buy_value = diff_shares * exec_price_buy
                        if buy_value > self.cash:
                            diff_shares = int(self.cash / exec_price_buy / 100) * 100
                            if diff_shares == 0:
                                continue
                            buy_value = diff_shares * exec_price_buy
                        self.buy(stock, price, buy_value, current_date)
                        if self.trades:
                            s = score_map[stock]
                            self.trades[-1]["score"] = float(s)
                            self.trades[-1]["advantage"] = round((s - score_cutoff) / score_std, 4)
                        if self.log:
                            new_shares = self.positions[stock]["shares"]
                            actual_value = new_shares * price
                            pct = actual_value / total_value * 100
                            self._write_log(
                                f"买入 {stock}: {diff_shares}股 @ {exec_price_buy:.4f} (目标: {target_shares}股, 占比: {pct:.2f}%)"
                            )

                    elif diff_shares < 0:
                        if not self._can_sell(stock, price, low_limit_dict, paused_dict):
                            reason = "停牌" if paused_dict.get(stock, 0) == 1 else "跌停"
                            self.skipped_trades.append({"date": current_date, "action": "卖出", "stock": stock, "reason": reason})
                            if self.log:
                                self._write_log(f"  {stock}: {reason}，暂不可卖出，跳过")
                            continue
                        sell_percent = (-diff_shares) / current_shares
                        old_shares = self.positions[stock]["shares"]
                        self.sell(stock, price, sell_percent, current_date)
                        sold_shares = old_shares - self.positions.get(stock, {}).get("shares", 0)
                        if self.log:
                            self._write_log(
                                f"卖出 {stock}: {sold_shares}股 @ {exec_price_sell:.4f} (目标: {target_shares}股)"
                            )

                    if self.log:
                        if stock in self.positions:
                            shares = self.positions[stock]["shares"]
                            cost = self.positions[stock]["cost"]
                            actual_value = shares * price
                            pct = actual_value / total_value * 100
                            self._write_log(
                                f"持有 {stock}: {shares}股 @ {price:.4f} (市值: {actual_value:.2f}, 占比: {pct:.2f}%)"
                            )

                if self.log:
                    self._write_log(f"调仓完成, 组合价值: {total_value:.2f}")
                    self._write_log(f"{'=' * 50}\n")

                self.equity_curve[-1]["stock_exposure"] = (total_value - self.cash) / total_value if total_value > 0 else 0

        return self.equity_curve


class ETFBacktester:
    """ETF回测主类"""

    # 类变量用于缓存数据
    _cached_data = {}
    _cached_features = {}

    def __init__(
        self,
        model_dir: str,
        data_path: str,
        device: str = "cuda",
        model_file: str = "best_model_sliding.pth",
        verbose: bool = False,
    ):
        """
        初始化回测器

        Args:
            model_dir: 模型目录路径
            data_path: ETF数据文件路径
            device: 设备 (cuda/cpu)
            model_file: 模型权重文件 (best_model.pth 或 best_model_sliding.pth)
            verbose: 是否打印初始化信息
        """
        self.model_dir = model_dir
        self.data_path = data_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_file = model_file
        self.verbose = verbose
        self.processed = None  # 初始化为None

        self._load_model()
        self._load_data()
        self._prepare_features()

    @classmethod
    def load_data_once(
        cls, data_path: str, scaler_path: str, feature_num: str,
        verbose: bool = False, store_unscaled: bool = False
    ):
        """
        预加载并缓存数据，供多次回测使用

        Args:
            data_path: 数据文件路径
            scaler_path: scaler文件路径
            feature_num: 特征编号
            verbose: 是否打印日志
            store_unscaled: 是否同时存储未缩放的特征（ML模型需要）
        """
        cache_key = f"{data_path}_{scaler_path}_{store_unscaled}"

        if cache_key in cls._cached_data:
            if verbose:
                print(f"使用缓存数据: {data_path}")
            return cls._cached_data[cache_key], cls._cached_features[cache_key]

        if verbose:
            print(f"加载并缓存数据: {data_path}")

        # 加载数据
        df = pd.read_csv(data_path)
        df["日期"] = pd.to_datetime(df["日期"])
        df["股票代码"] = df["股票代码"].astype(object).str.zfill(6)
        df = df.sort_values(["股票代码", "日期"]).reset_index(drop=True)

        # 加载scaler
        scaler = joblib.load(scaler_path)

        # 特征工程
        from train import feature_cloums_map, feature_engineer_func_map
        from tqdm import tqdm
        import multiprocessing as mp

        feature_engineer = feature_engineer_func_map[feature_num]
        features = feature_cloums_map[feature_num]

        groups = [group for _, group in df.groupby("股票代码", sort=False)]
        num_processes = 1  # Use single process to avoid OOM

        with mp.Pool(processes=num_processes) as pool:
            processed_list = list(
                tqdm(
                    pool.imap(feature_engineer, groups),
                    total=len(groups),
                    desc="特征工程",
                    disable=not verbose,
                )
            )

        processed = pd.concat(processed_list).reset_index(drop=True)

        stock_ids = sorted(processed["股票代码"].unique())
        stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
        processed["instrument"] = processed["股票代码"].map(stockid2idx)
        processed = processed.dropna(subset=["instrument"]).copy()
        processed["instrument"] = processed["instrument"].astype(np.int64)

        # 保存原始 instrument 再缩放（DL 模型需要整数索引而非缩放值）
        instrument_raw = processed["instrument"].copy()

        processed_cln = (
            processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

        # 保存未缩放版本（ML模型需要，tree-based不需要归一化）
        if store_unscaled:
            processed_raw = processed.copy()
            processed_raw[features] = processed_cln

        processed[features] = scaler.transform(processed_cln)
        processed["instrument"] = instrument_raw

        # 缓存
        cached_entry = {
            "df": df,
            "processed": processed,  # 添加处理后的数据
            "scaler": scaler,
            "stock_ids": stock_ids,
            "stockid2idx": stockid2idx,
        }
        if store_unscaled:
            cached_entry["processed_raw"] = processed_raw
        cls._cached_data[cache_key] = cached_entry
        cls._cached_features[cache_key] = {
            "features": features,
            "feature_engineer": feature_engineer,
        }

        if verbose:
            print(f"数据缓存完成: {len(df)} 条记录, {len(stock_ids)} 只股票")

        return cls._cached_data[cache_key], cls._cached_features[cache_key]

    @classmethod
    def from_cached_data(
        cls,
        model_dir: str,
        cached_data: dict,
        cached_features: dict,
        device: str = "cuda",
        model_file: str = "best_model_sliding.pth",
        verbose: bool = False,
    ):
        """
        使用缓存数据创建回测器

        Args:
            model_dir: 模型目录路径
            cached_data: 缓存的数据字典
            cached_features: 缓存的特征信息
            device: 设备
            model_file: 模型文件
            verbose: 是否打印日志
        """
        instance = cls.__new__(cls)
        instance.model_dir = model_dir
        instance.data_path = None
        instance.device = torch.device(device if torch.cuda.is_available() else "cpu")
        instance.model_file = model_file
        instance.verbose = verbose

        # 使用缓存数据
        instance.df = cached_data["df"]
        instance.processed = cached_data["processed"]  # 直接使用已处理的特征数据
        instance.processed_raw = cached_data.get("processed_raw")  # 未缩放版本（ML用）
        instance.scaler = cached_data["scaler"]
        instance.stock_ids = cached_data["stock_ids"]
        instance.stockid2idx = cached_data["stockid2idx"]
        instance.features = cached_features["features"]
        instance.feature_engineer = cached_features["feature_engineer"]

        instance._load_model()
        return instance

    def _load_model(self):
        """加载模型和配置"""
        config_path = f"{self.model_dir}/config.json"
        if not os.path.exists(config_path):
            parent = os.path.dirname(os.path.normpath(self.model_dir))
            parent_cfg = os.path.join(parent, "config.json")
            if os.path.exists(parent_cfg):
                config_path = parent_cfg
        model_path = f"{self.model_dir}/{self.model_file}"

        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.is_ml_model = self.model_file.endswith(".pkl")

        if self.is_ml_model:
            self.model = joblib.load(model_path)
            self.num_stocks = len(self.stock_ids) if self.stock_ids else 0
            self.seq_length = 1  # ML models: no sequence, date-level features
            if self.verbose:
                print(f"ML模型加载完成: {self.model_dir}/{self.model_file}")
        else:
            from config import get_model_config

            model_type = self.config.get("model_type", "transformer")
            model_defaults = get_model_config(model_type)
            model_defaults.update(self.config)
            self.config = model_defaults

            if self.df is not None:
                num_stocks = self.df["股票代码"].nunique()
            else:
                num_stocks = len(self.stock_ids)
            input_dim = len(self.features)

            from model import create_model

            self.model = create_model(
                self.config["model_type"], input_dim, self.config, num_stocks
            )
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()

            self.num_stocks = num_stocks
            self.seq_length = self.config["sequence_length"]

            if self.verbose:
                print(f"模型加载完成: {self.model_dir}/{self.model_file}")
                print(f"股票数量: {num_stocks}, 特征数量: {input_dim}")

    def _load_data(self):
        """加载数据 - 已弃用，使用缓存机制"""
        pass

    def _prepare_features(self):
        """准备特征数据 - 已弃用，直接使用缓存的processed数据"""
        if self.verbose:
            print(f"特征准备完成 (使用缓存): {self.processed.shape}")

    def _get_predictions(self, target_date) -> Optional[List[Dict]]:
        """获取模型预测"""
        if self.is_ml_model:
            return self._get_ml_predictions(target_date)

        # --- DL model: sequence-based prediction ---
        all_dates_sorted = sorted(self.processed["日期"].unique())
        try:
            target_idx = all_dates_sorted.index(target_date)
        except:
            return None
        if target_idx < self.seq_length:
            return None

        sequences = []
        valid_stock_ids = []

        for stock_id in self.stock_ids:
            stock_history = (
                self.processed[
                    (self.processed["股票代码"] == stock_id)
                    & (self.processed["日期"] <= target_date)
                ]
                .sort_values("日期")
                .tail(self.seq_length)
            )
            if len(stock_history) == self.seq_length:
                sequences.append(stock_history[self.features].values.astype(np.float32))
                valid_stock_ids.append(stock_id)

        if len(sequences) == 0:
            return None

        sequences_np = np.asarray(sequences, dtype=np.float32)

        with torch.no_grad():
            x = torch.from_numpy(sequences_np).unsqueeze(0).to(self.device)
            scores = self.model(x).squeeze(0).detach().cpu().numpy()

        order = np.argsort(scores)[::-1]
        predictions = []
        for rank, i in enumerate(order):
            predictions.append(
                {"rank": rank + 1, "stock_id": valid_stock_ids[i], "score": scores[i]}
            )

        return predictions

    def _get_ml_predictions(self, target_date) -> Optional[List[Dict]]:
        """ML model prediction: cross-sectional features per date, no sequence."""
        if self.processed_raw is None:
            print(f"WARNING: ML model needs unscaled data. "
                  f"Call load_data_once with store_unscaled=True")
            return None
        if not hasattr(self, "_ml_feat_names"):
            self._compute_ml_features(self.processed_raw)

        day_data = self.processed_raw[self.processed_raw["日期"] == target_date].sort_values("股票代码")
        if day_data.empty:
            return None

        X = day_data[self._ml_feat_names].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        import xgboost as xgb
        import lightgbm as lgb

        if isinstance(self.model, xgb.Booster):
            y_pred = self.model.predict(xgb.DMatrix(X))
        elif isinstance(self.model, lgb.Booster):
            y_pred = self.model.predict(X, predict_disable_shape_check=True)
        else:
            y_pred = self.model.predict(X)

        order = np.argsort(y_pred)[::-1]
        valid_ids = day_data["股票代码"].values
        predictions = []
        for rank, i in enumerate(order):
            predictions.append({
                "rank": rank + 1,
                "stock_id": valid_ids[i],
                "score": float(y_pred[i]),
            })
        return predictions

    def _compute_ml_features(self, df):
        """Compute cross-sectional features for ML models (once)."""
        from train_ml import ml_feature_engineering
        raw_cols = {"股票代码", "日期", "label", "instrument"}
        base = [c for c in self.features if c in df.columns and c not in raw_cols]
        new_cols = ml_feature_engineering(df, base, momentum=True)
        all_feats = base + new_cols
        self._ml_feat_names = [c for c in all_feats if c not in raw_cols]

    def run(
        self,
        start_date: str,
        end_date: str,
        top_k: int = 5,
        rebalance_days: int = 5,
        position_pct: float = 0.95,
        weight_strategy: str = "equal",
        strategy_params: dict = None,
        initial_capital: float = 1000000,
        commission: float = 0.0003,
        slippage: float = 0.001,
        first_rebalance_date: str = None,
        trade_mode: str = "open",
        log: bool = False,
    ) -> BacktestResult:
        """
        运行回测

        Args:
            start_date: 回测开始日期 (YYYY-MM-DD)
            end_date: 回测结束日期 (YYYY-MM-DD)
            top_k: 持仓股票数量
            rebalance_days: 调仓频率(天)
            position_pct: 仓位比例
            initial_capital: 初始资金
            commission: 手续费率
            first_rebalance_date: 首次调仓日期
            trade_mode: "open"（开盘交易，用前日收盘特征）或 "close"（收盘交易，用当日收盘特征）
            log: 是否打印交易过程日志

        Returns:
            BacktestResult: 回测结果
        """
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        all_dates = sorted(self.df["日期"].unique())
        backtest_dates = [d for d in all_dates if start_ts <= d < end_ts]

        if first_rebalance_date:
            first_reb_ts = pd.Timestamp(first_rebalance_date)
        else:
            first_reb_ts = backtest_dates[0] if len(backtest_dates) > 0 else None

        # 获取HS300数据 (使用510300华夏沪深300ETF作为代理)
        hs300_code = "510300.XSHG"
        hs300_data = self.df[self.df["股票代码"] == hs300_code][["日期", "收盘"]].copy()
        hs300_data = hs300_data.rename(columns={"日期": "date", "收盘": "close"})
        hs300_data["date"] = pd.to_datetime(hs300_data["date"])
        hs300_data = hs300_data[
            (hs300_data["date"] >= start_ts) & (hs300_data["date"] < end_ts)
        ].copy()

        def predictions_func(date):
            return self._get_predictions(date)

        log_file = None
        if log:
            log_file = "/home/linuxyl/THU-BDC2026/temp/backtest.log"

        engine = BacktestEngine(
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
            top_k=top_k,
            position_pct=position_pct,
            weight_strategy=weight_strategy,
            strategy_params=strategy_params,
            log=log,
            log_file=log_file,
        )

        results = engine.run(
            dates=backtest_dates,
            price_data=self.df,
            predictions_func=predictions_func,
            rebalance_days=rebalance_days,
            first_rebalance_date=first_reb_ts,
            trade_mode=trade_mode,
        )

        # 计算策略收益
        equity_df = pd.DataFrame(results)
        equity_df["date"] = pd.to_datetime(equity_df["date"])
        strategy_return = (
            equity_df["total_value"].iloc[-1] / initial_capital - 1
        ) * 100

        # 计算HS300收益
        if len(hs300_data) > 0:
            hs300_data = hs300_data.sort_values("date").reset_index(drop=True)
            hs300_return = (
                hs300_data["close"].iloc[-1] / hs300_data["close"].iloc[0] - 1
            ) * 100
        else:
            hs300_return = 0.0

        excess_return = strategy_return - hs300_return

        # 计算最大回撤
        cumulative = (equity_df["total_value"] / initial_capital).values
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max * 100
        max_drawdown = abs(drawdown.min())

        if max_drawdown > 0:
            end_idx = int(np.argmin(drawdown))

            # 找到回撤开始的索引
            max_idx = int(np.argmax(running_max[: end_idx + 1]))
            drawdown_days = end_idx - max_idx

            # 计算恢复天数
            recovery_idx = None
            for idx in range(end_idx + 1, len(cumulative)):
                if cumulative[idx] >= cumulative[max_idx]:
                    recovery_idx = idx
                    break

            recovery_days = None
            if recovery_idx is not None:
                recovery_days = int(recovery_idx - end_idx)

            recovered = recovery_idx is not None
        else:
            drawdown_days = 0
            recovery_days = None
            recovered = True

        rebalance_stats = _compute_rebalance_stats(
            equity_df, engine.predictions_history, initial_capital,
        )

        result = BacktestResult(
            start_date=start_date,
            end_date=end_date,
            strategy_return=strategy_return,
            hs300_return=hs300_return,
            excess_return=excess_return,
            max_drawdown=max_drawdown,
            drawdown_days=int(drawdown_days),
            recovered=recovered,
            recovery_days=recovery_days,
            log_file=log_file,
            equity_curve=equity_df,
            hs300_data=hs300_data,
            rebalance_stats=rebalance_stats,
        )

        if engine._log_fh:
            engine.close_log()

        self._last_engine = engine

        return result


    def generate_predictions_dict(
        self,
        start_date: str,
        end_date: str,
        top_k: int = 5,
        rebalance_days: int = None,
        first_rebalance_date: str = None,
    ) -> Dict[str, List[Dict]]:
        """为回测期间生成预测，返回 {date_str: [predictions]}

        若指定 rebalance_days，则仅生成可能被回测引擎查询的日期预测
        （同时覆盖 close 和 open 两种模式），大幅减少无用预测。
        """
        all_dates = sorted(self.df["日期"].unique())
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        backtest_dates = [d for d in all_dates if start_ts <= d < end_ts]

        if rebalance_days is not None:
            if first_rebalance_date:
                first_reb_ts = pd.Timestamp(first_rebalance_date)
            else:
                first_reb_ts = backtest_dates[0]
            start_idx = next(
                (i for i, d in enumerate(backtest_dates) if d >= first_reb_ts),
                0,
            )
            # 调仓日索引集合
            rebal_indices = set(range(start_idx, len(backtest_dates), rebalance_days))

            # 需要的预测日期 = 调仓日本身(close) + 前一交易日(open)
            need_dates = set()
            for i in rebal_indices:
                need_dates.add(backtest_dates[i])                     # close
                if i > 0:
                    need_dates.add(backtest_dates[i - 1])             # open
                elif i == 0:
                    need_dates.add(backtest_dates[0])                 # open fallback
            # 首个调仓日的前一天（open 模式 seed date 可能被 _get_predictions 需要）
            if start_idx > 0:
                need_dates.add(backtest_dates[start_idx - 1])
        else:
            need_dates = set(backtest_dates)
            prev_idx = all_dates.index(backtest_dates[0]) - 1
            if prev_idx >= 0:
                need_dates.add(all_dates[prev_idx])

        result = {}
        for d in sorted(need_dates):
            preds = self._get_predictions(d)
            if preds:
                result[d.strftime("%Y-%m-%d")] = preds
        return result


def _compute_rebalance_stats(equity_df: pd.DataFrame, predictions_history: List[Dict], initial_capital: float) -> Dict:
    """从 equity_curve + predictions_history 计算每次调仓收益和胜率，返回 dict"""
    stats = {"returns": [], "total": 0, "wins": 0, "win_rate": 0.0, "avg_return": 0.0}
    if not predictions_history:
        return stats

    equity = equity_df.copy()
    equity["date"] = pd.to_datetime(equity["date"])
    reb_dates = [pd.Timestamp(p["date"]) for p in predictions_history]
    returns = []

    for i in range(1, len(reb_dates)):
        d0, d1 = reb_dates[i - 1], reb_dates[i]
        v0 = equity.loc[equity["date"] == d0, "total_value"]
        v1 = equity.loc[equity["date"] == d1, "total_value"]
        if len(v0) == 0 or len(v1) == 0:
            continue
        ret = (v1.iloc[0] / v0.iloc[0] - 1) * 100
        returns.append(round(ret, 2))

    total = len(returns)
    wins = sum(1 for r in returns if r > 0)
    stats = {
        "returns": returns,
        "total": total,
        "wins": wins,
        "win_rate": round(wins / total * 100, 1) if total > 0 else 0.0,
        "avg_return": round(sum(returns) / total, 2) if total > 0 else 0.0,
    }
    return stats


def run_backtest_from_predictions(
    predictions_dict: Dict[str, List[Dict]],
    data_path: str,
    start_date: str,
    end_date: str,
    top_k: int = 5,
    rebalance_days: int = 5,
    position_pct: float = 0.95,
    initial_capital: float = 1000000,
    commission: float = 0.0003,
    slippage: float = 0.001,
    weight_strategy: str = "equal",
    strategy_params: dict = None,
    first_rebalance_date: str = None,
    trade_mode: str = "open",
    log: bool = False,
    verbose: bool = False,
    risk_manager_config: dict = None,
) -> BacktestResult:
    """
    从已保存的预测信号运行回测（无需加载模型，速度快）

    Args:
        predictions_dict: {date_str: [{rank, stock_id, score}, ...]}
        data_path: ETF数据文件路径
        ...其余参数同 run_backtest
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    raw_df = pd.read_csv(data_path, dtype={"股票代码": str})
    raw_df["股票代码"] = raw_df["股票代码"].astype(object).str.zfill(6)
    raw_df["日期"] = pd.to_datetime(raw_df["日期"])
    all_dates = sorted(raw_df["日期"].unique())
    backtest_dates = [d for d in all_dates if start_ts <= d < end_ts]

    if first_rebalance_date:
        first_reb_ts = pd.Timestamp(first_rebalance_date)
    else:
        first_reb_ts = backtest_dates[0] if backtest_dates else None

    hs300_code = "510300.XSHG"
    hs300_data = raw_df[raw_df["股票代码"] == hs300_code][["日期", "收盘"]].copy()
    hs300_data = hs300_data.rename(columns={"日期": "date", "收盘": "close"})
    hs300_data["date"] = pd.to_datetime(hs300_data["date"])
    hs300_data = hs300_data[(hs300_data["date"] >= start_ts) & (hs300_data["date"] < end_ts)].copy()

    def pred_func(date):
        d_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
        return predictions_dict.get(d_str)

    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage,
        top_k=top_k,
        position_pct=position_pct,
        weight_strategy=weight_strategy,
        strategy_params=strategy_params,
        log=log,
        risk_manager_config=risk_manager_config,
    )

    results = engine.run(
        dates=backtest_dates,
        price_data=raw_df,
        predictions_func=pred_func,
        rebalance_days=rebalance_days,
        first_rebalance_date=first_reb_ts,
        trade_mode=trade_mode,
    )

    equity_df = pd.DataFrame(results)
    equity_df["date"] = pd.to_datetime(equity_df["date"])
    strategy_return = (equity_df["total_value"].iloc[-1] / initial_capital - 1) * 100

    if len(hs300_data) > 0:
        hs300_data = hs300_data.sort_values("date").reset_index(drop=True)
        hs300_start = hs300_data["close"].iloc[0]
        hs300_end = hs300_data["close"].iloc[-1]
        hs300_return = (hs300_end / hs300_start - 1) * 100
    else:
        hs300_return = 0

    excess_return = strategy_return - hs300_return

    cumulative = (equity_df["total_value"] / initial_capital).values
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max * 100
    max_dd = abs(drawdown.min())
    if max_dd > 0:
        end_idx = int(np.argmin(drawdown))
        max_idx = int(np.argmax(running_max[: end_idx + 1]))
        dd_days = end_idx - max_idx
        recovery_idx = None
        for idx in range(end_idx + 1, len(cumulative)):
            if cumulative[idx] >= cumulative[max_idx]:
                recovery_idx = idx
                break
        recovery_days = int(recovery_idx - end_idx) if recovery_idx is not None else None
        recovered = recovery_idx is not None
    else:
        dd_days = 0
        recovery_days = None
        recovered = True

    rebalance_stats = _compute_rebalance_stats(
        equity_df, engine.predictions_history, initial_capital,
    )

    return BacktestResult(
        start_date=start_date,
        end_date=end_date,
        strategy_return=round(strategy_return, 2),
        hs300_return=round(hs300_return, 2),
        excess_return=round(excess_return, 2),
        max_drawdown=round(max_dd, 2),
        drawdown_days=dd_days,
        recovered=recovered,
        recovery_days=recovery_days,
        equity_curve=equity_df,
        hs300_data=hs300_data,
        rebalance_stats=rebalance_stats,
        trades=engine.trades,
    )


def run_backtest(
    model_dir: str,
    data_path: str,
    start_date: str,
    end_date: str,
    top_k: int = 5,
    rebalance_days: int = 5,
    position_pct: float = 0.95,
    weight_strategy: str = "equal",
    strategy_params: dict = None,
    initial_capital: float = 1000000,
    commission: float = 0.0003,
    slippage: float = 0.001,
    first_rebalance_date: str = None,
    trade_mode: str = "open",
    device: str = "cuda",
    model_file: str = "best_model_sliding.pth",
    log: bool = False,
    verbose: bool = False,
) -> BacktestResult:
    """
    运行ETF回测的便捷函数

    Args:
        model_dir: 模型目录路径
        data_path: ETF数据文件路径
        start_date: 回测开始日期 (YYYY-MM-DD)
        end_date: 回测结束日期 (YYYY-MM-DD)
        top_k: 持仓股票数量
        rebalance_days: 调仓频率(天)
        position_pct: 仓位比例
        initial_capital: 初始资金
        commission: 手续费率
        first_rebalance_date: 首次调仓日期
        trade_mode: "open"（开盘交易，用前日收盘特征）或 "close"（收盘交易，用当日收盘特征）
        device: 设备 (cuda/cpu)
        model_file: 模型权重文件 (best_model.pth 或 best_model_sliding.pth)
        log: 是否打印交易过程日志

    Returns:
        BacktestResult: 回测结果

    Example:
        >>> result = run_backtest(
        ...     model_dir="./model/search_itransformer_60_39/exp_40",
        ...     data_path="./data/data_41.csv",
        ...     start_date="2025-01-02",
        ...     end_date="2025-12-31",
        ...     first_rebalance_date="2025-01-02",
        ...     model_file="best_model.pth",
        ...     verbose=True
        ... )
        >>> result.print_summary()
    """
    # 首先预加载数据
    scaler_path = f"{model_dir}/scaler.pkl"
    import json

    with open(f"{model_dir}/config.json") as f:
        config = json.load(f)

    cached_data, cached_features = ETFBacktester.load_data_once(
        data_path=data_path,
        scaler_path=scaler_path,
        feature_num=config["feature_num"],
        verbose=verbose,
    )

    # 多次回测不同模型
    backtester = ETFBacktester.from_cached_data(
        model_dir=model_dir,
        cached_data=cached_data,
        cached_features=cached_features,
        device=device,
        model_file=model_file,
        verbose=False,
    )

    # 保存结果
    result = backtester.run(
        start_date=start_date,
        end_date=end_date,
        top_k=top_k,
        rebalance_days=rebalance_days,
        position_pct=position_pct,
        weight_strategy=weight_strategy,
        strategy_params=strategy_params,
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage,
        first_rebalance_date=first_rebalance_date,
        trade_mode=trade_mode,
        log=log,
    )

    # 显式清理GPU内存
    del backtester.model
    del backtester
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    return result


if __name__ == "__main__":
    import sys

    # 默认参数
    model_dir = "model/search_itransformer_74/exp_53"
    model_file = "best_model_sliding.pth"
    data_path = "data/data_74.csv"
    topk = 5
    start_date = "2025-01-02"
    end_date = "2025-12-31"
    first_rebalance_date = "2025-01-02"

    # 从命令行参数读取
    if len(sys.argv) >= 4:
        model_dir = sys.argv[1]
        data_path = sys.argv[2]
        start_date = sys.argv[3]
        end_date = sys.argv[4]
        if len(sys.argv) >= 6:
            first_rebalance_date = sys.argv[5]

    result = run_backtest(
        model_dir=model_dir,
        data_path=data_path,
        start_date=start_date,
        end_date=end_date,
        top_k=topk,
        first_rebalance_date=first_rebalance_date,
        log=True,
        model_file=model_file,
        verbose=True,
    )
    result.print_summary()
    result.plot()
