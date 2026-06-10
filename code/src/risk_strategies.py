import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class RiskStrategy(ABC):
    """止损策略基类"""

    @abstractmethod
    def get_multiplier(self, current_date, price_data, positions, equity_curve, date_data_map=None):
        """返回 0.0~1.0 仓位乘数"""
        pass

    @classmethod
    def from_config(cls, config: dict) -> "RiskStrategy":
        strategy = config.get("strategy", "none")
        params = config.get("params", {})
        registry = {
            "none": NoRiskControl,
            "market_breadth": MarketBreadthStrategy,
            "volatility_target": VolatilityTargetStrategy,
            "trend_filter": TrendFilterStrategy,
            "drawdown_stop": DrawdownStopStrategy,
        }
        klass = registry.get(strategy, NoRiskControl)
        return klass(params)


class NoRiskControl(RiskStrategy):
    def __init__(self, params=None):
        pass

    def get_multiplier(self, current_date, price_data, positions, equity_curve, date_data_map=None):
        return 1.0


class MarketBreadthStrategy(RiskStrategy):
    """正收益ETF占比低于阈值时减仓/空仓"""

    def __init__(self, params=None):
        p = params or {}
        self.lookback_days = p.get("lookback_days", 20)
        self.high_threshold = p.get("high_threshold", 0.30)
        self.low_threshold = p.get("low_threshold", 0.10)

    def get_multiplier(self, current_date, price_data, positions, equity_curve, date_data_map=None):
        if date_data_map is not None:
            all_dates = sorted(d for d in date_data_map if d <= current_date)
            if len(all_dates) < self.lookback_days + 1:
                return 1.0
            start_date = all_dates[-(self.lookback_days + 1)]
            start_prices = date_data_map[start_date].set_index("股票代码")["收盘"]
            end_prices = date_data_map[current_date].set_index("股票代码")["收盘"]
        else:
            data = price_data[price_data["日期"] <= current_date]
            if len(data) < self.lookback_days + 1:
                return 1.0
            all_dates = sorted(data["日期"].unique())
            if len(all_dates) < self.lookback_days + 1:
                return 1.0
            start_date = all_dates[-(self.lookback_days + 1)]
            end_date = all_dates[-1]
            start_prices = data[data["日期"] == start_date].set_index("股票代码")["收盘"]
            end_prices = data[data["日期"] == end_date].set_index("股票代码")["收盘"]
        if len(start_prices) < 5 or len(end_prices) < 5:
            return 1.0
        returns = (end_prices - start_prices) / start_prices
        pos_ratio = (returns > 0).mean()
        if pos_ratio >= self.high_threshold:
            return 1.0
        if pos_ratio <= self.low_threshold:
            return 0.0
        return (pos_ratio - self.low_threshold) / (self.high_threshold - self.low_threshold)


class VolatilityTargetStrategy(RiskStrategy):
    """全市场波动率过高时缩仓"""

    def __init__(self, params=None):
        p = params or {}
        self.lookback_days = p.get("lookback_days", 20)
        self.n_std = p.get("n_std", 1.0)
        self.max_std = p.get("max_std", 2.0)

    def get_multiplier(self, current_date, price_data, positions, equity_curve, date_data_map=None):
        data = price_data[price_data["日期"] <= current_date]
        if len(data) < self.lookback_days + 2:
            return 1.0
        all_dates = sorted(data["日期"].unique())
        if len(all_dates) < self.lookback_days + 2:
            return 1.0
        lookback_end = all_dates[-1]
        lookback_start_idx = max(0, len(all_dates) - self.lookback_days - 1)
        lookback_start = all_dates[lookback_start_idx]
        window = data[(data["日期"] >= lookback_start) & (data["日期"] <= lookback_end)]
        if len(window) < 5:
            return 1.0
        pivoted = window.pivot_table(index="日期", columns="股票代码", values="收盘")
        daily_rets = pivoted.pct_change().dropna(how="all")
        daily_vols = daily_rets.std()
        vols = daily_vols.dropna().values
        if len(vols) < 5:
            return 1.0
        med_vol = np.median(vols)
        vol_mean = np.mean(vols)
        vol_std = np.std(vols)
        if vol_std == 0:
            return 1.0
        z = (med_vol - vol_mean) / vol_std
        if z <= self.n_std:
            return 1.0
        if z >= self.max_std:
            return 0.0
        ratio = (self.max_std - z) / (self.max_std - self.n_std)
        return ratio


class TrendFilterStrategy(RiskStrategy):
    """全市场中位数均线比率低于阈值时空仓"""

    def __init__(self, params=None):
        p = params or {}
        self.fast = p.get("fast", 20)
        self.slow = p.get("slow", 60)
        self.entry_threshold = p.get("entry_threshold", 1.0)
        self.exit_threshold = p.get("exit_threshold", 0.98)

    def get_multiplier(self, current_date, price_data, positions, equity_curve, date_data_map=None):
        data = price_data[price_data["日期"] <= current_date]
        if len(data) < self.slow:
            return 1.0
        all_dates = sorted(data["日期"].unique())
        if len(all_dates) < self.slow:
            return 1.0
        start_idx = max(0, len(all_dates) - self.slow)
        window = data[data["日期"] >= all_dates[start_idx]]
        pivoted = window.pivot_table(index="日期", columns="股票代码", values="收盘")
        if len(pivoted) < self.slow:
            return 1.0
        ma_fast = pivoted.rolling(self.fast, min_periods=1).mean()
        ma_slow = pivoted.rolling(self.slow, min_periods=1).mean()
        ratio = (ma_fast / ma_slow).iloc[-1].dropna()
        if len(ratio) < 5:
            return 1.0
        med_ratio = ratio.median()
        if med_ratio >= self.entry_threshold:
            return 1.0
        if med_ratio <= self.exit_threshold:
            return 0.0
        return (med_ratio - self.exit_threshold) / (self.entry_threshold - self.exit_threshold)


class DrawdownStopStrategy(RiskStrategy):
    """组合回撤超过阈值时缩仓"""

    def __init__(self, params=None):
        p = params or {}
        self.dd_low = p.get("dd_low", 0.05)
        self.dd_high = p.get("dd_high", 0.10)

    def get_multiplier(self, current_date, price_data, positions, equity_curve, date_data_map=None):
        if not equity_curve:
            return 1.0
        values = [e["total_value"] for e in equity_curve]
        peak = max(values)
        current = values[-1]
        dd = (peak - current) / peak if peak > 0 else 0
        if dd <= self.dd_low:
            return 1.0
        if dd >= self.dd_high:
            return 0.0
        return (self.dd_high - dd) / (self.dd_high - self.dd_low)
