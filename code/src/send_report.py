"""
发送每日测评报告邮件
读取 output/latest_report.json 和 output/equity_curves.png
通过 SMTP 发送 HTML 格式邮件
"""

import os
import re
import sys
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_PATH = PROJECT_ROOT / "output" / "latest_report.json"
CHART_PATH = PROJECT_ROOT / "output" / "equity_curves.png"

# 邮件配置 (从环境变量读取)
os.environ['SMTP_USER'] = '3759608757@qq.com'
os.environ['SMTP_PASSWORD'] = 'gsiqpfqjjkvwcdfg'
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.qq.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "465"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
EMAIL_FROM = os.environ.get("EMAIL_FROM", SMTP_USER)
EMAIL_TO = os.environ.get("EMAIL_TO", "1280745039@qq.com")


TEMPLATE_PATH = PROJECT_ROOT / "code" / "src" / "report_template.html"

ETF_LIST_PATH = PROJECT_ROOT / "etf_data" / "etf_list_before_2022_74.csv"


def _add_window(metrics, key, values, n):
    """向 metrics 补充 N 日窗口数据"""
    if len(values) < n + 1:
        return
    ret = (values[-1] / values[-n - 1] - 1) * 100
    ann = ((1 + ret / 100) ** (252 / n) - 1) * 100 if n > 0 else 0
    sub = values[-n:]
    wins = sum(1 for i in range(1, len(sub)) if sub[i] > sub[i - 1])
    win_rate = wins / (len(sub) - 1) if len(sub) > 1 else 0
    dd = _compute_max_drawdown(sub)
    metrics[key] = {
        "strategy_return_pct": round(ret, 2),
        "annualized_return_pct": round(ann, 2),
        "daily_win_rate": round(win_rate, 4),
        "max_drawdown_pct": round(dd, 2),
        "total_days": n,
    }


def _compute_max_drawdown(values):
    peak = values[0]
    dd = 0
    for v in values:
        if v > peak:
            peak = v
        dd = min(dd, (v - peak) / peak)
    return abs(dd) * 100


def _load_etf_names():
    """加载 ETF 代码→名称映射"""
    if not ETF_LIST_PATH.exists():
        return {}
    import csv
    mapping = {}
    with open(ETF_LIST_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            code = row.get("代码", "").strip()
            name = row.get("名称", "").strip()
            if code and name:
                mapping[code] = name
    return mapping


def _xueqiu_url(stock_id):
    code = stock_id.split(".")[0]
    exchange = stock_id.split(".")[1] if "." in stock_id else ""
    prefix = "SH" if exchange == "XSHG" else "SZ"
    return f"https://xueqiu.com/S/{prefix}{code}"


def _pct(v, suffix="%"):
    return f"{v:+.2f}{suffix}" if isinstance(v, (int, float)) else str(v)


def _fmt_advantage(v):
    if v is None:
        return "-"
    if isinstance(v, int):
        return f"{v:+d}"
    return f"{v:+.2f}"


def _load_template():
    if TEMPLATE_PATH.exists():
        return TEMPLATE_PATH.read_text(encoding="utf-8")
    return ""


def _build_model_stats_table(sequences):
    rows = ""
    for key, seq in sequences.items():
        ms = seq.get("model_stats", {})
        m = seq.get("metrics", {})
        if not ms:
            continue
        display = key.replace("search_", "").replace("_exp_", " ")
        ret = ms.get("reb_pnl_pct", ms["last_trade_return_pct"])
        ret_clr = "#cc0000" if ret >= 0 else "#009900"
        l3 = ms.get("last_3_reb_avg_pct", ms["last_3_avg_return_pct"])
        l3_clr = "#cc0000" if l3 >= 0 else "#009900"
        l3w = ms.get("last_3_win_rate_pct", 0)
        ta = ms["total_avg_return_pct"]
        ta_clr = "#cc0000" if ta >= 0 else "#009900"
        sr = m.get("strategy_return_pct", 0)
        sr_clr = "#cc0000" if sr >= 0 else "#009900"
        er = m.get("excess_return_pct", 0)
        er_clr = "#cc0000" if er >= 0 else "#009900"
        ic = m.get("rank_ic")
        ic_str = f"{ic:+.4f}" if ic is not None else "—"
        ic_clr = "#cc0000" if ic is not None and ic >= 0 else "#009900" if ic is not None else "#999"
        ndcg = m.get("ndcg")
        ndcg_str = f"{ndcg:.4f}" if ndcg is not None else "—"
        mrr = m.get("mrr")
        mrr_str = f"{mrr:.4f}" if mrr is not None else "—"
        ks_p = m.get("ks_p")
        ks_p_str = f"{ks_p:.4f}" if ks_p is not None else "—"
        calmar = m.get("calmar_ratio", 0)
        sortino = m.get("sortino_ratio", 0)
        rows += f"""
        <tr>
            <td style="font-size:11px;color:#555;">{display}</td>
            <td style="text-align:right;font-weight:bold;color:{ret_clr};">{ret:+.2f}%</td>
            <td style="text-align:right;font-weight:bold;color:{l3_clr};">{l3:+.2f}%</td>
            <td style="text-align:right;">{l3w:.1f}%</td>
            <td style="text-align:right;">{ms['total_win_rate_pct']:.1f}%</td>
            <td style="text-align:right;">{ms['total_trades']}</td>
            <td style="text-align:right;font-weight:bold;color:{ta_clr};">{ta:+.2f}%</td>
            <td style="text-align:right;font-weight:bold;color:{sr_clr};">{sr:+.2f}%</td>
            <td style="text-align:right;">{m.get('sharpe_ratio', 0):.2f}</td>
            <td style="text-align:right;">{calmar:.2f}</td>
            <td style="text-align:right;">{sortino:.2f}</td>
            <td style="text-align:right;">{m.get('max_drawdown_pct', 0):.2f}%</td>
            <td style="text-align:right;font-weight:bold;color:{er_clr};">{er:+.2f}%</td>
            <td style="text-align:right;color:{ic_clr};">{ic_str}</td>
            <td style="text-align:right;">{ndcg_str}</td>
            <td style="text-align:right;">{mrr_str}</td>
            <td style="text-align:right;">{ks_p_str}</td>
        </tr>"""
    if not rows:
        return ""
    return f"""
    <h3 style="font-size:14px;">模型表现</h3>
    <table style="font-size:11px;">
        <thead>
            <tr>
                <th>模型</th>
                <th style="text-align:right;">调仓盈亏</th>
                <th style="text-align:right;">近3次平均盈亏</th>
                <th style="text-align:right;">近3次胜率</th>
                <th style="text-align:right;">总胜率</th>
                <th style="text-align:right;">总交易</th>
                <th style="text-align:right;">交易平均</th>
                <th style="text-align:right;">策略收益</th>
                <th style="text-align:right;">夏普</th>
                <th style="text-align:right;">卡玛</th>
                <th style="text-align:right;">索提诺</th>
                <th style="text-align:right;">最大回撤</th>
                <th style="text-align:right;">超额收益</th>
                <th style="text-align:right;">Rank IC</th>
            <th style="text-align:right;">NDCG</th>
            <th style="text-align:right;">MRR</th>
            <th style="text-align:right;">KS-p</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>"""


def _build_health_table(health_scores):
    rows = ""
    model_labels = {
        "search_itransformer_exp_54": "i54", "search_itransformer_exp_64": "i64",
        "search_itransformer_exp_6": "i6", "average": "avg", "voting": "vote",
    }
    for key, h in sorted(health_scores.items()):
        label = model_labels.get(key, key)
        score = h["score"]
        color = _health_color(score)
        details = h.get("details", {})
        corr = h.get("corr")
        detail_cells = ""
        for k in ["wr", "avgret", "dd", "vol"]:
            v = details.get(k, "")
            if v != "":
                detail_cells += f'<td style="text-align:right;font-size:10px;">{v}</td>'
            else:
                detail_cells += '<td style="text-align:right;font-size:10px;color:#999;">—</td>'
        if corr:
            r = corr["r"]
            p = corr["p"]
            sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ""))
            corr_str = f'r={r:+.3f}{sig}'
        else:
            corr_str = "—"
        rows += f"""<tr>
            <td style="font-size:11px;color:#555;">{label}</td>
            <td style="text-align:right;font-weight:bold;color:{color};">{score:.1f}</td>
            {detail_cells}
            <td style="text-align:right;font-size:10px;">{corr_str}</td>
        </tr>"""
    if not rows:
        return ""
    return f"""
    <h3 style="font-size:14px;">健康评分</h3>
    <div class="health-table">
    <table style="font-size:11px;">
        <thead><tr>
            <th>模型</th>
            <th style="text-align:right;">评分</th>
            <th style="text-align:right;font-weight:normal;">近{10}日胜率</th>
            <th style="text-align:right;font-weight:normal;">近{3}日收益</th>
            <th style="text-align:right;font-weight:normal;">近{5}日回撤</th>
            <th style="text-align:right;font-weight:normal;">近{3}日波动</th>
            <th style="text-align:right;font-weight:normal;">近14日r</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>
    <p style="font-size:10px;color:#888;margin-top:4px;">
        评分 ≥80🟢 ≥60🟡 ≥40🟠 &lt;40🔴 &nbsp;&nbsp; r=健康度与未来5日收益的Pearson相关系数 *=p&lt;0.05 **=p&lt;0.01 ***=p&lt;0.001
    </p>
    </div>"""


def _health_color(score):
    if score >= 80:
        return "green"
    if score >= 60:
        return "goldenrod"
    if score >= 40:
        return "darkorange"
    return "red"


BEST_CONFIG = [
    ('avgret', 3, 0.514, 1),
    ('dd', 5, 0.909, -1),
    ('wr', 10, 1.175, 1),
    ('vol', 3, 1.946, -1),
]


def _compute_health_score(model_data):
    ec = model_data.get("equity_curve", [])
    if len(ec) < 2:
        return {"score": 50.0, "details": {}, "corr": None}
    values = [e["total_value"] for e in ec]
    daily_rets = [(values[i] / values[i - 1] - 1) * 100 for i in range(1, len(values))]
    df = pd.DataFrame({"ret": daily_rets})
    df["cummax"] = values[1:]
    raw = {}
    for name, window, weight, direction in BEST_CONFIG:
        if name == "avgret":
            s = df["ret"].rolling(window).mean()
            raw[name] = s.fillna(0)
        elif name == "dd":
            running_max = np.maximum.accumulate(df["cummax"].values)
            dd_vals = (df["cummax"].values - running_max) / running_max * 100
            raw[name] = pd.Series(dd_vals).rolling(window).min().fillna(0)
        elif name == "wr":
            raw[name] = (df["ret"] > 0).rolling(window).mean().fillna(0.5)
        elif name == "vol":
            raw[name] = df["ret"].rolling(window).std().fillna(0)
    latest = {k: float(v.iloc[-1]) for k, v in raw.items()}
    all_hist = {k: v.values for k, v in raw.items()}
    scores = []
    details = {}
    for name, window, weight, direction in BEST_CONFIG:
        h = all_hist[name]
        v = latest[name]
        lo, hi = float(np.min(h)), float(np.max(h))
        if hi - lo < 1e-12:
            norm = 0.5
        else:
            norm = (v - lo) / (hi - lo)
            norm = max(0.0, min(1.0, norm))
        if direction == -1:
            norm = 1.0 - norm
        scores.append(norm * weight)
        if name == "wr":
            details["wr"] = f"{v*100:.0f}%"
        elif name == "avgret":
            details["avgret"] = f"{v:+.2f}%"
        elif name == "dd":
            details["dd"] = f"{v:.1f}%"
        elif name == "vol":
            details["vol"] = f"{v:.2f}%"
    raw_score = sum(scores)
    total_weight = sum(w for _, _, w, _ in BEST_CONFIG)
    score_01 = raw_score / total_weight if total_weight > 0 else 0.5
    
    # Compute health score time series for forward-return validation
    all_scores = []
    for i in range(len(daily_rets)):
        s = []
        for name, window, weight, direction in BEST_CONFIG:
            h = all_hist[name]
            v_i = float(raw[name].iloc[i])
            lo_i, hi_i = float(np.min(h[:i+1])), float(np.max(h[:i+1]))
            if hi_i - lo_i < 1e-12:
                n = 0.5
            else:
                n = (v_i - lo_i) / (hi_i - lo_i)
                n = max(0.0, min(1.0, n))
            if direction == -1:
                n = 1.0 - n
            s.append(n * weight)
        rs = sum(s) / total_weight if total_weight > 0 else 0.5
        all_scores.append(rs)
    all_scores = np.array(all_scores)
    
    # Forward returns (next 5 days)
    fwd_ret = []
    for i in range(len(daily_rets)):
        end = min(i + 5, len(daily_rets))
        fwd_ret.append(sum(daily_rets[i:end]))
    fwd_ret = np.array(fwd_ret)
    
    # Rolling correlation (last 14 days)
    corr_val = None
    window_corr = 14
    if len(all_scores) >= window_corr and np.std(all_scores[-window_corr:]) > 0 and np.std(fwd_ret[-window_corr:]) > 0:
        from scipy.stats import pearsonr
        r, p = pearsonr(all_scores[-window_corr:], fwd_ret[-window_corr:])
        corr_val = {"r": round(r, 3), "p": round(p, 4), "n": window_corr}
    
    return {
        "score": round(max(0.0, min(100.0, score_01 * 100)), 1),
        "details": details,
        "corr": corr_val,
    }


def _compute_rank_maps(target_date):
    """Compute 1-day (实时) and 5-day (近5日) return rankings for all ETFs."""
    _df = pd.read_csv(str(PROJECT_ROOT / "etf_data" / "etf_74.csv"))
    _df["日期"] = pd.to_datetime(_df["日期"])
    _df["股票代码"] = _df["股票代码"].astype(str).str.zfill(6)
    target_dt = pd.Timestamp(target_date)
    dates = sorted(_df["日期"].unique())
    avail = [d for d in dates if d <= target_dt]
    if len(avail) < 2:
        return {}
    pivot = _df[_df["日期"].isin(avail)].pivot_table(index="股票代码", columns="日期", values="收盘")
    pivot = pivot[[c for c in pivot.columns if c <= target_dt]]
    results = {}
    if len(pivot.columns) >= 6:
        ret_5d = (pivot.iloc[:, -1] / pivot.iloc[:, -6] - 1) * 100
        ranked = ret_5d.dropna().sort_values(ascending=False)
        for i, code in enumerate(ranked.index):
            results.setdefault(code, {})["rank_5d"] = i + 1
            results[code]["ret_5d"] = round(float(ranked.iloc[i]), 2)
    if len(pivot.columns) >= 2:
        ret_1d = (pivot.iloc[:, -1] / pivot.iloc[:, -2] - 1) * 100
        ranked = ret_1d.dropna().sort_values(ascending=False)
        for i, code in enumerate(ranked.index):
            results.setdefault(code, {})["rank_1d"] = i + 1
            results[code]["ret_1d"] = round(float(ranked.iloc[i]), 2)
    return results


def _build_pred_signals_table(seq_data, report_date):
    """从 predictions_history 构建预测信号表（前10名）"""
    ph_list = seq_data.get("predictions_history", [])
    if not ph_list:
        return ""

    # 找最新的调仓日预测
    latest_ph = None
    for ph in reversed(ph_list):
        if ph.get("predictions"):
            latest_ph = ph
            break
    if not latest_ph:
        return ""

    preds = latest_ph["predictions"][:10]
    ph_date = latest_ph["date"]
    cutoff_idx = min(3, len(preds) - 1) if preds else 0
    cutoff_score = preds[cutoff_idx]["score"] if preds and cutoff_idx >= 0 else 0
    score_std = max(np.std([p["score"] for p in preds]), 1e-12) if len(preds) > 1 else 1.0

    etf_names = _load_etf_names()
    rank_map = _compute_rank_maps(report_date)

    def _rank_cell(code, key):
        info = rank_map.get(code, {})
        r = info.get(key)
        if r is None:
            return '<td style="text-align:right;color:#999;font-size:11px;">-</td>'
        clr = "#cc0000" if r <= 25 else ("#f39c12" if r <= 50 else "#009900")
        return f'<td style="text-align:right;color:{clr};font-weight:bold;font-size:11px;">{r}</td>'

    rows = ""
    for p in preds:
        rank = p["rank"]
        code = p["stock_id"]
        name = etf_names.get(code, "")
        score = p["score"]
        advantage = (score - cutoff_score) / score_std
        adv_str = f"{advantage:+.4f}"
        adv_clr = "#cc0000" if advantage >= 0 else "#009900"
        score_str = f"{score:.4f}"
        name_display = f" ({name})" if name else ""
        rows += f"""<tr><td style="text-align:right;font-weight:bold;">{rank}</td><td><a href="{_xueqiu_url(code)}" target="_blank" style="text-decoration:none;color:inherit;">{code}</a>{name_display}</td><td style="text-align:right;font-family:monospace;">{score_str}</td><td style="text-align:right;font-family:monospace;color:{adv_clr};font-weight:bold;">{adv_str}</td>{_rank_cell(code, 'rank_1d')}{_rank_cell(code, 'rank_5d')}</tr>"""

    return f"""<h3>预测信号 (Top10, {ph_date})</h3><table><thead><tr><th style="text-align:right;">排名</th><th>代码</th><th style="text-align:right;">Score</th><th style="text-align:right;">优势</th><th style="text-align:right;font-size:11px;">实时<br>排名</th><th style="text-align:right;font-size:11px;">近5日<br>排名</th></tr></thead><tbody>{rows}</tbody></table>"""


def build_report_html(*, date, model_display, total_value, cash, holdings,
                      trades_list, metrics, next_rebalance, is_rebalance,
                      today_pnl_total, today_pnl_positions=None,
                      chart_data_url=None, model_stats_section="",
                       equity_data=None, scatter_section="", health_section="",
                       trade_mode="open", pred_signals_section="",
                       market_monitor_section="", pre_holdings=None,
                       rebalance_win_rate=None, source="", is_juejin=False):
    """构建报告HTML，各组件已预先准备好"""
    # 排行数据
    _rank_map = _compute_rank_maps(date) if 'date' in locals() or date else {}
    def _rc(code, key):
        info = _rank_map.get(code, {})
        r = info.get(key)
        if r is None: return '<td style="text-align:right;color:#999;font-size:10px;">-</td>'
        clr = "#cc0000" if r <= 25 else ("#f39c12" if r <= 50 else "#009900")
        return f'<td style="text-align:right;color:{clr};font-weight:bold;font-size:10px;">{r}</td>'
    # 持仓表
    pnl_by_stock = {p["stock_id"]: p for p in (today_pnl_positions or [])}
    holdings_rows = ""
    for h in holdings:
        code = h["stock_id"]
        name = h.get("name") or code
        shares = h["shares"]
        cost = h["cost"]
        price = h.get("price", 0)
        weight = (shares * price / total_value * 100) if total_value > 0 and price else 0
        mkt_val = shares * price
        pnl = pnl_by_stock.get(code, {})
        pnl_str = f"{pnl['pnl']:+.0f}" if pnl else ""
        pnl_color = "#cc0000" if (pnl and pnl["pnl"] >= 0) else "#009900"
        price_str = f"{price:.3f}" if price else "-"
        price_display = h.get("price_display", price)
        price_display_str = f"{price_display:.3f}" if price_display else "-"
        buy_price = h.get("buy_price", 0)
        buy_price_display = h.get("buy_price_display", buy_price)
        buy_price_display_str = f"{buy_price_display:.3f}" if buy_price_display else "-"
        rebal_pnl = round(shares * (price - buy_price), 2) if price and buy_price else 0
        rebal_cost = buy_price * shares
        rebal_pnl_pct = round(rebal_pnl / rebal_cost * 100, 2) if rebal_cost > 0 else 0
        rebal_str = f"{rebal_pnl:+.0f} ({rebal_pnl_pct:+.2f}%)"
        rebal_color = "#cc0000" if rebal_pnl >= 0 else "#009900"
        hl = h.get("high_limit", 0)
        ll = h.get("low_limit", 0)
        hl_str = f"{hl:.3f}" if hl else "-"
        ll_str = f"{ll:.3f}" if ll else "-"
        hl_color = "#cc0000" if price and hl and price >= hl else "#333"
        ll_color = "#009900" if price and ll and price <= ll else "#333"
        buy_date = h.get("buy_date", "")
        buy_date_str = buy_date[-5:] if len(buy_date) >= 5 else buy_date
        buy_factor = h.get("buy_factor", 1.0)
        buy_factor_str = f"{buy_factor:.4f}" if buy_factor else "-"
        holdings_rows += f"""<tr><td><a href="{_xueqiu_url(code)}" target="_blank" style="text-decoration:none;color:inherit;">{code}</a></td><td>{name}</td>{_rc(code, 'rank_1d')}{_rc(code, 'rank_5d')}<td style="text-align:right;">{price_display_str}</td><td style="text-align:right;">{buy_price_display_str}</td><td style="text-align:right;font-family:monospace;font-size:11px;">{buy_factor_str}</td><td style="text-align:center;font-size:11px;color:#888;">{buy_date_str}</td><td style="text-align:right;color:{hl_color};">{hl_str}</td><td style="text-align:right;color:{ll_color};">{ll_str}</td><td style="text-align:right;">{shares:,}</td><td style="text-align:right;">{mkt_val:,.0f}</td><td style="text-align:right;font-weight:bold;">{weight:.2f}%</td><td style="text-align:right;color:{pnl_color};font-weight:bold;">{pnl_str}</td><td style="text-align:right;font-weight:bold;color:{rebal_color};">{rebal_str}</td></tr>"""

    cash_weight = (cash / total_value * 100) if total_value > 0 else 0
    holdings_rows += f"""<tr style="color:#999;"><td>现金</td><td>未投资资金</td><td style="text-align:right;color:#999;font-size:10px;">-</td><td style="text-align:right;color:#999;font-size:10px;">-</td><td style="text-align:right;">-</td><td style="text-align:right;">-</td><td style="text-align:right;font-size:11px;">-</td><td style="text-align:center;font-size:11px;color:#999;">-</td><td style="text-align:right;">-</td><td style="text-align:right;">-</td><td style="text-align:right;">-</td><td style="text-align:right;">{cash:,.0f}</td><td style="text-align:right;">{cash_weight:.2f}%</td><td style="text-align:right;">-</td><td style="text-align:right;">-</td></tr>"""

    pnl_total_color = "#cc0000" if today_pnl_total >= 0 else "#009900"
    total_rebal_pnl = sum(round(h["shares"] * (h["price"] - h.get("buy_price", 0)), 2) for h in holdings if h.get("price") and h.get("buy_price"))
    total_rebal_cost = sum(h.get("buy_price", 0) * h["shares"] for h in holdings if h.get("price") and h.get("buy_price"))
    total_rebal_pnl_pct = round(total_rebal_pnl / total_rebal_cost * 100, 2) if total_rebal_cost > 0 else 0
    rebal_total_color = "#cc0000" if total_rebal_pnl >= 0 else "#009900"
    total_shares = sum(h["shares"] for h in holdings)
    holdings_rows += f"""<tr style="font-weight:bold;border-top:2px solid #333;"><td colspan="4" style="text-align:right;">合计</td><td colspan="6" style="text-align:right;">&nbsp;</td><td style="text-align:right;">{total_shares:,}</td><td style="text-align:right;">{total_value:,.0f}</td><td style="text-align:right;">{(total_value - cash) / total_value * 100:.2f}%</td><td style="text-align:right;color:{pnl_total_color};">{today_pnl_total:+.0f}</td><td style="text-align:right;color:{rebal_total_color};">{total_rebal_pnl:+.0f} ({total_rebal_pnl_pct:+.2f}%)</td></tr>"""

    # 交易表
    trades_section = ""
    if trades_list and any(t["action"] in ("买入", "卖出", "跳过") for t in trades_list):
        stock_model_count = {}
        exp_models = set()
        for t in trades_list:
            s = t["stock"]
            m = t.get('model_key', '')
            if m and "exp_" in m:
                exp_models.add(m)
                if s not in stock_model_count:
                    stock_model_count[s] = set()
                stock_model_count[s].add(m)
        stock_model_count = {s: len(ms) for s, ms in stock_model_count.items()}
        total_exp_models = len(exp_models)

        trades_rows = ""
        prev_model = None
        for t in trades_list:
            cur_model = t.get('model_key', '')
            if cur_model and cur_model != prev_model:
                section_display = re.sub(r'_\w+_exp_', ' ', cur_model)
                trades_rows += f"""<tr class="section"><td colspan="11">▸ {section_display}</td></tr>"""
                prev_model = cur_model
            if t["action"] == "卖出":
                action_color = "#009900"
            elif t["action"] == "买入":
                action_color = "#cc0000"
            elif t["action"] == "跳过":
                action_color = "#999"
                row_style = ' style="background:#f5f5f5;color:#999;"'
            else:
                action_color = "#666"
                row_style = ""
            if t["action"] == "跳过":
                reason = t.get("reason", "")
                hl = t.get("high_limit", 0)
                ll = t.get("low_limit", 0)
                hl_str = f"{hl:.3f}" if hl else "-"
                ll_str = f"{ll:.3f}" if ll else "-"
                trades_rows += f"""<tr class="skipped"><td style="color:#888;">{cur_model}</td><td><span style="color:{action_color};font-weight:bold;">{t['action']}</span></td><td><a href="{_xueqiu_url(t['stock'])}" target="_blank" style="text-decoration:none;color:#aaa;">{t['stock']}</a></td><td class="skip-reason">{reason}</td><td style="text-align:right;">{hl_str}</td><td style="text-align:right;">{ll_str}</td><td style="text-align:right;">-</td><td style="text-align:right;">-</td><td style="text-align:right;color:#999;">-</td><td style="text-align:right;color:#999;">-</td><td style="text-align:right;color:#999;">-</td></tr>"""
                continue
            name_display = t.get('name') or t['stock']
            hl = t.get("high_limit", 0)
            ll = t.get("low_limit", 0)
            hl_str = f"{hl:.3f}" if hl else "-"
            ll_str = f"{ll:.3f}" if ll else "-"
            price = t.get("price", 0)
            hl_color = "#cc0000" if price and hl and price >= hl else "#333"
            ll_color = "#009900" if price and ll and price <= ll else "#333"
            cnt = stock_model_count.get(t["stock"], 0)
            if cnt > 1 and total_exp_models:
                name_display += f" ({cnt}/{total_exp_models})"
            shares_display = f"{t['shares']:,}" if t.get('shares') else "-"
            price_display = f"{t['price']:.3f}" if t.get('price') else "-"
            tc = t.get('trade_cost')
            if tc is not None and tc != 0:
                tc_display = f"{tc:.0f}"
                tc_style = "color: #cc0000;"
            else:
                tc_style = "color: #999;"
                tc_display = "-"
            adv = t.get('advantage')
            if adv is not None:
                adv_style = f"color: {'#cc0000' if adv >= 0 else '#009900'};"
                adv_display = _fmt_advantage(adv)
            else:
                adv_style = "color: #999;"
                adv_display = "-"
            reb = t.get("reb_pnl")
            reb_pct = t.get("reb_pnl_pct")
            reb_amt = t.get("reb_pnl_amount")
            if reb is not None or reb_pct is not None:
                pct = reb if reb is not None else reb_pct
                amt = reb_amt if reb_amt is not None else 0
                reb_style = f"color: {'#cc0000' if pct >= 0 else '#009900'};"
                reb_display = f"{amt:+.0f} ({pct:+.2f}%)"
            else:
                reb_style = "color: #999;"
                reb_display = "-"
            trades_rows += f"""<tr><td style="color:#888;">{cur_model}</td><td><span style="color:{action_color};font-weight:bold;">{t['action']}</span></td><td><a href="{_xueqiu_url(t['stock'])}" target="_blank" style="text-decoration:none;color:inherit;">{t['stock']}</a></td><td>{name_display}</td><td style="text-align:right;color:{hl_color};">{hl_str}</td><td style="text-align:right;color:{ll_color};">{ll_str}</td><td style="text-align:right;">{shares_display}</td><td style="text-align:right;">{price_display}</td><td style="text-align:right;{tc_style}">{tc_display}</td><td style="text-align:right;{reb_style}">{reb_display}</td><td style="text-align:right;{adv_style}">{adv_display}</td></tr>"""
        trades_section = f"""<h3>今日调仓</h3><table><thead><tr><th style="font-size:11px;">模型</th><th>操作</th><th>代码</th><th>名称</th><th style="text-align:right;">涨停价</th><th style="text-align:right;">跌停价</th><th style="text-align:right;">数量</th><th style="text-align:right;">价格</th><th style="text-align:right;">交易成本</th><th style="text-align:right;">盈亏</th><th style="text-align:right;">优势</th></tr></thead><tbody>{trades_rows}</tbody></table>"""

    # 指标
    mdd_detail = metrics.get("max_drawdown_details", {})
    mdd_period = f"{mdd_detail.get('start_date', '')} ~ {mdd_detail.get('end_date', '')} ({mdd_detail.get('duration_days', 0)}天)" if mdd_detail.get('start_date') else ""

    dd_periods = metrics.get("drawdown_periods", [])
    dd_rows = ""
    for dp in dd_periods:
        recovery = dp.get("recovery") or "进行中"
        dd_rows += f"""<tr><td>{dp['start']}</td><td>{dp['trough']}</td><td>{recovery}</td><td style="text-align:right;">{dp['depth_pct']:.2f}%</td><td style="text-align:right;">{dp['duration_days']}天</td><td style="text-align:right;">{dp.get('recovery_days') or '-'}天</td></tr>"""
    if not dd_rows:
        dd_rows = "<tr><td colspan='6' style='color:#999;text-align:center;'>暂无回撤</td></tr>"

    win_rate_str = f"{rebalance_win_rate:.1f}%" if isinstance(rebalance_win_rate, (int, float)) else ""

    window_labels = {"window_3d": "近3天(交易日)", "window_5d_real": "近5天(交易日)", "window_10d": "近10天(交易日)", "window_1m": "近20天(交易日)"}
    window_rows = ""
    # 加载HS300数据用于对比
    _hs_prices = None
    _hs_path = PROJECT_ROOT / "etf_data" / "etf_74.csv"
    for wkey in ["window_3d", "window_5d_real", "window_10d", "window_1m"]:
        w = metrics.get(wkey)
        if w:
            w_ret = w.get("strategy_return_pct", 0)
            w_ann = w.get("annualized_return_pct", 0)
            w_win = w.get("daily_win_rate", 0)
            w_win_str = f"{w_win*100:.1f}%" if isinstance(w_win, (int, float)) else ""
            w_dd = w.get("max_drawdown_pct", 0)
            w_days = w.get("total_days", 0)
            # 计算同期HS300收益
            w_hs = ""
            if _hs_path.exists() and w_days > 0:
                if _hs_prices is None:
                    _tmp = pd.read_csv(_hs_path)
                    _tmp["日期"] = pd.to_datetime(_tmp["日期"])
                    _hs = _tmp[_tmp["股票代码"] == "510300.XSHG"].sort_values("日期")
                    _hs_prices = _hs.set_index("日期")["收盘"]
                if len(_hs_prices) >= w_days + 1:
                    _hs_ret = (_hs_prices.iloc[-1] / _hs_prices.iloc[-w_days - 1] - 1) * 100
                    w_hs = f'<td style="text-align:right;color:{"#cc0000" if _hs_ret >= 0 else "#009900"};">{_hs_ret:+.2f}%</td>'
            window_rows += f"""<tr><td>{window_labels.get(wkey, wkey)}</td><td style="text-align:right;color:{'#cc0000' if w_ret >= 0 else '#009900'};font-weight:bold;">{_pct(w_ret)}</td>{w_hs}<td style="text-align:right;">{_pct(w_ann)}</td><td style="text-align:right;">{w_win_str}</td><td style="text-align:right;">{w_dd:.2f}%</td></tr>"""

    # 账户总值栏
    sr = metrics.get("strategy_return_pct", 0)
    total_pnl_color = "#cc0000" if sr >= 0 else "#009900"
    today_pnl_color = "#cc0000" if today_pnl_total >= 0 else "#009900"
    today_pnl_pct = round(today_pnl_total / (total_value - today_pnl_total) * 100, 2) if (total_value - today_pnl_total) > 0 else 0
    init_cap = total_value / (1 + sr / 100) if sr != -100 else total_value
    total_pnl_abs = round(total_value - init_cap, 2)
    total_bar = f"""<div class="total-bar"><div><div style="font-size:12px;color:#666;">账户总值</div><div class="amt">¥{total_value:,.2f}</div></div><div class="dtl"><div>今日 <span class="chg {today_pnl_color}">{today_pnl_total:+.0f} ({today_pnl_pct:+.2f}%)</span></div><div style="font-size:11px;">累计收益 <span class="chg {total_pnl_color}">{total_pnl_abs:+,.0f} ({sr:+.2f}%)</span></div></div></div>"""

    # 指标 3x4 表格
    sr_v = _pct(metrics.get("strategy_return_pct", 0))
    ar_v = _pct(metrics.get("annualized_return_pct", 0))
    hs_v = _pct(metrics.get("hs300_return_pct", 0))
    er_v = _pct(metrics.get("excess_return_pct", 0))
    wr_v = win_rate_str
    md_v = f"{metrics.get('max_drawdown_pct', 0):.2f}%"
    sh_v = f"{metrics.get('sharpe_ratio', 0):.2f}"
    ca_v = f"{metrics.get('calmar_ratio', 0):.2f}"
    so_v = f"{metrics.get('sortino_ratio', 0):.2f}"
    av_v = f"{metrics.get('annualized_volatility_pct', 0):.2f}%"
    td_v = str(metrics.get("total_days", 0))
    cs_v = f"¥{cash:,.0f}"
    def _cell(label, val, clr="#333"):
        return f"<td><div class=\"l\">{label}</div><div class=\"v\" style=\"color:{clr};\">{val}</div></td>"
    def _clr(v):
        if v.startswith("+"): return "#cc0000"
        if v.startswith("-"): return "#009900"
        return "#333"
    r1 = _cell("策略收益", sr_v, _clr(sr_v)) + _cell("年化收益", ar_v, _clr(ar_v)) + _cell("沪深300", hs_v, _clr(hs_v)) + _cell("超额收益", er_v, _clr(er_v))
    r2 = _cell("调仓胜率", wr_v) + _cell("最大回撤", md_v) + _cell("夏普比率", sh_v) + _cell("卡玛比率", ca_v)
    r3 = _cell("索提诺", so_v) + _cell("年化波动", av_v) + _cell("总交易日", td_v) + _cell("现金", cs_v)
    metrics_rows = f"<tr>{r1}</tr><tr>{r2}</tr><tr>{r3}</tr>"

    # 填充模板
    html = _load_template()
    mode_label = "开盘交易" if trade_mode == "open" else "收盘交易"
    juejin_badge = '<span style="display:inline-block;background:#e74c3c;color:#fff;font-size:10px;font-weight:bold;padding:1px 6px;border-radius:3px;margin-left:6px;">掘金</span>' if is_juejin else ""
    source_tag = f' | 来源: {source}' if source else ""
    html = html.replace("{{MODEL_INFO}}", f"模型: {model_display}{juejin_badge}{source_tag} | 日期: {date} | 模式: {mode_label} | 下个调仓日: {next_rebalance}")
    html = html.replace("{{TOTAL_BAR}}", total_bar)
    html = html.replace("{{METRICS_ROWS}}", metrics_rows)
    html = html.replace("{{WINDOW_ROWS}}", window_rows)
    html = html.replace("{{DD_ROWS}}", dd_rows)
    html = html.replace("{{MODEL_STATS_SECTION}}", model_stats_section)
    html = html.replace("{{HEALTH_SECTION}}", health_section)
    html = html.replace("{{HOLDINGS_TITLE}}", f"当前持仓 ({len(holdings)} 只)")
    html = html.replace("{{HOLDINGS_ROWS}}", holdings_rows)

    pre_holdings_section = ""
    if pre_holdings:
        pre_rows = ""
        for h in pre_holdings:
            code = h["stock_id"]
            name = h.get("name") or code
            shares = h["shares"]
            price = h.get("price_display", h.get("price", 0))
            pre_rows += f'<tr><td>{code}</td><td>{name}</td><td style="text-align:right;">{price:.3f}</td><td style="text-align:right;">{shares:,}</td></tr>'
        pre_holdings_section = f'<h3>上期持仓 ({len(pre_holdings)} 只)</h3><table><thead><tr><th>代码</th><th>名称</th><th style="text-align:right;">现价</th><th style="text-align:right;">股数</th></tr></thead><tbody>{pre_rows}</tbody></table>'
    html = html.replace("{{PRE_HOLDINGS_SECTION}}", pre_holdings_section)

    html = html.replace("{{TRADES_SECTION}}", trades_section)
    html = html.replace("{{CHART_SRC}}", chart_data_url or "cid:chart_img")
    html = html.replace("{{SCATTER_SECTION}}", scatter_section)
    html = html.replace("{{MARKET_MONITOR_SECTION}}", market_monitor_section)
    html = html.replace("{{PRED_SIGNALS_SECTION}}", pred_signals_section)
    html = html.replace("{{EQUITY_DATA}}", json.dumps(equity_data) if equity_data else "{}")
    html = html.replace("{{GENERATED_TIME}}", datetime.now().strftime("%Y-%m-%d %H:%M"))
    return html


def send_report(model_key=None, verbose=False):
    if not SMTP_USER or not SMTP_PASSWORD:
        print("错误: 请设置 SMTP_USER 和 SMTP_PASSWORD 环境变量")
        return False

    if not REPORT_PATH.exists():
        print(f"错误: 未找到报告文件 {REPORT_PATH}")
        print("请先运行 daily_eval 生成报告")
        return False

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    date = report["date"]
    sequences = report.get("sequences", {})
    is_rebalance = report["is_rebalance_day"]

    if model_key and model_key in sequences:
        seq_data = sequences[model_key]
    else:
        model_key = next(iter(sequences))
        seq_data = sequences[model_key]

    metrics = seq_data["metrics"]
    # 补充近3天、近10天窗口（如果回测长度足够）
    ec = seq_data.get("equity_curve", [])
    if len(ec) >= 4:
        vals = [e["total_value"] for e in ec]
        _add_window(metrics, "window_3d", vals, 3)
    if len(ec) >= 6:
        vals = [e["total_value"] for e in ec]
        _add_window(metrics, "window_5d_real", vals, 5)
    if len(ec) >= 11:
        vals = [e["total_value"] for e in ec]
        _add_window(metrics, "window_10d", vals, 10)
    holdings = seq_data.get("holdings", report.get("holdings", []))
    pre_holdings = report.get("pre_holdings", [])
    cash = seq_data.get("cash", report.get("cash", 0))
    total_value = metrics.get("latest_value", 0)
    rebalance_win_rate = seq_data.get("model_stats", {}).get("total_win_rate_pct")
    source = report.get("source", "")

    trades = report.get("all_today_trades", report.get("today_trades", []))

    today_pnl_data = seq_data.get("today_pnl", {})
    today_pnl_total = today_pnl_data.get("total_pnl", 0)
    today_pnl_positions = today_pnl_data.get("positions", [])

    next_rebalance = report.get("next_rebalance_date", "")
    _display_names = {"average": "平均", "voting": "投票"}
    if model_key in ("juejin",):
        real_key = next((k for k in sequences if k not in ("juejin", "average", "voting")), model_key)
        model_display = _display_names.get(real_key, real_key.replace("search_", "").replace("_exp_", " "))
    else:
        model_display = _display_names.get(model_key, model_key.replace("search_", "").replace("_exp_", " "))

    model_stats_section = _build_model_stats_table(sequences)

    health_scores = {}
    for key, seq in sequences.items():
        ec = seq.get("equity_curve", [])
        if ec:
            health_scores[key] = _compute_health_score(seq)
    health_section = _build_health_table(health_scores)

    equity_data = {}
    for key, seq in sequences.items():
        ec = seq.get("equity_curve", [])
        if ec:
            display = key.replace("search_", "").replace("_exp_", " ")
            equity_data[display] = ec
    hs300 = report.get("hs300_curve", [])
    if hs300:
        equity_data["沪深300"] = hs300

    trade_mode = report.get("trade_mode", "open")

    # 预测信号表
    pred_signals_section = _build_pred_signals_table(seq_data, date)

    # 市场监控
    market_monitor_section = ""
    try:
        from market_monitor import run_market_monitor
        _, _, mm_html = run_market_monitor(verbose=verbose)
        market_monitor_section = mm_html
    except Exception as e:
        print(f"  [市场监控] 生成失败: {e}")

    # 补齐展示用 raw 价格 + 复权因子
    if holdings:
        try:
            _raw_prices = pd.read_csv(str(PROJECT_ROOT / "etf_data" / "etf_74.csv"))
            _raw_prices["日期"] = pd.to_datetime(_raw_prices["日期"])
            _target = pd.Timestamp(date)
            for h in holdings:
                code = h["stock_id"]
                _sub = _raw_prices[_raw_prices["股票代码"] == code]
                if "price_display" not in h:
                    _tc = _sub.loc[_sub["日期"] == _target, "收盘_原始"]
                    h["price_display"] = float(_tc.values[0]) if not _tc.empty else h.get("price", 0)
                if "buy_factor" not in h and h.get("buy_date"):
                    _bf = _sub.loc[_sub["日期"] == pd.Timestamp(h["buy_date"]), "复权因子"]
                    h["buy_factor"] = float(_bf.values[0]) if not _bf.empty else 1.0
                if "buy_price_display" not in h and h.get("buy_date"):
                    _bp = _sub.loc[_sub["日期"] == pd.Timestamp(h["buy_date"]), "开盘_原始" if trade_mode == "open" else "收盘_原始"]
                    h["buy_price_display"] = float(_bp.values[0]) if not _bp.empty else h.get("buy_price", 0)
        except Exception:
            pass

    html_body = build_report_html(
        date=date,
        model_display=model_display,
        total_value=total_value,
        cash=cash,
        holdings=holdings,
        pre_holdings=pre_holdings,
        trades_list=trades,
        metrics=metrics,
        next_rebalance=next_rebalance,
        is_rebalance=is_rebalance,
        today_pnl_total=today_pnl_total,
        today_pnl_positions=today_pnl_positions,
        model_stats_section=model_stats_section,
        equity_data=equity_data,
        health_section=health_section,
        trade_mode=trade_mode,
        pred_signals_section=pred_signals_section,
        market_monitor_section=market_monitor_section,
        rebalance_win_rate=rebalance_win_rate,
        source=source,
        is_juejin=(model_key == "juejin"),
    )

    msg = MIMEMultipart()
    msg["Subject"] = f"ETF 每日测评报告 ({model_display}) - {date}"
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    # 保存HTML到本地
    report_html_path = PROJECT_ROOT / "output" / "latest_report.html"
    report_html_path.parent.mkdir(parents=True, exist_ok=True)
    report_html_path.write_text(html_body, encoding="utf-8")

    msg.attach(MIMEText(html_body, "html", "utf-8"))

    if CHART_PATH.exists():
        with open(CHART_PATH, "rb") as f:
            img_data = f.read()
        img = MIMEImage(img_data, name="equity_curves.png")
        img.add_header("Content-ID", "<chart_img>")
        msg.attach(img)
    else:
        print(f"警告: 未找到图表文件 {CHART_PATH}")

    try:
        print(f"正在发送邮件至 {EMAIL_TO} ...")
        if SMTP_PORT == 465:
            server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        else:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()

        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(EMAIL_FROM, EMAIL_TO.split(","), msg.as_string())
        server.quit()
        print("邮件发送成功")
        return True
    except Exception as e:
        print(f"邮件发送失败: {e}")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="发送 ETF 测评报告")
    parser.add_argument("--model-key", type=str, default=None, help="指定报告的模型标识 (如 tcn_exp_5)")
    parser.add_argument("--to", type=str, default=None, help="覆盖接收人邮箱 (多个用逗号分隔)")
    args = parser.parse_args()

    if args.to:
        EMAIL_TO = args.to

    send_report(model_key=args.model_key)