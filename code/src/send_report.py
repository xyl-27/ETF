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
            <td style="text-align:right;">{m.get('max_drawdown_pct', 0):.2f}%</td>
            <td style="text-align:right;font-weight:bold;color:{er_clr};">{er:+.2f}%</td>
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
                <th style="text-align:right;">最大回撤</th>
                <th style="text-align:right;">超额收益</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>"""


def build_report_html(*, date, model_display, total_value, cash, holdings,
                      trades_list, metrics, next_rebalance, is_rebalance,
                      today_pnl_total, today_pnl_positions=None,
                      chart_data_url=None, model_stats_section="",
                      equity_data=None):
    """构建报告HTML，各组件已预先准备好"""
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
        buy_price = h.get("buy_price", 0)
        buy_price_str = f"{buy_price:.3f}" if buy_price else "-"
        rebal_pnl = round(shares * (price - buy_price), 2) if price and buy_price else 0
        rebal_cost = buy_price * shares
        rebal_pnl_pct = round(rebal_pnl / rebal_cost * 100, 2) if rebal_cost > 0 else 0
        rebal_str = f"{rebal_pnl:+.0f} ({rebal_pnl_pct:+.2f}%)"
        rebal_color = "#cc0000" if rebal_pnl >= 0 else "#009900"
        holdings_rows += f"""<tr><td><a href="{_xueqiu_url(code)}" target="_blank" style="text-decoration:none;color:inherit;">{code}</a></td><td>{name}</td><td style="text-align:right;">{price_str}</td><td style="text-align:right;">{buy_price_str}</td><td style="text-align:right;">{shares:,}</td><td style="text-align:right;">{mkt_val:,.0f}</td><td style="text-align:right;font-weight:bold;">{weight:.2f}%</td><td style="text-align:right;color:{pnl_color};font-weight:bold;">{pnl_str}</td><td style="text-align:right;font-weight:bold;color:{rebal_color};">{rebal_str}</td></tr>"""

    cash_weight = (cash / total_value * 100) if total_value > 0 else 0
    holdings_rows += f"""<tr style="color:#999;"><td>现金</td><td>未投资资金</td><td style="text-align:right;">-</td><td style="text-align:right;">-</td><td style="text-align:right;">-</td><td style="text-align:right;">{cash:,.0f}</td><td style="text-align:right;">{cash_weight:.2f}%</td><td style="text-align:right;">-</td><td style="text-align:right;">-</td></tr>"""

    pnl_total_color = "#cc0000" if today_pnl_total >= 0 else "#009900"
    total_rebal_pnl = sum(round(h["shares"] * (h["price"] - h.get("buy_price", 0)), 2) for h in holdings if h.get("price") and h.get("buy_price"))
    total_rebal_cost = sum(h.get("buy_price", 0) * h["shares"] for h in holdings if h.get("price") and h.get("buy_price"))
    total_rebal_pnl_pct = round(total_rebal_pnl / total_rebal_cost * 100, 2) if total_rebal_cost > 0 else 0
    rebal_total_color = "#cc0000" if total_rebal_pnl >= 0 else "#009900"
    holdings_rows += f"""<tr style="font-weight:bold;border-top:2px solid #333;"><td colspan="6" style="text-align:right;">合计</td><td style="text-align:right;font-weight:bold;">{(total_value - cash) / total_value * 100:.2f}%</td><td style="text-align:right;color:{pnl_total_color};">{today_pnl_total:+.0f}</td><td style="text-align:right;color:{rebal_total_color};">{total_rebal_pnl:+.0f} ({total_rebal_pnl_pct:+.2f}%)</td></tr>"""

    # 交易表
    trades_section = ""
    if trades_list and any(t["action"] in ("买入", "卖出") for t in trades_list):
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
                trades_rows += f"""<tr class="section"><td colspan="8">▸ {section_display}</td></tr>"""
                prev_model = cur_model
            if t["action"] == "买入":
                action_color = "#cc0000"
            elif t["action"] == "卖出":
                action_color = "#009900"
            else:
                action_color = "#666"
            name_display = t.get('name') or t['stock']
            cnt = stock_model_count.get(t["stock"], 0)
            if cnt > 1 and total_exp_models:
                name_display += f" ({cnt}/{total_exp_models})"
            shares_display = f"{t['shares']:,}" if t.get('shares') else "-"
            price_display = f"{t['price']:.3f}" if t.get('price') else "-"
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
            trades_rows += f"""<tr><td style="color:#888;">{cur_model}</td><td><span style="color:{action_color};font-weight:bold;">{t['action']}</span></td><td><a href="{_xueqiu_url(t['stock'])}" target="_blank" style="text-decoration:none;color:inherit;">{t['stock']}</a></td><td>{name_display}</td><td style="text-align:right;">{shares_display}</td><td style="text-align:right;">{price_display}</td><td style="text-align:right;{reb_style}">{reb_display}</td><td style="text-align:right;{adv_style}">{adv_display}</td></tr>"""
        trades_section = f"""<h3>今日调仓</h3><table><thead><tr><th style="font-size:11px;">模型</th><th>操作</th><th>代码</th><th>名称</th><th style="text-align:right;">数量</th><th style="text-align:right;">价格</th><th style="text-align:right;">盈亏</th><th style="text-align:right;">优势</th></tr></thead><tbody>{trades_rows}</tbody></table>"""

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

    win_rate = metrics.get("daily_win_rate")
    win_rate_str = f"{win_rate*100:.1f}%" if isinstance(win_rate, (int, float)) else ""

    window_labels = {"window_5d": "近5天(交易日)", "window_1m": "近一个月"}
    window_rows = ""
    for wkey in ["window_5d", "window_1m"]:
        w = metrics.get(wkey)
        if w:
            w_ret = w.get("strategy_return_pct", 0)
            w_ann = w.get("annualized_return_pct", 0)
            w_win = w.get("daily_win_rate", 0)
            w_win_str = f"{w_win*100:.1f}%" if isinstance(w_win, (int, float)) else ""
            w_dd = w.get("max_drawdown_pct", 0)
            window_rows += f"""<tr><td>{window_labels.get(wkey, wkey)}</td><td style="text-align:right;color:{'#cc0000' if w_ret >= 0 else '#009900'};font-weight:bold;">{_pct(w_ret)}</td><td style="text-align:right;">{_pct(w_ann)}</td><td style="text-align:right;">{w_win_str}</td><td style="text-align:right;">{w_dd:.2f}%</td></tr>"""

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
    r2 = _cell("日胜率", wr_v) + _cell("最大回撤", md_v) + _cell("夏普比率", sh_v) + _cell("卡玛比率", ca_v)
    r3 = _cell("索提诺", so_v) + _cell("年化波动", av_v) + _cell("总交易日", td_v) + _cell("现金", cs_v)
    metrics_rows = f"<tr>{r1}</tr><tr>{r2}</tr><tr>{r3}</tr>"

    # 填充模板
    html = _load_template()
    html = html.replace("{{MODEL_INFO}}", f"模型: {model_display} | 日期: {date} | 下个调仓日: {next_rebalance}")
    html = html.replace("{{TOTAL_BAR}}", total_bar)
    html = html.replace("{{METRICS_ROWS}}", metrics_rows)
    html = html.replace("{{WINDOW_ROWS}}", window_rows)
    html = html.replace("{{DD_ROWS}}", dd_rows)
    html = html.replace("{{MODEL_STATS_SECTION}}", model_stats_section)
    html = html.replace("{{HOLDINGS_TITLE}}", f"当前持仓 ({len(holdings)} 只)")
    html = html.replace("{{HOLDINGS_ROWS}}", holdings_rows)
    html = html.replace("{{TRADES_SECTION}}", trades_section)
    html = html.replace("{{CHART_SRC}}", chart_data_url or "cid:chart_img")
    html = html.replace("{{EQUITY_DATA}}", json.dumps(equity_data) if equity_data else "{}")
    html = html.replace("{{GENERATED_TIME}}", datetime.now().strftime("%Y-%m-%d %H:%M"))
    return html


def send_report(model_key=None):
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
    holdings = seq_data.get("holdings", report.get("holdings", []))
    cash = seq_data.get("cash", report.get("cash", 0))
    total_value = metrics.get("latest_value", 0)

    trades = report.get("all_today_trades", report.get("today_trades", []))

    today_pnl_data = seq_data.get("today_pnl", {})
    today_pnl_total = today_pnl_data.get("total_pnl", 0)
    today_pnl_positions = today_pnl_data.get("positions", [])

    next_rebalance = report.get("next_rebalance_date", "")
    model_display = model_key.replace("search_", "").replace("_exp_", " ")

    model_stats_section = _build_model_stats_table(sequences)

    equity_data = {}
    for key, seq in sequences.items():
        ec = seq.get("equity_curve", [])
        if ec:
            display = key.replace("search_", "").replace("_exp_", " ")
            equity_data[display] = ec

    html_body = build_report_html(
        date=date,
        model_display=model_display,
        total_value=total_value,
        cash=cash,
        holdings=holdings,
        trades_list=trades,
        metrics=metrics,
        next_rebalance=next_rebalance,
        is_rebalance=is_rebalance,
        today_pnl_total=today_pnl_total,
        today_pnl_positions=today_pnl_positions,
        model_stats_section=model_stats_section,
        equity_data=equity_data,
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