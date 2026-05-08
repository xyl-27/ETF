"""
发送每日测评报告邮件
读取 output/latest_report.json 和 output/equity_curves.png
通过 SMTP 发送 HTML 格式邮件
"""

import os
import sys
import smtplib
import csv
import json
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_PATH = PROJECT_ROOT / "output" / "latest_report.json"
CHART_PATH = PROJECT_ROOT / "output" / "equity_curves.png"
ETF_LIST_PATH = PROJECT_ROOT / "etf_data" / "etf_list_before_2022_74.csv"

# 邮件配置 (从环境变量读取)
os.environ['SMTP_USER'] = '3759608757@qq.com'
os.environ['SMTP_PASSWORD'] = 'gsiqpfqjjkvwcdfg'
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.qq.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "465"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
EMAIL_FROM = os.environ.get("EMAIL_FROM", SMTP_USER)
EMAIL_TO = os.environ.get("EMAIL_TO", "1280745039@qq.com")

def load_etf_names():
    mapping = {}
    if not ETF_LIST_PATH.exists():
        return mapping
    try:
        with open(ETF_LIST_PATH, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row.get("代码", "").strip()
                name = row.get("名称", "").strip()
                if code and name:
                    mapping[code] = name
    except Exception:
        pass
    return mapping

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
        # 默认取 sequences 中的第一个模型
        model_key = next(iter(sequences))
        seq_data = sequences[model_key]

    metrics = seq_data["metrics"]
    holdings = seq_data.get("holdings", report.get("holdings", []))
    cash = seq_data.get("cash", report.get("cash", 0))
    total_value = metrics.get("latest_value", 0)
    
    trades = [t for t in report.get("today_trades", []) if t.get("model_key") == model_key or not report.get("today_trades")]
    if not trades:
        trades = report.get("today_trades", [])

    etf_names = load_etf_names()

    # 今日盈亏
    today_pnl_data = seq_data.get("today_pnl", {})
    today_pnl_total = today_pnl_data.get("total_pnl", 0)

    # 构建持仓表格
    pnl_by_stock = {p["stock_id"]: p for p in today_pnl_data.get("positions", [])}
    holdings_rows = ""
    for h in holdings:
        code = h["stock_id"]
        name = etf_names.get(code, code)
        shares = h["shares"]
        cost = h["cost"]
        weight = (cost / total_value * 100) if total_value > 0 else 0
        pnl = pnl_by_stock.get(code, {})
        pnl_str = f"{pnl['pnl']:+.2f}" if pnl else ""
        pnl_color = "#cc0000" if (pnl and pnl["pnl"] >= 0) else "#009900"
        holdings_rows += f"""
        <tr>
            <td>{code}</td>
            <td>{name}</td>
            <td style="text-align: right;">{shares:,}</td>
            <td style="text-align: right;">{cost:,.2f}</td>
            <td style="text-align: right; font-weight: bold;">{weight:.2f}%</td>
            <td style="text-align: right; color: {pnl_color}; font-weight: bold;">{pnl_str}</td>
        </tr>
        """

    # 现金行
    cash_weight = (cash / total_value * 100) if total_value > 0 else 0
    holdings_rows += f"""
    <tr style="color: #999;">
        <td>现金</td>
        <td>未投资资金</td>
        <td style="text-align: right;">-</td>
        <td style="text-align: right;">{cash:,.2f}</td>
        <td style="text-align: right;">{cash_weight:.2f}%</td>
        <td style="text-align: right;">-</td>
    </tr>
    """

    # 今日盈亏汇总行
    pnl_total_color = "#cc0000" if today_pnl_total >= 0 else "#009900"
    holdings_rows += f"""
    <tr style="font-weight: bold; border-top: 2px solid #333;">
        <td colspan="4" style="text-align: right;">今日合计盈亏</td>
        <td style="text-align: right;"></td>
        <td style="text-align: right; color: {pnl_total_color};">{today_pnl_total:+.2f}</td>
    </tr>
    """

    # 构建交易表格
    trades_rows = ""
    if trades:
        for t in trades:
            action_color = "#cc0000" if t["action"] == "买入" else "#009900"
            trades_rows += f"""
            <tr>
                <td><span style="color: {action_color}; font-weight: bold;">{t['action']}</span></td>
                <td>{t['stock']}</td>
                <td>{etf_names.get(t['stock'].replace('.XSHG','').replace('.XSHE',''), '')}</td>
                <td style="text-align: right;">{t['shares']:,}</td>
                <td style="text-align: right;">{t['price']:.4f}</td>
            </tr>
            """
    else:
        trades_rows = "<tr><td colspan='5' style='color: #999; text-align: center;'>无调仓操作</td></tr>"

    next_rebalance = report.get("next_rebalance_date", "")

    model_display = model_key.replace("_", " ").title()
    
    # 构建详细指标HTML
    def _pct(v, suffix="%"):
        return f"{v:+.2f}{suffix}" if isinstance(v, (int, float)) else str(v)
    
    mdd_detail = metrics.get("max_drawdown_details", {})
    mdd_period = f"{mdd_detail.get('start_date', '')} ~ {mdd_detail.get('end_date', '')} ({mdd_detail.get('duration_days', 0)}天)" if mdd_detail.get('start_date') else ""
    
    win_rate = metrics.get("daily_win_rate")
    win_rate_str = f"{win_rate*100:.1f}%" if isinstance(win_rate, (int, float)) else ""
    
    # 近期窗口指标
    window_labels = {"window_5d": "近5天(交易日)", "window_1m": "近一个月"}
    window_rows = ""
    for wkey in ["window_5d", "window_1m"]:
        w = metrics.get(wkey)
        if w:
            w_ret = w.get("strategy_return_pct", 0)
            w_ann = w.get("annualized_return_pct", 0)
            w_win = w.get("daily_win_rate", 0)
            if isinstance(w_win, (int, float)):
                w_win_str = f"{w_win*100:.1f}%"
            else:
                w_win_str = ""
            w_dd = w.get("max_drawdown_pct", 0)
            window_rows += f"""
            <tr>
                <td>{window_labels.get(wkey, wkey)}</td>
                <td style="text-align: right; color: {'#cc0000' if w_ret >= 0 else '#009900'}; font-weight: bold;">{_pct(w_ret)}</td>
                <td style="text-align: right;">{_pct(w_ann)}</td>
                <td style="text-align: right;">{w_win_str}</td>
                <td style="text-align: right;">{_pct(w_dd)}</td>
            </tr>"""
    
    html_body = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; }}
            .header {{ background-color: #f8f9fa; padding: 15px; border-bottom: 2px solid #007bff; margin-bottom: 20px; }}
            .header h2 {{ margin: 0; color: #007bff; }}
            .header .model {{ font-size: 0.9em; color: #666; margin-top: 5px; }}
            .metrics {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }}
            .metric-box {{ flex: 1; min-width: 100px; padding: 10px; background: #fff; border: 1px solid #ddd; border-radius: 5px; text-align: center; }}
            .metric-box .label {{ font-size: 0.85em; color: #666; }}
            .metric-box .value {{ font-size: 1.1em; font-weight: bold; color: #333; }}
            .value.pos {{ color: #cc0000; }}
            .value.neg {{ color: #009900; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ padding: 8px 10px; border-bottom: 1px solid #eee; text-align: left; }}
            th {{ background-color: #f8f9fa; font-weight: 600; }}
            .chart {{ margin-top: 20px; text-align: center; }}
            .chart img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
            .footer {{ margin-top: 30px; padding-top: 15px; border-top: 1px solid #eee; font-size: 0.85em; color: #999; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>ETF 每日测评报告</h2>
            <div class="model">模型: {model_display} | 日期: {date} | 下个调仓日: {next_rebalance}</div>
        </div>

        <div class="metrics">
            <div class="metric-box">
                <div class="label">策略收益</div>
                <div class="value {'pos' if metrics['strategy_return_pct'] >= 0 else 'neg'}">
                    {metrics['strategy_return_pct']:+.2f}%
                </div>
            </div>
            <div class="metric-box">
                <div class="label">年化收益</div>
                <div class="value {'pos' if metrics.get('annualized_return_pct', 0) >= 0 else 'neg'}">
                    {_pct(metrics.get('annualized_return_pct', 0))}
                </div>
            </div>
            <div class="metric-box">
                <div class="label">沪深300</div>
                <div class="value {'pos' if metrics['hs300_return_pct'] >= 0 else 'neg'}">
                    {metrics['hs300_return_pct']:+.2f}%
                </div>
            </div>
            <div class="metric-box">
                <div class="label">超额收益</div>
                <div class="value {'pos' if metrics['excess_return_pct'] >= 0 else 'neg'}">
                    {metrics['excess_return_pct']:+.2f}%
                </div>
            </div>
            <div class="metric-box">
                <div class="label">日胜率</div>
                <div class="value">{win_rate_str}</div>
            </div>
            <div class="metric-box">
                <div class="label">最大回撤</div>
                <div class="value">{metrics['max_drawdown_pct']:.2f}%</div>
            </div>
            <div class="metric-box">
                <div class="label">夏普比率</div>
                <div class="value">{metrics['sharpe_ratio']:.2f}</div>
            </div>
            <div class="metric-box">
                <div class="label">卡玛比率</div>
                <div class="value">{metrics.get('calmar_ratio', 0):.2f}</div>
            </div>
            <div class="metric-box">
                <div class="label">索提诺</div>
                <div class="value">{metrics.get('sortino_ratio', 0):.2f}</div>
            </div>
            <div class="metric-box">
                <div class="label">账户总值</div>
                <div class="value">{total_value:,.2f}</div>
            </div>
        </div>

        <h3 style="font-size: 14px;">近期表现</h3>
        <table style="font-size: 12px;">
            <thead>
                <tr>
                    <th>区间</th>
                    <th style="text-align: right;">收益</th>
                    <th style="text-align: right;">年化</th>
                    <th style="text-align: right;">日胜率</th>
                    <th style="text-align: right;">最大回撤</th>
                </tr>
            </thead>
            <tbody>
                {window_rows}
            </tbody>
        </table>

        <h3 style="font-size: 14px;">回撤区间</h3>
        <p style="font-size: 12px;">{mdd_period}</p>

        <h3 style="font-size: 14px;">当前持仓 ({len(holdings)} 只)</h3>
        <table style="font-size: 12px;">
            <thead>
                <tr>
                    <th>代码</th>
                    <th>名称</th>
                    <th style="text-align: right;">股数</th>
                    <th style="text-align: right;">成本</th>
                    <th style="text-align: right;">仓位</th>
                    <th style="text-align: right;">今日盈亏</th>
                </tr>
            </thead>
            <tbody>
                {holdings_rows}
            </tbody>
        </table>

        <h3 style="font-size: 14px;">今日调仓</h3>
        <table style="font-size: 12px;">
            <thead>
                <tr>
                    <th>操作</th>
                    <th>代码</th>
                    <th>名称</th>
                    <th style="text-align: right;">数量</th>
                    <th style="text-align: right;">价格</th>
                </tr>
            </thead>
            <tbody>
                {trades_rows}
            </tbody>
        </table>
        <div class="chart">
            <h3>收益曲线对比</h3>
            <img src="cid:chart_img" alt="Equity Curves">
        </div>

        <div class="footer">
            由 ETF 每日测评系统自动生成 | {datetime.now().strftime("%Y-%m-%d %H:%M")}
        </div>
    </body>
    </html>
    """

    msg = MIMEMultipart()
    msg["Subject"] = f"ETF 每日测评报告 ({model_display}) - {date}"
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

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