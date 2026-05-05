"""
发送每日测评报告邮件
读取 output/latest_report.json 和 output/equity_curves.png
通过 SMTP 发送 HTML 格式邮件
"""

import os
import sys
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = PROJECT_ROOT / "output" / "latest_report.json"
CHART_PATH = PROJECT_ROOT / "output" / "equity_curves.png"

# 邮件配置 (从环境变量读取)
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.qq.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "465"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
EMAIL_FROM = os.environ.get("EMAIL_FROM", SMTP_USER)
EMAIL_TO = os.environ.get("EMAIL_TO", SMTP_USER)

def send_report():
    if not SMTP_USER or not SMTP_PASSWORD:
        print("错误: 请设置 SMTP_USER 和 SMTP_PASSWORD 环境变量")
        return False

    if not REPORT_PATH.exists():
        print(f"错误: 未找到报告文件 {REPORT_PATH}")
        print("请先运行 daily_eval 生成报告")
        return False

    # 1. 读取报告数据
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    date = report["date"]
    metrics = report["metrics"]
    holdings = report["holdings"]
    trades = report.get("today_trades", [])
    is_rebalance = report["is_rebalance_day"]

    # 2. 构建 HTML 正文
    holdings_rows = ""
    for h in holdings:
        holdings_rows += f"""
        <tr>
            <td>{h['stock_id']}</td>
            <td>{h['shares']:,}</td>
            <td>{h['cost']:,.2f}</td>
        </tr>
        """

    trades_rows = ""
    if trades:
        for t in trades:
            action_color = "#2ecc71" if t["action"] == "买入" else "#e74c3c"
            trades_rows += f"""
            <tr>
                <td><span style="color: {action_color}; font-weight: bold;">{t['action']}</span></td>
                <td>{t['stock']}</td>
                <td>{t['shares']:,}</td>
                <td>{t['price']:.4f}</td>
            </tr>
            """
    else:
        trades_rows = "<tr><td colspan='4' style='color: #999;'>无调仓操作</td></tr>"

    html_body = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; }}
            .header {{ background-color: #f8f9fa; padding: 15px; border-bottom: 2px solid #007bff; margin-bottom: 20px; }}
            .header h2 {{ margin: 0; color: #007bff; }}
            .metrics {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }}
            .metric-box {{ flex: 1; min-width: 120px; padding: 10px; background: #fff; border: 1px solid #ddd; border-radius: 5px; text-align: center; }}
            .metric-box .label {{ font-size: 0.85em; color: #666; }}
            .metric-box .value {{ font-size: 1.2em; font-weight: bold; color: #333; }}
            .value.pos {{ color: #2ecc71; }}
            .value.neg {{ color: #e74c3c; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ padding: 8px 12px; border-bottom: 1px solid #eee; text-align: left; }}
            th {{ background-color: #f8f9fa; font-weight: 600; }}
            .chart {{ margin-top: 20px; text-align: center; }}
            .chart img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
            .footer {{ margin-top: 30px; padding-top: 15px; border-top: 1px solid #eee; font-size: 0.85em; color: #999; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>ETF 每日测评报告</h2>
            <div>日期: {date}</div>
        </div>

        <div class="metrics">
            <div class="metric-box">
                <div class="label">累计收益</div>
                <div class="value {'pos' if metrics['strategy_return_pct'] >= 0 else 'neg'}">
                    {metrics['strategy_return_pct']:+.2f}%
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
                <div class="label">最大回撤</div>
                <div class="value">{metrics['max_drawdown_pct']:.2f}%</div>
            </div>
            <div class="metric-box">
                <div class="label">账户总值</div>
                <div class="value">{report['total_value']:,.2f}</div>
            </div>
        </div>

        <h3>当前持仓</h3>
        <table>
            <thead>
                <tr>
                    <th>标的</th>
                    <th>股数</th>
                    <th>成本</th>
                </tr>
            </thead>
            <tbody>
                {holdings_rows}
            </tbody>
        </table>

        <h3>今日调仓</h3>
        <table>
            <thead>
                <tr>
                    <th>操作</th>
                    <th>标的</th>
                    <th>数量</th>
                    <th>价格</th>
                </tr>
            </thead>
            <tbody>
                {trades_rows}
            </tbody>
        </table>

        <div class="chart">
            <h3>收益曲线</h3>
            <img src="cid:chart_img" alt="Equity Curves">
        </div>

        <div class="footer">
            由 ETF 每日测评系统自动生成 | {datetime.now().strftime("%Y-%m-%d %H:%M")}
        </div>
    </body>
    </html>
    """

    # 3. 构建邮件
    msg = MIMEMultipart()
    msg["Subject"] = f"ETF 每日测评报告 - {date}"
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    msg.attach(MIMEText(html_body, "html", "utf-8"))

    # 4. 附加图片
    if CHART_PATH.exists():
        with open(CHART_PATH, "rb") as f:
            img_data = f.read()
        img = MIMEImage(img_data, name="equity_curves.png")
        img.add_header("Content-ID", "<chart_img>")
        msg.attach(img)
    else:
        print(f"警告: 未找到图表文件 {CHART_PATH}")

    # 5. 发送邮件
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
    parser.add_argument("--to", type=str, default=None, help="覆盖接收人邮箱 (多个用逗号分隔)")
    args = parser.parse_args()

    if args.to:
        EMAIL_TO = args.to

    send_report()
