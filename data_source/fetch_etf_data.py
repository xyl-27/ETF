"""获取 ETF 日线数据，生成 etf_data_74_new.csv

API 返回的 close/open 等已经是前复权价格。
factor = 前复权 / 原始，所以原始 = 前复权 / factor。

前复权列: 收盘（回测/模型）
原始列: 收盘_原始（展示用）
"""

import pandas as pd
import base64
from datetime import datetime
from IPython.display import HTML, display, Javascript

# from jqdata import get_price  # 按实际数据源调整


def fetch_single_etf(code, start_date, end_date):
    """获取单只ETF数据并计算前复权价格"""
    df = get_price(
        code,
        start_date=start_date,
        end_date=end_date,
        frequency="daily",
        fields=[
            "open", "close", "high", "low", "volume", "money",
            "high_limit", "low_limit", "pre_close", "paused", "factor",
        ],
    )
    if df is None or len(df) == 0:
        return None

    df["股票代码"] = code
    df["日期"] = df.index
    factor = df["factor"]

    # 前复权价格（API 已返回，用于回测和模型）
    df["开盘"] = df["open"]
    df["收盘"] = df["close"]
    df["最高"] = df["high"]
    df["最低"] = df["low"]
    df["前收盘"] = df["pre_close"]

    # 原始价格（反算，用于展示）
    df["开盘_原始"] = df["open"] / factor
    df["收盘_原始"] = df["close"] / factor
    df["最高_原始"] = df["high"] / factor
    df["最低_原始"] = df["low"] / factor
    df["前收盘_原始"] = df["pre_close"] / factor

    # 复权因子
    df["复权因子"] = factor

    # 成交量、成交额（不受复权影响）
    df["成交量"] = df["volume"]
    df["成交额"] = df["money"]

    # 涨跌停价（保持原始值）
    df["涨停价"] = df["high_limit"]
    df["跌停价"] = df["low_limit"]
    df["停牌"] = df["paused"]

    # 技术指标（用前复权价格，跨因子边界正确）
    df["涨跌额"] = df["收盘"] - df["前收盘"]
    df["涨跌幅"] = (df["收盘"] - df["前收盘"]) / df["前收盘"] * 100
    df["振幅"] = (df["最高"] - df["最低"]) / df["前收盘"] * 100

    # 换手率（暂不处理）
    df["换手率"] = 0.0

    return df[[
        "股票代码", "日期",
        "开盘", "收盘", "最高", "最低", "前收盘",
        "涨跌额", "涨跌幅", "振幅",
        "成交量", "成交额", "涨停价", "跌停价", "停牌", "换手率", "复权因子",
        "开盘_原始", "收盘_原始", "最高_原始", "最低_原始", "前收盘_原始"
    ]]


def fetch_all_etf(etf_list_file="etf_list_before_2022_74.csv"):
    """获取全部ETF数据并合并"""
    etf_list = pd.read_csv(etf_list_file)
    df_list = []
    for code in etf_list["代码"].values:
        df = fetch_single_etf(code, START_DATE, END_DATE)
        if df is not None:
            df_list.append(df)
    all_data = pd.concat(df_list, ignore_index=True)
    all_data = all_data.sort_values(["股票代码", "日期"])
    all_data = all_data.fillna(0)
    return all_data


def auto_download_csv(b64, filename):
    """自动触发 CSV 文件下载"""
    download_js = f'''
    (function() {{
        var blob = new Blob([atob("{b64}")], {{type: "text/csv;charset=utf-8"}});
        var link = document.createElement("a");
        var url = URL.createObjectURL(blob);
        link.href = url;
        link.download = "{filename}";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }})();
    '''
    display(Javascript(download_js))


# ==================== 主流程 ====================

END_DATE = datetime.now().strftime("%Y-%m-%d")
START_DATE = "2026-01-01"

print(f"开始获取数据: {START_DATE} → {END_DATE}")

all_data = fetch_all_etf()

# 保存为 CSV
csv_data = all_data.to_csv(index=False, encoding="utf-8-sig")
b64 = base64.b64encode(csv_data.encode()).decode()
filename = "etf_data_74_new.csv"

# 方式1：显示下载链接（手动点击）
display(HTML(
    f'<a href="data:text/csv;base64,{b64}" download="{filename}" '
    f'style="font-size:16px;padding:8px 16px;background:#4CAF50;color:white;'
    f'text-decoration:none;border-radius:4px;">📥 点击下载 CSV 文件</a>'
))

# 方式2：自动触发下载（取消注释启用）
# auto_download_csv(b64, filename)

# 同时保存到本地文件
all_data.to_csv(filename, index=False, encoding="utf-8-sig")

print(f"\n✅ 数据获取完成！")
print(f"   总记录数: {len(all_data)} 条")
print(f"   ETF 数量: {all_data['股票代码'].nunique()} 只")
print(f"   日期范围: {all_data['日期'].min()} → {all_data['日期'].max()}")
print(f"   文件保存: {filename}")