import pandas as pd
import os

DATA_DIR = "/home/linuxyl/THU-BDC2026/data"

def merge_jq_factors():
    stock_data = pd.read_csv(os.path.join(DATA_DIR, "stock_data.csv"))
    hs300_jq = pd.read_csv(os.path.join(DATA_DIR, "hs300_jq.csv"))

    stock_data["股票代码"] = stock_data["股票代码"].astype(str).str.strip()
    hs300_jq["股票代码"] = hs300_jq["股票代码"].str.replace(".XSHG", "").str.replace(".XSHE", "").str.strip()

    stock_data["日期"] = pd.to_datetime(stock_data["日期"]).dt.strftime("%Y-%m-%d")
    hs300_jq["日期"] = pd.to_datetime(hs300_jq["日期"]).dt.strftime("%Y-%m-%d")

    jq_cols = hs300_jq.columns.tolist()[2:]

    merged = stock_data.merge(
        hs300_jq[["股票代码", "日期"] + jq_cols],
        on=["股票代码", "日期"],
        how="left"
    )

    merged.to_csv(
        os.path.join(DATA_DIR, "stock_data_with_jqfactors.csv"),
        index=False
    )
    print(f"Saved: {len(merged)} rows, {len(merged.columns)} columns")

if __name__ == "__main__":
    merge_jq_factors()