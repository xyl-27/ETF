"""
增量更新 etf_74.csv
将 etf_data_74_new.csv 中的新数据追加到 etf_74.csv
"""

import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "etf_data"
BASE_FILE = DATA_DIR / "etf_74.csv"
NEW_FILE = DATA_DIR / "etf_data_74_new.csv"

def merge_etf_data(base_path=BASE_FILE, new_path=NEW_FILE, backup=True):
    print(f"基础数据: {base_path}")
    print(f"增量数据: {new_path}")

    if not base_path.exists():
        print(f"基础数据不存在: {base_path}")
        return False

    if not new_path.exists():
        print(f"增量数据不存在: {new_path}")
        return False

    df_base = pd.read_csv(base_path, dtype={"股票代码": str})
    df_new = pd.read_csv(new_path, dtype={"股票代码": str})

    df_base["日期"] = pd.to_datetime(df_base["日期"])
    df_new["日期"] = pd.to_datetime(df_new["日期"])

    max_base_date = df_base["日期"].max()
    min_new_date = df_new["日期"].min()
    max_new_date = df_new["日期"].max()

    print(f"\n基础数据范围: {df_base['日期'].min().date()} ~ {max_base_date.date()} ({len(df_base)} 条)")
    print(f"增量数据范围: {min_new_date.date()} ~ {max_new_date.date()} ({len(df_new)} 条)")

    # 过滤出基础数据之后的新数据
    df_incremental = df_new[df_new["日期"] > max_base_date]

    if df_incremental.empty:
        print("\n没有需要更新的数据")
        return False

    print(f"新增数据: {len(df_incremental)} 条 ({df_incremental['日期'].min().date()} ~ {df_incremental['日期'].max().date()})")

    # 备份原文件
    if backup:
        backup_path = base_path.with_suffix(".bak")
        df_base.to_csv(backup_path, index=False)
        print(f"已备份原文件: {backup_path}")

    # 合并并保存
    df_merged = pd.concat([df_base, df_incremental], ignore_index=True)
    df_merged["日期"] = df_merged["日期"].dt.strftime("%Y-%m-%d")
    df_merged = df_merged.sort_values(["股票代码", "日期"]).reset_index(drop=True)

    df_merged.to_csv(base_path, index=False)
    print(f"\n更新完成: {base_path}")
    print(f"总数据范围: {df_merged['日期'].min()} ~ {df_merged['日期'].max()} ({len(df_merged)} 条)")
    return True

if __name__ == "__main__":
    merge_etf_data()
