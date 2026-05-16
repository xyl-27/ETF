from pathlib import Path
import shutil
from datetime import datetime
import sys
import io
import pandas as pd

# 修复 Windows GBK 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from playwright.sync_api import sync_playwright
import time
import os

STATE_FILE = "joinquant_state.json"

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "etf_data"
BASE_FILE = DATA_DIR / "etf_74.csv"
NEW_FILE = DATA_DIR / "etf_data_74_new.csv"

def merge_etf_data(base_path=None, new_path=None, backup=True):
    base_path = base_path or BASE_FILE
    new_path = new_path or NEW_FILE

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
    print(f"\n基础数据范围: {df_base['日期'].min().date()} ~ {max_base_date.date()} ({len(df_base)} 条)")
    print(f"增量数据范围: {df_new['日期'].min().date()} ~ {df_new['日期'].max().date()} ({len(df_new)} 条)")

    # 删除 base 中与新文件重复日期的旧数据，再用新文件整体替换
    # （修复盘前抓取 flat 数据被锁定的问题：第二次下载的正确数据可以覆盖已有日期）
    new_dates = set(df_new["日期"].unique())
    df_base_clean = df_base[~df_base["日期"].isin(new_dates)]
    df_merged = pd.concat([df_base_clean, df_new], ignore_index=True)

    overlap_rows = df_base["日期"].isin(new_dates).sum()
    new_dates_set = new_dates - set(df_base["日期"].unique())
    if overlap_rows == 0 and not new_dates_set:
        print("没有需要更新的数据")
        return False
    if new_dates_set:
        print(f"新增交易日: {sorted(new_dates_set)[0]} ~ {sorted(new_dates_set)[-1]} ({len(new_dates_set)} 天)")
    if overlap_rows:
        print(f"覆盖更新: {overlap_rows} 条记录（已有日期数据替换为新版本）")

    if backup:
        backup_path = base_path.with_suffix(".bak")
        df_base.to_csv(backup_path, index=False)
        print(f"已备份原文件: {backup_path}")
    df_merged["日期"] = df_merged["日期"].dt.strftime("%Y-%m-%d")
    df_merged = df_merged.sort_values(["股票代码", "日期"]).reset_index(drop=True)

    df_merged.to_csv(base_path, index=False)
    print(f"\n更新完成: {base_path}")
    print(f"总数据范围: {df_merged['日期'].min()} ~ {df_merged['日期'].max()} ({len(df_merged)} 条)")

    # 数据质量检查
    try:
        _dates = sorted(df_merged["日期"].unique())
        _last, _prev = _dates[-1], _dates[-2] if len(_dates) >= 2 else None
        _last_dt = pd.to_datetime(_last)
        _issues = []
        if _last_dt.weekday() >= 5:
            _issues.append(f"最新日期 {_last} 是{['周六','周日'][_last_dt.weekday()-5]}，非交易日")
        if _prev:
            _cur = df_merged[df_merged["日期"] == _last]
            _prv = df_merged[df_merged["日期"] == _prev]
            _merged = _cur[["股票代码","开盘","收盘","最高","最低"]].merge(
                _prv[["股票代码","收盘"]].rename(columns={"收盘":"prev_close"}), on="股票代码")
            _all_flat = (_merged["收盘"] == _merged["开盘"]).all()
            _all_vs_prev = (_merged["收盘"] == _merged["prev_close"]).all()
            if _all_flat and _all_vs_prev:
                _issues.append(f"最新日期 {_last} 所有 {len(_merged)} 只股票 open=close=high=low=昨收，数据无效（可能是盘前或休市抓取）")
            elif _all_flat:
                _flat_pct = (_merged["收盘"] == _merged["开盘"]).mean() * 100
                _issues.append(f"最新日期 {_last} {_flat_pct:.0f}% 的股票 open=close，数据可能不完整")
            _total_chg = (_merged["收盘"] - _merged["prev_close"]).abs().sum()
            if _total_chg == 0:
                _issues.append(f"最新日期 {_last} 所有股票收盘价和前一交易日完全相同，数据疑似无效")
        if _issues:
            print("⚠️ 数据质量警告:")
            for _msg in _issues:
                print(f"  - {_msg}")
            print("💡 建议等盘中或收盘后重新运行")
    except Exception as _e:
        print(f"数据质量检查失败: {_e}")

    return True

def login_with_cache(username, password):
    """带缓存登录，首次登录后保存状态"""
    
    playwright = sync_playwright().start()
    
    # 启动浏览器
    browser = playwright.chromium.launch(
        channel="msedge",
        headless=False,
        args=['--disable-blink-features=AutomationControlled']
    )
    
    # 检查是否有保存的状态
    if os.path.exists(STATE_FILE):
        print("发现已保存的登录状态，直接加载...")
        context = browser.new_context(storage_state=STATE_FILE)
        page = context.new_page()
        
        # 验证登录是否有效
        page.goto("https://www.joinquant.com/view/user/floor?type=mainFloor")
        page.wait_for_load_state("networkidle")
        time.sleep(2)
        
        # 检查是否真的登录成功
        if "login" not in page.url:
            print("✅ 使用缓存登录成功")
            return page, playwright, browser
        else:
            print("⚠ 缓存已失效，重新登录...")
            os.remove(STATE_FILE)
    
    # 没有缓存或缓存失效，正常登录
    print("首次登录或缓存失效，开始登录...")
    context = browser.new_context()
    page = context.new_page()
    
    # 登录流程...
    print("访问登录页...")
    page.goto("https://www.joinquant.com/user/login")
    page.wait_for_load_state("networkidle")
    time.sleep(2)
    
    print("切换到密码登录...")
    try:
        password_tab = page.locator("text=密码登录")
        if password_tab.count():
            password_tab.click()
    except:
        pass
    time.sleep(1)
    
    print("填写手机号...")
    page.fill("input[placeholder*='手机']", username.strip())
    print("填写密码...")
    page.fill("input[type='password']", password)
    
    print("勾选协议...")
    checkbox = page.locator("input[type='checkbox']")
    if checkbox.count():
        checkbox.first.check()
    
    time.sleep(0.5)
    
    print("点击登录...")
    page.click("button.login-submit.btnPwdSubmit")
    
    print("等待登录完成...")
    time.sleep(5)
    
    # 保存登录状态
    print("保存登录状态...")
    context.storage_state(path=STATE_FILE)
    print(f"✅ 登录状态已保存到 {STATE_FILE}")
    
    return page, playwright, browser


def run_notebook_and_download(page, notebook_url, timeout=180):
    """
    运行聚宽研究文件，通过文件API直接下载
    """
    import requests
    
    print("=" * 50)
    print("访问研究文件...")
    page.goto(notebook_url)
    page.wait_for_load_state("networkidle")
    time.sleep(5)
    
    # 切换到 iframe
    print("切换到 iframe...")
    page.wait_for_selector("#research", timeout=30000)
    frame = page.frame_locator("#research")
    print("✓ 已切换到 iframe")
    
    # 点击重启并运行全部
    print("点击重启并运行全部按钮...")
    run_btn = frame.locator('button[title="重启内核,然后重新运行整个代码(显示确认对话框)"]')
    if run_btn.count() > 0:
        run_btn.first.click()
        print("✓ 已点击重启按钮")
        time.sleep(2)
        
        # 点击确认按钮
        confirm_btn = frame.locator('button:has-text("重启并运行所有单元格")')
        if confirm_btn.count() > 0:
            confirm_btn.first.click()
            print("✓ 已确认重启并运行")
    
    # 等待完成提示
    print("等待数据获取完成...")
    while True:
        outputs = frame.locator(".output_subarea.output_text")
        for i in range(outputs.count()):
            try:
                text = outputs.nth(i).inner_text()
                if "数据获取完成" in text:
                    print(f"\n✓ {text.strip()}")
                    time.sleep(12)
                    break
            except:
                pass
        else:
            print(".", end="", flush=True)
            time.sleep(5)
            continue
        break
    
    # 直接通过文件URL下载
    print("\n下载文件...")
    # 从浏览器获取cookies
    cookies = page.context.cookies()
    cookie_str = "; ".join([f"{c['name']}={c['value']}" for c in cookies])
    
    # 文件下载URL（根据你的截图）
    file_url = "https://www.joinquant.com/user/73090038144/files/ETF/etf_74.csv?download=1"
    
    headers = {
        'Cookie': cookie_str,
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(file_url, headers=headers, stream=True)
        
        if response.status_code == 200:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            target_file = NEW_FILE
            
            # 保存文件
            with open(target_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ 文件已下载: {target_file}")
            print(f"文件大小: {target_file.stat().st_size / 1024:.2f} KB")

            # 自动合并到 etf_74.csv
            merge_etf_data()
            return str(target_file)
        else:
            print(f"❌ 下载失败: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ 下载异常: {e}")
        return None
if __name__ == "__main__":
    import os
    USERNAME = os.environ.get("JQ_USERNAME", "your_phone")
    PASSWORD = os.environ.get("JQ_PASSWORD", "your_password")
    NOTEBOOK_URL = "https://www.joinquant.com/research?target=research&url=/user/73090038144/notebooks/ETF/get_etf_data.ipynb"
    
    page, playwright, browser = login_with_cache(USERNAME, PASSWORD)
    
    try:
        downloaded_file = run_notebook_and_download(page, NOTEBOOK_URL, 9)
        if downloaded_file:
            print(f"下载完成: {downloaded_file}")
        else:
            print("下载失败")
        
    finally:
        browser.close()
        playwright.stop()
