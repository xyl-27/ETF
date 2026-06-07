"""
model_manager.py — 模型存档、重训（锁定超参数）、对比

Commands:
  archive         存档 model/bayes_* → model/archive/{date}/
  reproduce       从实盘模型锁定超参 + 改日期重训
  batch-reproduce 批量 reproduce 所有 2026 实验到新日期
  compare         对比两版模型在相同区间上的回测表现
  upgrade         archive + reproduce + compare 一键完成
"""

import os
import sys
import json
import re
import shutil
import gc
import time
import torch
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(SCRIPT_DIR))

_LOG_FILE = None


def _log(msg: str, end: str = "\n"):
    print(msg, end=end, flush=True)
    if _LOG_FILE:
        with open(_LOG_FILE, "a") as f:
            f.write(msg + end)


# ============================================================
#  Helpers
# ============================================================

def _parse_model_type_from_dir(dir_name: str) -> str:
    parts = dir_name.split("_", 1)
    rest = parts[1] if len(parts) > 1 else parts[0]
    return rest.split("_")[0]


def _parse_val_dates_from_dir(dir_name: str):
    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})", dir_name)
    return m.groups() if m else (None, None)


def _get_exp_score(exp_dir: Path, search_root: Path) -> float:
    """从 search_results.json 读取指定实验的 score，读取失败则回退到 exp 序号顺序。"""
    results_file = search_root / "search_results.json"
    if results_file.exists():
        try:
            with open(results_file) as f:
                results = json.load(f)
            exp_idx = int(exp_dir.name.split("_")[-1])
            for r in results:
                if r.get("exp_idx") == exp_idx and r.get("success"):
                    return float(r["score"])
        except Exception:
            pass
    exp_idx = int(exp_dir.name.split("_")[-1])
    return -float(exp_idx)


# ============================================================
#  Archive
# ============================================================

def cmd_archive(args):
    model_dir = PROJECT_ROOT / "model"
    archive_base = model_dir / "archive"
    date_str = datetime.now().strftime("%Y-%m-%d")
    dest = archive_base / date_str

    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    manifest = {
        "archive_date": date_str,
        "source_root": str(model_dir),
        "models": [],
    }

    def _copy_model(src: Path, dst: Path):
        if args.dry_run:
            _log(f"  [DRY-RUN] cp -r {src} → {dst}")
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, symlinks=True)

    def _record(dir_name: str, src: Path, is_live: bool = False):
        exp_count = len([x for x in src.iterdir() if x.is_dir() and x.name.startswith("exp_")])
        import subprocess as _sp
        try:
            _r = _sp.run(["du", "-sb", str(src)], capture_output=True, text=True, timeout=30)
            size_mb = int(_r.stdout.split()[0]) / 1e6
        except Exception:
            size_mb = 0
        manifest["models"].append({
            "dir": dir_name, "type": _parse_model_type_from_dir(dir_name),
            "val_start": _parse_val_dates_from_dir(dir_name)[0],
            "val_end": _parse_val_dates_from_dir(dir_name)[1],
            "experiments": exp_count, "size_mb": round(size_mb, 1), "is_live": is_live,
        })
        _log(f"  ✓ {dir_name} ({exp_count} exp, {size_mb:.0f}MB)")

    _log("存档 model/bayes_* …")
    for d in sorted(os.listdir(str(model_dir))):
        if not d.startswith("bayes_"):
            continue
        src = model_dir / d
        dst = dest / d
        _copy_model(src, dst)
        _record(d, src)

    live_src = PROJECT_ROOT / "juejin" / "live"
    live_dst = dest / "live"
    if live_src.exists():
        _log("存档 juejin/live/bayes_* …")
        for d in sorted(os.listdir(str(live_src))):
            if not d.startswith("bayes_"):
                continue
            src = live_src / d
            dst = live_dst / d
            _copy_model(src, dst)
            _record(f"live/{d}", src, is_live=True)

    if not args.dry_run:
        with open(dest / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        _log(f"\nManifest: {dest / 'manifest.json'}")

    _log(f"\n存档完成: {dest}")


# ============================================================
#  Reproduce — 核心逻辑
# ============================================================

def _reproduce_one(
    source_dir: str,
    new_val_start: str,
    new_val_end: str,
    force: bool = False,
    quiet: bool = False,
) -> Tuple[bool, str, str]:
    """
    锁定超参 + 改日期重训单个实验。
    Returns: (success, target_dir, message)
    """
    source_dir = Path(source_dir)
    if not source_dir.exists():
        return False, "", f"源目录不存在: {source_dir}"

    config_path = source_dir / "config.json"
    if not config_path.exists():
        return False, "", f"{config_path} 不存在"
    with open(config_path) as f:
        source_config = json.load(f)

    model_type = source_config.get("model_type", "tcn")
    exp_idx = source_dir.name.split("_")[-1]
    output_name = f"{model_type}_from_exp_{exp_idx}_{new_val_start}_{new_val_end}"
    search_dir = str(PROJECT_ROOT / "model" / "reproduced" / output_name)

    if os.path.exists(search_dir):
        if force:
            shutil.rmtree(search_dir)
        else:
            return False, search_dir, "目标目录已存在（使用 force 覆盖）"

    import config as config_module

    config = config_module.config.copy()
    for k, v in source_config.items():
        if k in ("val_start_date", "val_end_date", "output_dir", "output_base"):
            continue
        config[k] = v
    config["val_start_date"] = new_val_start
    config["val_end_date"] = new_val_end
    config["output_dir"] = f"./model/reproduced/{output_name}"
    os.makedirs(search_dir, exist_ok=True)

    if not quiet:
        _log(f"  预处理 ({new_val_start} ~ {new_val_end}) …", end="")
    from train_search_v2 import preprocess_and_save
    preprocessed_data, scaler = preprocess_and_save(config, search_dir)
    if not quiet:
        _log(" 完成")

    model_defaults = config_module.get_model_config(model_type)
    param_keys = set(model_defaults.keys()) | {"learning_rate"}
    if "num_experts" in source_config:
        param_keys.add("num_experts")
    params = {k: source_config[k] for k in param_keys if k in source_config}

    if not quiet:
        _log(f"  训练 (源: {source_dir.name}) …", end="")
    from train_search_v2 import run_experiment
    try:
        result = run_experiment(
            params, config, preprocessed_data, scaler,
            search_dir, "reproduced", config_module,
        )
    except Exception as e:
        import traceback
        return False, search_dir, f"训练失败: {e}\n{traceback.format_exc()}"

    source_info = {
        "source_dir": str(source_dir), "model_type": model_type,
        "source_exp_idx": exp_idx,
        "source_val_start": source_config.get("val_start_date"),
        "source_val_end": source_config.get("val_end_date"),
        "new_val_start": new_val_start, "new_val_end": new_val_end,
        "fixed_params": params,
        "reproduced_at": datetime.now().isoformat(),
        "training_result": {k: result.get(k) for k in ("score", "metric", "sliding_score", "best_epoch", "success")},
    }
    with open(os.path.join(search_dir, "source.json"), "w") as f:
        json.dump(source_info, f, indent=2, ensure_ascii=False)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if not quiet:
        _log(" 完成")
        _log(f"  指标: {source_info['training_result']}")

    return True, search_dir, f"OK score={result.get('score', '?')}"


def cmd_reproduce(args):
    ok, target_dir, msg = _reproduce_one(
        args.source, args.val_start, args.val_end, args.force
    )
    if ok:
        _log(f"\n重训完成: {target_dir}")
        _log(f"  {msg}")
    else:
        _log(f"错误: {msg}")
        sys.exit(1)


# ============================================================
#  Batch Reproduce
# ============================================================

def cmd_batch_reproduce(args):
    global _LOG_FILE
    _LOG_FILE = args.log

    val_start = args.val_start
    val_end = args.val_end
    max_per_type = args.max_per_type

    # 1. Scan source experiments
    model_dir = PROJECT_ROOT / "model"
    all_sources: List[Tuple[str, str, str]] = []  # (model_type, exp_dir, exp_name)

    DL_TYPES = {"dlinear", "timesnet", "tcn", "gru", "patchtst", "itransformer", "mamba", "nlinear"}

    _log("扫描 2026 实验 …")
    for d in sorted(os.listdir(str(model_dir))):
        if "2026-01-01_2026-03-31" not in d or not d.startswith("bayes_"):
            continue
        mt = _parse_model_type_from_dir(d)
        if mt not in DL_TYPES:
            continue
        exp_root = model_dir / d
        exps = sorted(os.listdir(str(exp_root)))
        for exp in exps:
            if not exp.startswith("exp_"):
                continue
            exp_dir = exp_root / exp
            if not exp_dir.is_dir() or not (exp_dir / "config.json").exists():
                continue
            all_sources.append((mt, str(exp_dir), exp))

    _log(f"发现 {len(all_sources)} 个实验")

    # 2. Group by model_type, apply limit (按 search_results.json 的 score 降序选 Top-N)
    from collections import defaultdict
    by_type: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
    for mt, exp_dir, exp_name in all_sources:
        score = _get_exp_score(exp_dir, Path(exp_dir).parent)
        by_type[mt].append((exp_dir, exp_name, score))

    plan = []
    for mt in sorted(by_type.keys()):
        exps = sorted(by_type[mt], key=lambda x: -x[2])  # sort by score desc
        limit = max_per_type if max_per_type > 0 else len(exps)
        for exp_dir, exp_name, _ in exps[:limit]:
            plan.append((mt, exp_dir, exp_name))

    _log(f"\n计划: {len(plan)} 个实验 (按 score 选 Top-{max_per_type if max_per_type > 0 else '全部'} / 类型)")
    for mt in sorted(by_type.keys()):
        count = min(max_per_type, len(by_type[mt])) if max_per_type > 0 else len(by_type[mt])
        top_score = by_type[mt][0][2] if by_type[mt] else 0
        _log(f"  {mt}: {count}/{len(by_type[mt])}  (best score: {top_score:.4f})")

    # 3. Process
    success, failed, skipped = 0, 0, 0
    total = len(plan)
    start_global = time.time()

    for idx, (mt, exp_dir, exp_name) in enumerate(plan):
        output_name = f"{mt}_from_{exp_name}_{val_start}_{val_end}"
        target_dir = str(PROJECT_ROOT / "model" / "reproduced" / output_name)

        # Skip if already exists
        if args.resume and os.path.exists(target_dir) and os.path.isdir(target_dir):
            # Verify it has model files
            has_files = any(
                os.path.exists(os.path.join(target_dir, f))
                for f in ("best_model.pth", "best_model_sliding.pth", "best_model_ndcg.pth")
            )
            if has_files:
                skipped += 1
                _log(f"[{idx + 1}/{total}] SKIP {output_name} (已存在)")
                continue

        elapsed = time.time() - start_global
        _log(f"[{idx + 1}/{total}] ({elapsed / 60:.1f}min) {mt}/{exp_name} …", end=" ")

        t0 = time.time()
        ok, _, msg = _reproduce_one(exp_dir, val_start, val_end, force=False, quiet=True)
        dt = time.time() - t0

        if ok:
            success += 1
            _log(f"✓ ({dt:.0f}s) {msg}")
        else:
            failed += 1
            _log(f"✗ ({dt:.0f}s) {msg}")

        # Estimate remaining time
        if idx > 0:
            avg = (time.time() - start_global) / (idx + 1)
            remaining = avg * (total - idx - 1)
            _log(f"  预计剩余: {remaining / 60:.0f}min")

    total_elapsed = time.time() - start_global
    _log(f"\n{'=' * 50}")
    _log(f"批量 reproduce 完成")
    _log(f"  总计: {total}, 成功: {success}, 跳过: {skipped}, 失败: {failed}")
    _log(f"  耗时: {total_elapsed / 60:.1f}min")
    _log(f"{'=' * 50}")

    if failed > 0:
        sys.exit(1)


# ============================================================
#  Compare — 对比两版模型
# ============================================================

def cmd_compare(args):
    from daily_eval import load_full_config

    cfg = load_full_config()
    model_type = args.model_type
    val_a = args.val_a
    val_b = args.val_b
    bt_start = cfg.get("start_date", "2026-04-01")

    data_path = PROJECT_ROOT / "etf_data" / "etf_74.csv"
    if data_path.exists():
        import pandas as pd
        df = pd.read_csv(data_path)
        df["日期"] = pd.to_datetime(df["日期"])
        bt_end = df["日期"].max().strftime("%Y-%m-%d")
    else:
        bt_end = "2026-05-12"
    _log(f"回测区间: {bt_start} → {bt_end}")

    def _discover(val_dates: str, include_reproduced: bool = True):
        results = []
        tag = f"bayes_{model_type}_74_3_{val_dates}"
        exp_root = PROJECT_ROOT / "model" / tag
        if exp_root.exists():
            for exp in sorted(os.listdir(str(exp_root))):
                if not exp.startswith("exp_"):
                    continue
                exp_dir = exp_root / exp
                if not exp_dir.is_dir():
                    continue
                for mf in ("best_model.pth", "best_model_sliding.pth", "best_model_ndcg.pth"):
                    if (exp_dir / mf).exists():
                        results.append((str(exp_dir), mf))
        if include_reproduced:
            repro_root = PROJECT_ROOT / "model" / "reproduced"
            if repro_root.exists():
                for d in sorted(os.listdir(str(repro_root))):
                    if not d.startswith(f"{model_type}_from_"):
                        continue
                    if val_dates not in d:
                        continue
                    exp_dir = repro_root / d
                    if not exp_dir.is_dir():
                        continue
                    for mf in ("best_model.pth", "best_model_sliding.pth", "best_model_ndcg.pth"):
                        if (exp_dir / mf).exists():
                            results.append((str(exp_dir), mf))
        return results

    exps_a = _discover(val_a)
    exps_b = _discover(val_b)

    _log(f"  版本 A ({val_a}): {len(exps_a)} 实验")
    _log(f"  版本 B ({val_b}): {len(exps_b)} 实验")

    if not exps_a or not exps_b:
        _log("错误: 某版本无实验，无法对比")
        sys.exit(1)

    scaler_path = None
    for exp_d, _ in exps_a + exps_b:
        sp = f"{exp_d}/scaler.pkl"
        if os.path.exists(sp):
            scaler_path = sp
            break
    if not scaler_path:
        _log("错误: 未找到 scaler.pkl")
        sys.exit(1)

    _log("\n加载数据 …")
    from backtest import run_backtest_from_predictions, ETFBacktester

    cached_data, cached_features = ETFBacktester.load_data_once(
        data_path=str(data_path), scaler_path=scaler_path,
        feature_num="39", verbose=False, store_unscaled=True,
    )

    rbd = cfg.get("rebalance_days", 5)
    top_k = cfg.get("top_k", 3)
    ppct = cfg.get("position_pct", 0.95)
    icap = cfg.get("initial_capital", 100000)
    tmode = cfg.get("trade_mode", "open")

    def _backtest_one(exp_dir, model_file):
        try:
            bt = ETFBacktester.from_cached_data(
                model_dir=exp_dir, cached_data=cached_data, cached_features=cached_features,
                device="cpu", model_file=model_file, verbose=False,
            )
            preds = bt.generate_predictions_dict(
                start_date=bt_start, end_date=bt_end,
                rebalance_days=rbd, first_rebalance_date=bt_start,
            )
            r = run_backtest_from_predictions(
                predictions_dict=preds, data_path=str(data_path),
                start_date=bt_start, end_date=bt_end,
                top_k=top_k, rebalance_days=rbd, position_pct=ppct,
                initial_capital=icap, trade_mode=tmode,
                weight_strategy="equal", verbose=False,
            )
            return {
                "return": r.strategy_return, "dd": r.max_drawdown,
                "hs300": r.hs300_return, "excess": r.excess_return,
                "win_rate": r.rebalance_stats.get("win_rate", 0),
                "avg_return": r.rebalance_stats.get("avg_return", 0),
                "rebalances": r.rebalance_stats.get("total", 0),
            }
        except Exception as e:
            _log(f"  ✗ {Path(exp_dir).name}: {e}")
            return None

    max_exps = args.max_exps or max(len(exps_a), len(exps_b))
    results_a, results_b = [], []

    for label, exps, results in [
        (f"版本 A ({val_a})", exps_a[:max_exps], results_a),
        (f"版本 B ({val_b})", exps_b[:max_exps], results_b),
    ]:
        _log(f"\n回测 {label} …")
        for exp_dir, mf in exps:
            tag = f"{Path(exp_dir).parent.name}/{Path(exp_dir).name}/{mf}"
            _log(f"  {tag} …", end=" ", flush=True)
            r = _backtest_one(exp_dir, mf)
            if r:
                r["version"] = "A" if label.startswith("版本 A") else "B"
                r["label"] = tag
                results.append(r)
                _log(f"收益 {r['return']:.2f}%, 回撤 {r['dd']:.2f}%")
            else:
                _log("失败")

    if not results_a or not results_b:
        _log("错误: 无有效回测结果")
        sys.exit(1)

    import pandas as pd
    df_a = pd.DataFrame(results_a)
    df_b = pd.DataFrame(results_b)

    _log(f"\n{'=' * 60}")
    _log(f"对比: {model_type}  {val_a} (A) vs {val_b} (B)")
    _log(f"回测区间: {bt_start} → {bt_end}")
    _log(f"{'=' * 60}")

    summary = pd.DataFrame({
        "指标": ["实验数", "收益均值(%)", "收益中位数(%)", "收益标准差(%)",
                 "回撤均值(%)", "回撤中位数(%)", "胜率均值(%)", "日均收益均值(%)"],
        f"A ({val_a})": [len(df_a), f"{df_a['return'].mean():.2f}", f"{df_a['return'].median():.2f}",
                         f"{df_a['return'].std():.2f}", f"{df_a['dd'].mean():.2f}", f"{df_a['dd'].median():.2f}",
                         f"{df_a['win_rate'].mean():.1f}", f"{df_a['avg_return'].mean():.2f}"],
        f"B ({val_b})": [len(df_b), f"{df_b['return'].mean():.2f}", f"{df_b['return'].median():.2f}",
                         f"{df_b['return'].std():.2f}", f"{df_b['dd'].mean():.2f}", f"{df_b['dd'].median():.2f}",
                         f"{df_b['win_rate'].mean():.1f}", f"{df_b['avg_return'].mean():.2f}"],
    })
    print(summary.to_string(index=False))

    best_a = max(results_a, key=lambda x: x["return"])
    best_b = max(results_b, key=lambda x: x["return"])
    _log(f"\n🏆 A 最佳: {best_a['label']} → 收益 {best_a['return']:.2f}%, 回撤 {best_a['dd']:.2f}%")
    _log(f"🏆 B 最佳: {best_b['label']} → 收益 {best_b['return']:.2f}%, 回撤 {best_b['dd']:.2f}%")

    output_path = PROJECT_ROOT / "output" / f"comparison_{model_type}_{val_a}_vs_{val_b}.json"
    with open(output_path, "w") as f:
        json.dump({
            "model_type": model_type, "val_a": val_a, "val_b": val_b,
            "backtest_start": bt_start, "backtest_end": bt_end,
            "results_a": results_a, "results_b": results_b,
            "summary_a": {k: float(v) for k, v in df_a.mean().items() if k not in ("version", "label")},
            "summary_b": {k: float(v) for k, v in df_b.mean().items() if k not in ("version", "label")},
        }, f, indent=2, ensure_ascii=False)
    _log(f"\n结果已保存: {output_path}")


# ============================================================
#  Upgrade
# ============================================================

def cmd_upgrade(args):
    _log("=" * 60)
    _log("  一键升级")
    _log("=" * 60)

    _log("\n[1/3] 存档旧模型 …")
    cmd_archive(args)

    _log("\n[2/3] 重训模型 …")
    ok, target_dir, msg = _reproduce_one(
        args.source, args.new_val_start, args.new_val_end, args.force
    )
    if ok:
        _log(f"重训完成: {target_dir}")
    else:
        _log(f"重训失败: {msg}")
        sys.exit(1)

    _log("\n[3/3] 对比新旧版本 …")
    source_dir = Path(args.source)
    config_path = source_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            sc = json.load(f)
        old_val_a = f"{sc.get('val_start_date')}_{sc.get('val_end_date')}"
    else:
        old_val_a = "unknown"

    args.val_a = old_val_a
    args.val_b = f"{args.new_val_start}_{args.new_val_end}"
    if not args.max_exps:
        args.max_exps = 20
    cmd_compare(args)
    _log("\n一键升级完成 ✅")


# ============================================================
#  CLI
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="模型管理器: 存档 / 重训 / 对比",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # archive
    p = sub.add_parser("archive", help="存档 model/bayes_* → model/archive/{date}/")
    p.add_argument("--dry-run", action="store_true", help="仅预览，不实际拷贝")

    # reproduce
    p = sub.add_parser("reproduce", help="从实盘模型锁定超参 + 改日期重训")
    p.add_argument("--source", required=True, help="源实验目录路径")
    p.add_argument("--val-start", required=True, help="新验证集开始日期 (YYYY-MM-DD)")
    p.add_argument("--val-end", required=True, help="新验证集结束日期 (YYYY-MM-DD)")
    p.add_argument("--force", action="store_true", help="覆盖已存在的目标目录")

    # batch-reproduce
    p = sub.add_parser("batch-reproduce", help="批量 reproduce 所有 2026 实验到新日期")
    p.add_argument("--val-start", required=True, help="新验证集开始日期 (YYYY-MM-DD)")
    p.add_argument("--val-end", required=True, help="新验证集结束日期 (YYYY-MM-DD)")
    p.add_argument("--max-per-type", type=int, default=0, help="每个模型类型最多实验数 (0=全部)")
    p.add_argument("--resume", action="store_true", help="跳过已完成的实验")
    p.add_argument("--log", default="/tmp/batch_reproduce.log", help="日志文件路径")

    # compare
    p = sub.add_parser("compare", help="对比两版模型在相同区间上的表现")
    p.add_argument("--model-type", required=True, help="模型类型 (tcn/gru/patchtst/…)")
    p.add_argument("--val-a", required=True, help="版本 A 日期范围 (如 2025-01-01_2025-06-30)")
    p.add_argument("--val-b", required=True, help="版本 B 日期范围 (如 2026-01-01_2026-03-31)")
    p.add_argument("--max-exps", type=int, default=0, help="每版最多回测实验数 (0=全部)")

    # upgrade
    p = sub.add_parser("upgrade", help="archive + reproduce + compare 一键完成")
    p.add_argument("--source", required=True, help="源实验目录路径")
    p.add_argument("--new-val-start", required=True, help="新验证集开始日期 (YYYY-MM-DD)")
    p.add_argument("--new-val-end", required=True, help="新验证集结束日期 (YYYY-MM-DD)")
    p.add_argument("--force", action="store_true", help="覆盖已存在的目标目录")
    p.add_argument("--max-exps", type=int, default=20, help="对比时每版最多回测实验数")
    p.add_argument("--dry-run", action="store_true", help="存档时仅预览")

    args = parser.parse_args()

    # Setup log file for batch commands
    if hasattr(args, "log") and args.command in ("batch-reproduce",):
        global _LOG_FILE
        _LOG_FILE = args.log
        os.makedirs(os.path.dirname(args.log), exist_ok=True)

    fns = {
        "archive": cmd_archive,
        "reproduce": cmd_reproduce,
        "batch-reproduce": cmd_batch_reproduce,
        "compare": cmd_compare,
        "upgrade": cmd_upgrade,
    }
    fns[args.command](args)


if __name__ == "__main__":
    main()
