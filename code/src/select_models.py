"""
模型选择脚本 - 从所有搜索目录中挑选最佳模型

用法:
    python code/src/select_models.py                    # 默认: 选最好的10个
    python code/src/select_models.py --top-n 5          # 选最好的5个
    python code/src/select_models.py --mode fusion      # 融合模式(默认)
    python code/src/select_models.py --mode single      # 单模型模式
    python code/src/select_models.py --manual exp_12 exp_3 exp_7  # 手动指定
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from config import config


def find_all_experiments(model_base_dir: str) -> list:
    """扫描所有搜索目录，收集所有实验"""
    all_experiments = []

    if not os.path.exists(model_base_dir):
        return all_experiments

    for search_dir in sorted(os.listdir(model_base_dir)):
        search_path = os.path.join(model_base_dir, search_dir)
        if not os.path.isdir(search_path):
            continue

        # 尝试从 search_results.json 读取
        results_path = os.path.join(search_path, "search_results.json")
        if os.path.exists(results_path):
            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
                for r in results:
                    if r.get("success"):
                        exp_idx = r["exp_idx"]
                        all_experiments.append({
                            "exp_dir": os.path.join(search_path, f"exp_{exp_idx}"),
                            "model_file": "best_model_sliding.pth",
                            "score": r.get("sharpe", r.get("score", 0)),
                            "search_dir": search_path,
                            "params": r.get("params", {}),
                        })
            except Exception:
                pass

        # 尝试从 exp_N 子目录读取 final_score.txt
        for exp_name in sorted(os.listdir(search_path)):
            exp_path = os.path.join(search_path, exp_name)
            if not os.path.isdir(exp_path) or exp_name.startswith("exp_") and any(
                os.path.exists(os.path.join(exp_path, f"exp_{i}")) for i in range(100) if os.path.exists(os.path.join(exp_path, f"exp_{i}"))
            ):
                continue
            # Only look at directories starting with exp_
            if not exp_name.startswith("exp_"):
                continue

            # Skip if already added from search_results.json
            if any(e["exp_dir"] == exp_path for e in all_experiments):
                continue

            final_score_path = os.path.join(exp_path, "final_score.txt")
            if os.path.exists(final_score_path):
                try:
                    with open(final_score_path, "r") as f:
                        lines = f.read().strip().split("\n")
                    score = None
                    for line in lines:
                        if "Best sliding_final_score" in line:
                            score = float(line.split(":")[-1].strip())
                            break
                    if score is None and "Best weekly_final_score" in lines[0] or len(lines) > 1:
                        for line in lines:
                            if "weekly_final_score" in line:
                                score = float(line.split(":")[-1].strip())
                                break

                    if score is not None:
                        model_file = "best_model_sliding.pth"
                        if not os.path.exists(os.path.join(exp_path, model_file)):
                            model_file = "best_model.pth"

                        all_experiments.append({
                            "exp_dir": exp_path,
                            "model_file": model_file,
                            "score": score,
                            "search_dir": search_path,
                            "params": {},
                        })
                except Exception:
                    pass

    return all_experiments


def save_model_selection(models: list, mode: str, output_path: str):
    """保存模型选择到txt文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"mode: {mode}\n")
        f.write(f"count: {len(models)}\n")
        f.write("\n")
        f.write("models:\n")
        for m in models:
            rel_dir = os.path.relpath(m["exp_dir"], PROJECT_ROOT)
            f.write(f"{rel_dir},{m['model_file']},{m['score']:.6f}\n")


def main(args):
    model_base = config.get("output_base", "./model")
    # Handle nested output_base like ./model -> use parent dir
    if model_base.startswith("./"):
        model_base = model_base[2:]
    # Get the parent dir of output_base to scan all search dirs
    base_parent = str(PROJECT_ROOT)

    # Actually scan the model base dir and its parent for search_*/grid_*/bayes_* dirs
    candidates = []
    for search_dir in sorted(os.listdir(os.path.join(base_parent, "model"))):
        full_path = os.path.join(base_parent, "model", search_dir)
        if os.path.isdir(full_path) and any(search_dir.startswith(p) for p in ("search_", "grid_", "bayes_")):
            candidates.append(full_path)

    all_experiments = []
    for search_path in candidates:
        results_path = os.path.join(search_path, "search_results.json")
        if os.path.exists(results_path):
            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
                for r in results:
                    if r.get("success"):
                        exp_idx = r["exp_idx"]
                        exp_dir = os.path.join(search_path, f"exp_{exp_idx}")
                        model_file = "best_model_sliding.pth"
                        if not os.path.exists(os.path.join(exp_dir, model_file)):
                            model_file = "best_model.pth"
                        if os.path.exists(os.path.join(exp_dir, model_file)):
                            all_experiments.append({
                                "exp_dir": exp_dir,
                                "model_file": model_file,
                                "score": r.get("sharpe", r.get("score", 0)),
                                "search_dir": search_path,
                                "params": r.get("params", {}),
                            })
            except Exception:
                pass

    if not all_experiments:
        print("未找到任何训练好的模型")
        return

    # 按分数排序
    all_experiments.sort(key=lambda x: x["score"], reverse=True)

    print(f"共找到 {len(all_experiments)} 个实验")
    print(f"\nTop 10:")
    for i, exp in enumerate(all_experiments[:10]):
        rel = os.path.relpath(exp["exp_dir"], PROJECT_ROOT)
        print(f"  {i+1}. {rel} score={exp['score']:.4f}")

    # 手动指定模式
    if args.manual:
        selected = []
        for exp_name in args.manual:
            found = False
            for exp in all_experiments:
                if exp["exp_dir"].endswith(exp_name) or os.path.basename(exp["exp_dir"]) == exp_name:
                    selected.append(exp)
                    found = True
                    break
            if not found:
                print(f"警告: 未找到实验 {exp_name}")
        if not selected:
            print("错误: 未找到任何指定的实验")
            return
        mode = "single" if len(selected) == 1 else "fusion"
    else:
        top_n = args.top_n
        selected = all_experiments[:top_n]
        mode = args.mode

    # 保存
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model_selection.txt")

    save_model_selection(selected, mode, output_path)

    print(f"\n已保存 {len(selected)} 个模型到: {output_path}")
    print(f"模式: {mode}")
    for i, exp in enumerate(selected):
        rel = os.path.relpath(exp["exp_dir"], PROJECT_ROOT)
        print(f"  {i+1}. {rel} ({exp['model_file']}) score={exp['score']:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=10, help="选择最佳模型数量")
    parser.add_argument("--mode", type=str, default="fusion", choices=["single", "fusion"])
    parser.add_argument("--manual", type=str, nargs="*", help="手动指定实验目录名, 如 exp_12 exp_3")
    args = parser.parse_args()

    main(args)
