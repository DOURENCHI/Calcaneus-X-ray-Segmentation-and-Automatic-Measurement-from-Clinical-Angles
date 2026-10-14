# -*- coding: utf-8 -*-
"""
一条命令跑完整条链路 —— 可复现入口

这个脚本本身不做任何计算，只负责**按正确顺序**调用其余 7 个脚本，
并保证每次运行的环境一致（随机种子、hash seed、工作目录）。

用法：
    python run_all.py                       # 跑全部 8 步
    python run_all.py --tag check           # 产物名加后缀，不覆盖仓库里的原始记录
    python run_all.py --skip-preprocess     # 跳过 74 张原始片的脱敏（约 1 分钟）
    python run_all.py --python D:\\Miniconda\\envs\\medimg\\python.exe

产物：全部写在 results/ 下（与原逐脚本运行完全一致）。

关于「重跑的数字和仓库里记录的不一样」——这不是 bug，先读 README 第 7.5 节。
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 42


def build_steps(args):
    """返回 (显示名, 脚本相对路径, 参数列表, 主要产物) 的列表。"""
    tag = f"_{args.tag}" if args.tag else ""
    p = args.python

    steps = [
        ("1 数据自检",      "calcaneus_dataset.py",   [],
         "打印 11 例配对结果与张量形状"),
        ("2 网络结构",      "unet.py",                [],
         "打印 base=8/16/32 的参数量"),
        ("3 过拟合测试",    "train_calcaneus.py",
         ["--overfit", "--epochs", str(args.epochs_overfit), "--tag", f"overfit{tag}"],
         f"results/calcaneus_train_log_overfit{tag}.csv 等"),
        ("4 8-3 留出训练",  "train_calcaneus.py",
         ["--epochs", str(args.epochs_split), "--tag", f"split83{tag}"],
         f"results/calcaneus_unet_best_split83{tag}.pth 等"),
        ("5 预测 + 对照图", "predict_calcaneus.py",
         ["--weights", f"results/calcaneus_unet_best_split83{tag}.pth", "--tag", f"_split83{tag}"],
         f"results/calcaneus_predict_dice_split83{tag}.csv 等"),
        ("6 角度 + 误差传导", "calcaneus_angles.py",
         ["--mask", "results", "--out", f"calcaneus_angles{tag}"],
         f"results/calcaneus_angles{tag}_results.csv 等"),
        ("7 失败案例解剖",  "analyze_failure_case.py",
         ["--weights", f"results/calcaneus_unet_best_split83{tag}.pth", "--tag", tag],
         f"results/calcaneus_case_analysis{tag}.csv 等"),
    ]
    if not args.skip_preprocess:
        steps.append(("8 原始片预处理", "calcaneus_preprocess.py", [],
                      "74 张脱敏副本 + manifest + 尺寸分布图"))
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable,
                    help="用哪个解释器跑子脚本，默认用当前解释器")
    ap.add_argument("--tag", default="",
                    help="给所有产物名加后缀，避免覆盖仓库里的原始记录（推荐首次跑用 check）")
    ap.add_argument("--epochs-overfit", type=int, default=80)
    ap.add_argument("--epochs-split", type=int, default=100)
    ap.add_argument("--skip-preprocess", action="store_true",
                    help="跳过第 8 步（74 张原始片脱敏，需要原始片目录）")
    ap.add_argument("--keep-going", action="store_true",
                    help="某一步失败后继续跑剩下的（默认失败即停）")
    args = ap.parse_args()

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(SEED)          # 必须在子进程启动前设置才有效
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("MPLBACKEND", "Agg")

    steps = build_steps(args)

    print("=" * 96)
    print("跟骨 X 光分割与角度测量 —— 全链路一键复现")
    print("=" * 96)
    print(f"工作目录   : {HERE}")
    print(f"解释器     : {args.python}")
    print(f"随机种子   : {SEED}（同时作为 PYTHONHASHSEED 传给子进程）")
    print(f"产物后缀   : {args.tag if args.tag else '（无，会覆盖仓库里的原始记录）'}")
    print(f"共 {len(steps)} 步")
    print("=" * 96)

    # ---------------- 前置检查 ----------------
    missing = [s for s in ["calcaneus_dataset.py", "unet.py", "train_calcaneus.py",
                           "predict_calcaneus.py", "calcaneus_angles.py",
                           "analyze_failure_case.py"] if not (HERE / s).exists()]
    if missing:
        print(f"[错误] 缺少脚本: {missing}")
        return 2
    try:
        sys.path.insert(0, str(HERE))
        from calcaneus_dataset import DATA_ROOT
        ok = (DATA_ROOT / "images").exists() and (DATA_ROOT / "labels").exists()
        print(f"数据根目录 : {DATA_ROOT}   {'[OK]' if ok else '[缺失]'}")
        if not ok:
            print()
            print("找不到标注数据。请把数据放到下面任一位置，或用环境变量指定：")
            print(r"  set CALCANEUS_ROOT=D:\你的路径\result\result")
            print("目录里需要同时有 images/ 与 labels/，按编号同名配对（如 13.png）。")
            return 2
    except Exception as e:
        print(f"[警告] 数据自检失败: {e}")
    print()

    # ---------------- 逐脚本执行 ----------------
    results, t_all = [], time.time()
    for name, script, argv, expect in steps:
        print("-" * 96)
        print(f">>> {name}   ({script})")
        print("-" * 96)
        t0 = time.time()
        proc = subprocess.run([args.python, str(HERE / script)] + argv,
                              cwd=str(HERE), env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace")
        dt = time.time() - t0
        # 把子进程输出原样透传（最后 6 行足够看清结论，避免刷屏）
        lines = [l for l in (proc.stdout or "").splitlines() if l.strip()]
        for l in lines[-6:]:
            print("    " + l)
        if len(lines) > 6:
            print(f"    ...（完整输出共 {len(lines)} 行）")
        ok = proc.returncode == 0
        results.append((name, script, ok, dt, expect if ok else "见上方输出"))
        print(f"    {'[OK]' if ok else '[失败]'} 用时 {dt:.1f} 秒")
        if not ok and not args.keep_going:
            print()
            print("[中断] 某一步失败，已停止。修好之后重跑 run_all.py 即可。")
            break

    # ---------------- 汇总 ----------------
    print()
    print("=" * 96)
    print(f"汇总（总用时 {time.time() - t_all:.1f} 秒）")
    print("=" * 96)
    print(f"{'步骤':<20} {'结果':<6} {'用时':>8}   主要产物")
    print("-" * 96)
    for name, script, ok, dt, expect in results:
        print(f"{name:<20} {'OK' if ok else '失败':<6} {dt:>7.1f}s   {expect}")
    n_ok = sum(1 for r in results if r[2])
    print("-" * 96)
    print(f"{n_ok}/{len(results)} 步成功")
    if n_ok == len(results):
        print()
        print("全部跑通。下面是看结果的地方：")
        print("  · 训练曲线      results/calcaneus_train_curve_split83.png")
        print("  · 预测对照图    results/calcaneus_predict_grid_split83.png")
        print("  · 逐例 Dice     results/calcaneus_predict_dice_split83.csv")
        print("  · 角度/误差传导 results/calcaneus_angles_results.csv")
        print("  · 失败案例解剖  results/calcaneus_case13_analysis.png")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
