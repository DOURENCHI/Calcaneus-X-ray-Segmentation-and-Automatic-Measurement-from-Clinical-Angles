# -*- coding: utf-8 -*-
"""
跟骨原始片预处理（74 张）

⚠️ 隐私：这 74 张的文件名就是患者姓名。本脚本做两件事：
   1. 生成一份**脱敏副本**（重命名为 case_001 ... 的顺序编号）
   2. 把「原名 → 编号」的映射单独写到一个明显标注为 PRIVATE 的文件里
      —— 这个映射文件绝不要上传到任何公开平台，也不要写进论文/PPT。
   原始文件不做任何删除或修改。

⚠️ 重要事实：这 74 张**没有任何掩膜**。它们只能用于：
   - 检查预处理是否正确、看数据分布
   - 后续用训练好的模型做无监督推理（没有 GT，不能算 Dice）
   不能拿来做监督训练。

用法：
    & "D:\\Miniconda\\envs\\medimg\\python.exe" calcaneus_preprocess.py
    & "D:\\Miniconda\\envs\\medimg\\python.exe" calcaneus_preprocess.py --no-copy
    & "D:\\Miniconda\\envs\\medimg\\python.exe" calcaneus_preprocess.py --out D:\\某个目录

路径解析：
    原片目录按 _RAW_CANDIDATES 依次探测（只认**确实含图片**的目录），
    也可用环境变量 CALCANEUS_RAW 直接指定；
    输出目录默认 = 原片容器的上一级 / <容器名>_预处理，
    也可用 --out 或环境变量 CALCANEUS_OUT 指定。
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------
# 路径：同样不写死，按候选列表探测
# ---------------------------------------------------------------------
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 顺序 = 优先级。目录被改过名（2026-10 起 跟骨1 → 跟骨/跟骨1），所以新旧都留着。
_RAW_CANDIDATES = [
    Path(r"D:\工程\方向与规划\项目\跟骨\跟骨\跟骨1"),      # 2026-10 起的结构
    Path(r"D:\工程\方向与规划\项目\跟骨\跟骨"),            # 若原片被拍平到这一层
    Path(r"D:\工程\方向与规划\项目\跟骨\跟骨1\跟骨1"),      # 旧结构
    Path(r"D:\工程\方向与规划\项目\跟骨\跟骨1"),
    Path(r"D:\论文\方向与规划\项目\跟骨\跟骨\跟骨1"),
    Path(r"D:\论文\方向与规划\项目\跟骨\跟骨1\跟骨1"),
    Path(r"D:\论文\方向与规划\项目\跟骨\跟骨1"),
]


def _has_images(d: Path) -> bool:
    """该目录**自身**（不递归）是否含图片文件。

    只看"非空"是不够的：外层容器目录（如 ...\\跟骨）非空，但里面只有一个子目录，
    被选中后会在统计阶段扫到 0 张图。
    """
    try:
        return d.is_dir() and any(
            p.is_file() and p.suffix.lower() in _IMG_EXTS for p in d.iterdir())
    except OSError:
        return False


def find_raw_dir():
    env = os.environ.get("CALCANEUS_RAW")
    if env and Path(env).exists():
        return Path(env)
    for c in _RAW_CANDIDATES:
        if _has_images(c):
            return c
    tried = "\n  ".join(str(c) for c in _RAW_CANDIDATES)
    raise FileNotFoundError(
        "没有找到原始片目录。请用环境变量 CALCANEUS_RAW 指定，"
        "或把原片放到下列任一位置：\n  " + tried)


def find_output_dir(raw: Path, out=None):
    """脱敏副本的输出目录。

    优先级：--out 参数 > CALCANEUS_OUT 环境变量 > 默认规则。
    默认规则：放在「容纳原片那一层目录」的**上一级**，命名为 <该层目录名>_预处理。
    两种常见结构都落在同一个位置：

        ...\\跟骨\\跟骨1\\跟骨1   ->   ...\\跟骨\\跟骨_预处理
        ...\\跟骨\\跟骨\\跟骨1    ->   ...\\跟骨\\跟骨_预处理
    """
    if out:
        return Path(out)
    env = os.environ.get("CALCANEUS_OUT")
    if env:
        return Path(env)
    container = raw.parent
    if raw.name == container.name:       # 同名重复嵌套：跟骨1/跟骨1
        container = container.parent
    return container.parent / (container.name + "_预处理")


def img_stats(gray: np.ndarray):
    return dict(mean=float(gray.mean()), std=float(gray.std()),
                p1=float(np.percentile(gray, 1)), p50=float(np.percentile(gray, 50)),
                p99=float(np.percentile(gray, 99)), gmax=int(gray.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-copy", action="store_true", help="只统计，不生成脱敏副本")
    ap.add_argument("--thumb", type=int, default=160, help="拼图缩略图边长")
    ap.add_argument("--out", default=None,
                    help="脱敏副本的输出目录（默认 ..\\<容器目录名>_预处理）")
    args = ap.parse_args()

    raw = find_raw_dir()
    out = find_output_dir(raw, args.out)
    out.mkdir(parents=True, exist_ok=True)

    exts = _IMG_EXTS
    files = sorted([p for p in raw.iterdir() if p.is_file() and p.suffix.lower() in exts],
                   key=lambda p: p.name)

    print("=" * 78)
    print(f"原始片目录 : {raw}")
    print(f"输出目录   : {out}")
    print(f"扫描到     : {len(files)} 个图像文件")
    print("=" * 78)

    from collections import Counter
    ext_count = Counter(p.suffix.lower() for p in files)
    print("扩展名分布 : " + ", ".join(f"{k}×{v}" for k, v in ext_count.most_common()))
    stems = Counter(p.stem for p in files)
    dup = {k: v for k, v in stems.items() if v > 1}
    # 注意：这里**不能**打印重名的主名 —— 主名就是患者姓名，一旦打到终端，
    # 截图、录屏、日志都会把它带出去，与本脚本的脱敏目的相悖。只报告组数。
    msg = f"不同文件名 : {len(stems)} 个"
    if dup:
        msg += f"，其中 {len(dup)} 组重名（主名不打印；一个病例有两份文件，脱敏后会得到两个编号）"
    print(msg)
    print()

    gray_dir = out / "images_gray"
    if not args.no_copy:
        gray_dir.mkdir(parents=True, exist_ok=True)

    rows, mapping_rows, thumbs = [], [], []
    for idx, p in enumerate(files, 1):
        new_id = f"case_{idx:03d}"
        try:
            im = Image.open(p)
            mode, size = im.mode, im.size
            gray = np.asarray(im.convert("L"))
        except Exception as e:
            print(f"  [跳过] {new_id} 读取失败: {e}")
            continue

        st = img_stats(gray)
        rows.append(dict(id=new_id, ext=p.suffix.lower().lstrip("."),
                         orig_mode=mode, width=size[0], height=size[1],
                         aspect=round(size[0] / size[1], 3),
                         gray_mean=round(st["mean"], 1), gray_std=round(st["std"], 1),
                         gray_p1=round(st["p1"], 1), gray_p50=round(st["p50"], 1),
                         gray_p99=round(st["p99"], 1), gray_max=st["gmax"]))
        mapping_rows.append(dict(id=new_id, original_filename=p.name))

        if not args.no_copy:
            Image.fromarray(gray).save(gray_dir / f"{new_id}.png")
            thumbs.append((new_id, np.asarray(
                im.convert("L").resize((args.thumb, args.thumb), Image.Resampling.BILINEAR))))

    n = len(rows)
    W = np.array([r["width"] for r in rows]); H = np.array([r["height"] for r in rows])
    ar = np.array([r["aspect"] for r in rows])
    print("尺寸分布:")
    print(f"  宽  {W.min()} ~ {W.max()}   中位数 {int(np.median(W))}")
    print(f"  高  {H.min()} ~ {H.max()}   中位数 {int(np.median(H))}")
    print(f"  宽高比 {ar.min():.2f} ~ {ar.max():.2f}   中位数 {np.median(ar):.2f}")
    print(f"  互不相同的尺寸组合: {len(set(zip(W.tolist(), H.tolist())))} 种 / {n} 张")
    print()
    print("灰度统计（用于决定归一化策略）:")
    print(f"  mean  {min(r['gray_mean'] for r in rows):.0f} ~ {max(r['gray_mean'] for r in rows):.0f}")
    print(f"  p99   {min(r['gray_p99'] for r in rows):.0f} ~ {max(r['gray_p99'] for r in rows):.0f}")
    print(f"  最大亮度不是 255 的样本数: {sum(1 for r in rows if r['gray_max'] < 255)} / {n}"
          "   ← 若不为 0，说明存在未铺满动态范围的片子")

    # ---------- 清单（脱敏） ----------
    man_path = out / "manifest_deidentified.csv"
    with open(man_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    # ---------- 映射（PRIVATE） ----------
    map_path = out / "_PRIVATE_原名映射_切勿公开.csv"
    with open(map_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=["id", "original_filename"])
        wr.writeheader(); wr.writerows(mapping_rows)

    print()
    print(f"脱敏清单   : {man_path}")
    print(f"⚠️ 隐私映射 : {map_path}   ← 含患者姓名，绝不要上传/写进论文")

    # ---------- 图 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for fp in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
            if os.path.exists(fp):
                font_manager.fontManager.addfont(fp)
                plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=fp).get_name()]
                break
        plt.rcParams["axes.unicode_minus"] = False

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7), dpi=150)
        axes[0].scatter(W, H, s=22, c="#1a5fb4", alpha=0.8)
        axes[0].axhline(np.median(H), color="#888780", ls="--", lw=0.9)
        axes[0].axvline(np.median(W), color="#888780", ls="--", lw=0.9)
        axes[0].set_xlabel("宽度"); axes[0].set_ylabel("高度")
        axes[0].set_title(f"74 张原始片尺寸分布（{len(set(zip(W.tolist(), H.tolist())))} 种不同尺寸）", fontsize=11)
        axes[0].grid(alpha=0.25, lw=0.6)

        axes[1].hist(W, bins=20, color="#1a5fb4", alpha=0.85, label="宽度")
        axes[1].hist(H, bins=20, color="#c0dd97", alpha=0.85, label="高度")
        axes[1].set_xlabel("像素"); axes[1].set_ylabel("张数")
        axes[1].set_title("宽 / 高分布", fontsize=11)
        axes[1].legend(fontsize=9); axes[1].grid(alpha=0.25, lw=0.6)

        axes[2].hist([r["gray_mean"] for r in rows], bins=20, color="#1a5fb4", alpha=0.85, label="均值")
        axes[2].hist([r["gray_p99"] for r in rows], bins=20, color="#c0dd97", alpha=0.85, label="p99")
        axes[2].set_xlabel("灰度"); axes[2].set_ylabel("张数")
        axes[2].set_title("灰度统计（决定归一化策略）", fontsize=11)
        axes[2].legend(fontsize=9); axes[2].grid(alpha=0.25, lw=0.6)

        for ax in axes:
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "raw_size_distribution.png")
        print(f"尺寸分布图 : {out / 'raw_size_distribution.png'}")

        if thumbs:
            cols = 10
            rr = (len(thumbs) + cols - 1) // cols
            side = args.thumb / 100
            fig2, axes2 = plt.subplots(rr, cols, figsize=(cols * side * 0.75, rr * side * 0.85), dpi=150)
            axes2 = np.atleast_2d(axes2)
            for k, (sid, arr) in enumerate(thumbs):
                ax = axes2[k // cols][k % cols]
                ax.imshow(arr, cmap="gray")
                ax.set_title(sid.replace("case_", ""), fontsize=6, pad=1)
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color("#cccccc"); s.set_linewidth(0.5)
            for k in range(len(thumbs), rr * cols):
                axes2[k // cols][k % cols].axis("off")
            fig2.suptitle(f"74 张原始片总览（已脱敏编号 case_001~case_{len(thumbs):03d}）", fontsize=12)
            fig2.tight_layout(rect=[0, 0, 1, 0.97])
            fig2.savefig(out / "contact_sheet.png")
            print(f"总览拼图   : {out / 'contact_sheet.png'}")
    except Exception as e:
        print("画图失败:", e)

    print()
    print("=" * 78)
    print("结论：")
    print(f"  · 74 张全部为无掩膜的原始片，只能做预处理验证与无监督推理")
    print(f"  · 尺寸极不统一（{len(set(zip(W.tolist(), H.tolist())))} 种不同尺寸）⇒ 送入网络前必须统一缩放")
    print("  · 脱敏副本已生成，原名映射单独存为 PRIVATE 文件，请注意保管")
    print("=" * 78)


if __name__ == "__main__":
    main()
