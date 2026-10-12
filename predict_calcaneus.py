# -*- coding: utf-8 -*-
"""
跟骨 U-Net 分割基线 —— 预测与可视化

加载 train_calcaneus.py 存下来的权重，对全部 11 例做预测，
算出每例的 Dice / IoU，并输出一张"原图 / 人工标注 / 模型预测 / 叠加对比"的对照图。

用法：
    python predict_calcaneus.py
    python predict_calcaneus.py --weights results/calcaneus_unet_best.pth --size 256

产物（写在 results/ 下）：
    calcaneus_predict_grid.png   11 例预测对照图
    calcaneus_predict_dice.csv   逐例 Dice / IoU
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch

from calcaneus_dataset import CalcaneusDataset, dice_score, iou_score, discover_ids
from unet import UNet, count_parameters

RESULT_DIR = Path(__file__).resolve().parent / "results"


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        if os.path.exists(f):
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=f).get_name()]
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default=str(RESULT_DIR / "calcaneus_unet_best.pth"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--tag", type=str, default="", help="产物文件名后缀，用来区分不同权重的结果")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"找不到权重 {weights}\n请先运行：python train_calcaneus.py --overfit --epochs 80")

    model = UNet(in_ch=1, out_ch=1, base=args.base).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    print("已加载权重:", weights)
    print(f"参数量 = {count_parameters(model):,}   device = {device}")
    print()

    ids = discover_ids()
    ds = CalcaneusDataset(ids=ids, size=args.size, return_id=True)

    rows, panels = [], []
    with torch.no_grad():
        for idx in range(len(ds)):
            image, mask, sample_id = ds[idx]
            logits = model(image.unsqueeze(0).to(device))
            prob = torch.sigmoid(logits).cpu()[0]
            pred = (prob > 0.5).float()

            d = dice_score(pred, mask)
            j = iou_score(pred, mask)
            rows.append({"id": sample_id, "dice": round(d, 4), "iou": round(j, 4),
                         "gt_fg_%": round(mask.mean().item() * 100, 2),
                         "pred_fg_%": round(pred.mean().item() * 100, 2)})
            panels.append((sample_id, image[0].numpy(), mask[0].numpy(), pred[0].numpy(), d))

    # ---------------- 打印表格 ----------------
    print(f"{'id':>4} | {'Dice':>7} {'IoU':>7} | {'GT前景%':>8} {'预测前景%':>10}")
    print("-" * 46)
    for r in rows:
        print(f"{r['id']:>4} | {r['dice']:>7.4f} {r['iou']:>7.4f} | "
              f"{r['gt_fg_%']:>8.2f} {r['pred_fg_%']:>10.2f}")
    mean_d = sum(r["dice"] for r in rows) / len(rows)
    mean_j = sum(r["iou"] for r in rows) / len(rows)
    print("-" * 46)
    print(f"均值 | {mean_d:>7.4f} {mean_j:>7.4f}")
    print()

    csv_path = RESULT_DIR / f"calcaneus_predict_dice{args.tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "dice", "iou", "gt_fg_%", "pred_fg_%"])
        writer.writeheader()
        writer.writerows(rows)

    # ---------------- 对照图 ----------------
    plt = setup_matplotlib()
    n = len(panels)
    # 图幅刻意做成"宽而矮"：保证整张图 + 图注能落在同一页 A4 上（11 行太高会跨页）
    fig, axes = plt.subplots(n, 4, figsize=(11.5, 1.18 * n + 0.5), dpi=120)
    col_titles = ["原图 (images)", "人工标注 (labels)", "模型预测 (pred)", "叠加：绿=TP 红=FP 蓝=FN"]

    for i, (sid, img, gt, pred, d) in enumerate(panels):
        ax = axes[i]
        ax[0].imshow(img, cmap="gray")
        ax[0].text(0.05, 0.5, f"#{sid}", transform=ax[0].transAxes, fontsize=9,
                   color="#1a5fb4", ha="left", va="center",
                   bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
        ax[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        ax[2].imshow(pred, cmap="gray", vmin=0, vmax=1)

        rgb = np.stack([img] * 3, axis=-1)
        rgb = (rgb - rgb.min()) / max(rgb.max() - rgb.min(), 1e-6) * 0.55 + 0.25
        tp = (gt > 0.5) & (pred > 0.5)
        fp = (gt <= 0.5) & (pred > 0.5)
        fn = (gt > 0.5) & (pred <= 0.5)
        rgb[tp] = [0.15, 0.75, 0.25]     # 绿 = 预测对
        rgb[fp] = [0.90, 0.15, 0.15]     # 红 = 多预测
        rgb[fn] = [0.15, 0.35, 0.95]     # 蓝 = 漏预测
        ax[3].imshow(np.clip(rgb, 0, 1))

        ax[2].set_title(f"Dice = {d:.3f}", fontsize=9)
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        if i == 0:
            for a, t in zip(ax, col_titles):
                if a is not ax[2]:
                    a.set_title(t, fontsize=10)
            ax[2].set_title(f"{col_titles[2]}\nDice = {d:.3f}", fontsize=10)

    fig.suptitle(f"跟骨 X 光分割预测对照（共 {n} 例，Dice 均值 {mean_d:.4f}，IoU 均值 {mean_j:.4f}）",
                 fontsize=12, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    grid_path = RESULT_DIR / f"calcaneus_predict_grid{args.tag}.png"
    fig.savefig(grid_path)

    print("产物:")
    print(f"  {grid_path}")
    print(f"  {csv_path}")
    print()
    print("看图要点：叠加图里绿色是预测对的，红色是模型多预测的，蓝色是漏掉的。")
    print("如果红色/蓝色集中在跟骨边缘或关节面附近，那才是真正会影响角度测量的地方。")


if __name__ == "__main__":
    main()
