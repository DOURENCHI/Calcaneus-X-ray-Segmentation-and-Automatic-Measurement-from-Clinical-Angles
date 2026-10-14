# -*- coding: utf-8 -*-
"""
端到端单例演示 —— 一张 X 光片进去，两个角度出来

这条命令可以端到端跑通一张片子：
    python demo_single_case.py              # 自动挑分割最好的一例
    python demo_single_case.py --id 91      # 指定某一例

它串起整条链路：
    X 光片 → U-Net 分割 → 二值掩膜 → 上边界轮廓 → 4 个解剖关键点
           → 余弦定理 → Böhler 角 / Gissane 角 → 与参数图印出值对照

产出一张 4 格演示图，并在终端打印一份可读的结果摘要。

⚠️ 必须一起说明的边界（终端与图上都会写）：
   · 关键点用的是从已有参数图**逐像素反推**出来的参考位置（Böhler 复现误差 0.34°），
     不是自动提取的 —— 因为 B 点的自动提取规则至今还原不出来（误差 23.9 px），
     A/C 点勉强（6.9 / 10.2 px），只有 D 点可靠（3.9 px）。详见 README 7.4 节。
   · 所以这条"全自动"链路的最后一环仍然需要人工给关键点。
     因此这条链路应表述为「差一环的全自动」，而不是「全自动」。
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch

from calcaneus_angles import PRINTED, REFERENCE, bohler_angle, contour, gissane_angle
from calcaneus_dataset import CalcaneusDataset, DATA_ROOT, dice_score, iou_score
from unet import UNet, count_parameters

RESULT_DIR = Path(__file__).resolve().parent / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 口径说明：本仓库的顶点取法（为对齐已有结果）
ARMS = {
    "Böhler": ("A", "B", "C"),      # 顶点 A，两臂 A→B 与 A→C
    "Gissane": ("D", "A", "C"),     # 顶点 D，两臂 D→A 与 D→C
}


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


def draw_angle(ax, lm, spec, color, label):
    """在图上画出一个角：顶点 + 两条臂（延伸出去）+ 角度文字。"""
    vk, k1, k2 = spec
    v = np.array(lm[vk], float)

    for k in (k1, k2):
        p = np.array(lm[k], float)
        d = p - v
        n = np.linalg.norm(d)
        if n < 1e-6:
            continue
        e = v + d / n * (n + 22)          # 臂向外延伸，便于看清
        ax.plot([v[0], e[0]], [v[1], e[1]], color=color, lw=1.8, alpha=0.9,
                solid_capstyle="round")
        ax.annotate("", xy=e, xytext=v,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, alpha=0.9,
                                    shrinkA=0, shrinkB=0))

    # 顶点
    ax.plot(*v, marker="o", ms=7, mfc=color, mec="white", mew=1.2, zorder=6)

    # 两条臂之间的圆弧
    a1 = math.atan2(lm[k1][1] - v[1], lm[k1][0] - v[0])
    a2 = math.atan2(lm[k2][1] - v[1], lm[k2][0] - v[0])
    deg1, deg2 = math.degrees(a1), math.degrees(a2)
    if deg2 < deg1:
        deg1, deg2 = deg2, deg1
    if deg2 - deg1 > 180:
        deg1, deg2 = deg2, deg1 + 360
    tt = np.radians(np.linspace(deg1, deg2, 60))
    r = 26
    ax.plot(v[0] + r * np.cos(tt), v[1] + r * np.sin(tt), color=color, lw=1.4, alpha=0.85)

    mid = np.radians((deg1 + deg2) / 2.0)
    ax.text(v[0] + (r + 16) * math.cos(mid), v[1] + (r + 16) * math.sin(mid),
            label, color=color, fontsize=10.5, ha="center", va="center", weight="600")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="", help="要演示哪一例；留空则自动挑 Dice 最高的一例")
    ap.add_argument("--weights", default=str(RESULT_DIR / "calcaneus_unet_best_split83.pth"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--base", type=int, default=16)
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"找不到权重 {weights}\n请先运行：python run_all.py（或 python train_calcaneus.py --epochs 100 --tag split83）")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_ch=1, out_ch=1, base=args.base).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    ids = list(REFERENCE.keys())
    ds = CalcaneusDataset(ids=ids, size=args.size, return_id=True)

    print("=" * 92)
    print("端到端演示：X 光片 → 分割掩膜 → 关键点 → Böhler / Gissane 角")
    print("=" * 92)
    print(f"权重 {weights.name}   参数量 {count_parameters(model):,}   device = {device}")
    print()

    # ---- 1. 先跑一遍全部，选目标 ----
    recs = {}
    with torch.no_grad():
        for k in range(len(ds)):
            image, mask, sid = ds[k]
            prob = torch.sigmoid(model(image.unsqueeze(0).to(device))).cpu()[0]
            pred = (prob > 0.5).float()
            recs[sid] = dict(image=image, mask=mask, pred=pred,
                             dice=dice_score(pred, mask), iou=iou_score(pred, mask))

    target = args.id.strip() if args.id.strip() else max(recs, key=lambda s: recs[s]["dice"])
    if target not in recs:
        raise SystemExit(f"数据里没有编号 {target}")

    r = recs[target]
    image, mask, pred = r["image"], r["mask"], r["pred"]
    gt_np = mask[0].numpy() > 0.5
    pred_np = pred[0].numpy() > 0.5
    lm = REFERENCE[target]

    print(f"演示样本      : #{target}")
    print(f"分割           : Dice {r['dice']:.4f}   IoU {r['iou']:.4f}")
    print(f"前景占比       : GT {gt_np.mean() * 100:.2f}%   预测 {pred_np.mean() * 100:.2f}%")
    print()
    print("关键点（从已有参数图逐像素反推得到的参考位置）：")
    for k in "ABCD":
        print(f"    {k} = (x={lm[k][0]:>6.1f}, y={lm[k][1]:>6.1f})")
    print()

    b = bohler_angle(lm)
    g = gissane_angle(lm)
    pb, pg = PRINTED[target]
    print("角度计算（余弦定理）：")
    print(f"    Böhler  顶点 A，两臂 A→B 与 A→C      计算值 {b:>7.2f}°   参数图印出 {pb:>7.2f}°   差 {b - pb:>+6.2f}°")
    print(f"    Gissane 顶点 D，两臂 D→A 与 D→C      计算值 {g:>7.2f}°   参数图印出 {pg:>7.2f}°   差 {g - pg:>+6.2f}°")
    print()
    print("临床参考区间：Böhler 正常 25°~40°（骨折塌陷时变小）；Gissane 正常 120°~145°（塌陷时变大）")
    dev_b = "低于正常下限（与关节内骨折塌陷一致）" if b < 25 else "在正常区间内"
    dev_g = "高于正常上限（与关节内骨折塌陷一致）" if g > 145 else "在正常区间内"
    print(f"    本例 Böhler {b:.2f}° → {dev_b}")
    print(f"    本例 Gissane {g:.2f}° → {dev_g}")
    print()

    # ---- 2. 画 4 格演示图 ----
    plt = setup_matplotlib()
    img = image[0].numpy()
    cp = contour(pred_np)

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 10.2), dpi=150)

    ax = axes[0][0]
    ax.imshow(img, cmap="gray")
    ax.set_title("① 输入：跟骨侧位 X 光片", fontsize=11.5)

    ax = axes[0][1]
    ax.imshow(img, cmap="gray")
    rgba = np.zeros((*pred_np.shape, 4))
    rgba[cp] = [0.90, 0.15, 0.15, 0.95]        # 只画轮廓，更清楚
    ax.imshow(rgba)
    ax.contour(gt_np.astype(float), levels=[0.5], colors=["#1a7f4b"],
               linewidths=1.1, linestyles="--")
    ax.set_title(f"② U-Net 分割（红=预测轮廓，绿虚=人工 GT）\nDice {r['dice']:.3f}　IoU {r['iou']:.3f}",
                 fontsize=11)

    ax = axes[1][0]
    ax.imshow(img, cmap="gray")
    # 只画一圈亮色轮廓做上下文，避免和测量线抢视觉
    ax.imshow(np.where(cp, 1.0, np.nan), cmap="Greys", alpha=0.0)
    cy, cx = np.nonzero(cp)
    ax.scatter(cx, cy, s=1.0, c="#39d353", alpha=0.55, linewidths=0, zorder=3)
    draw_angle(ax, lm, ARMS["Böhler"], "#1a5fb4", "")
    draw_angle(ax, lm, ARMS["Gissane"], "#c026d3", "")
    offs = {"A": (-16, 12), "B": (-6, -16), "C": (10, -8), "D": (12, 14)}
    for k in "ABCD":
        ax.plot(*lm[k], marker="o", ms=4.4, color="#ffd400", mec="#222", mew=0.8, zorder=7)
        dx, dy = offs[k]
        ax.text(lm[k][0] + dx, lm[k][1] + dy, k, color="#ffd400", fontsize=12,
                weight="700", ha="center", va="center", zorder=8,
                bbox=dict(boxstyle="circle,pad=0.12", facecolor="#222222", edgecolor="none",
                          alpha=0.75))
    ax.text(0.02, 0.02,
            f"■ Böhler  {b:.2f}°  （顶点 A）\n■ Gissane {g:.2f}°  （顶点 D）",
            transform=ax.transAxes, fontsize=10.2, va="bottom", ha="left", linespacing=1.6,
            color="#ffffff",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#1c1c1e", edgecolor="none", alpha=0.82))
    ax.set_title("③ 关键点与角度测量（顶点口径见右侧说明）", fontsize=11)

    ax = axes[1][1]
    ax.axis("off")
    txt = (
        f"样本 #{target}\n\n"
        f"分割\n"
        f"   Dice          {r['dice']:.4f}\n"
        f"   IoU           {r['iou']:.4f}\n"
        f"   前景占比       GT {gt_np.mean() * 100:.2f}% / 预测 {pred_np.mean() * 100:.2f}%\n\n"
        f"角度（余弦定理）\n"
        f"   Böhler        {b:.2f}°\n"
        f"   Gissane       {g:.2f}°\n\n"
        f"与参数图印出值对照\n"
        f"   Böhler         {b:.2f}° vs {pb:.2f}°   （差 {b - pb:+.2f}°）\n"
        f"   Gissane        {g:.2f}° vs {pg:.2f}°   （差 {g - pg:+.2f}°）\n\n"
        f"顶点口径（本仓库，为对齐已有结果）\n"
        f"   Böhler        顶点 A，两臂 A→B、A→C\n"
        f"   Gissane       顶点 D，两臂 D→A、D→C\n\n"
        f"注意：关键点为逐像素反推的参考位置，\n"
        f"   不是自动提取的；B 点自动提取至今失败\n"
        f"   （误差 23.9 px，见 README 7.4）"
    )
    ax.text(0.015, 0.985, txt, transform=ax.transAxes, fontsize=10.8, va="top", ha="left",
            linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.7", facecolor="#f7f8fa", edgecolor="#e3e5e8"))

    for a in (axes[0][0], axes[0][1], axes[1][0]):
        a.set_xticks([]); a.set_yticks([])
        for s in ("top", "right"):
            a.spines[s].set_visible(False)

    fig.suptitle(f"跟骨 X 光分割与角度测量 · 端到端演示（样本 #{target}）",
                 fontsize=13.5, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.972])
    out = RESULT_DIR / f"calcaneus_demo_case{target}.png"
    fig.savefig(out)
    print(f"演示图已存 : {out}")
    print()
    print("=" * 92)
    print("一句话总结：一张侧位片进模型，出掩膜、出轮廓、出两个临床角度；")
    print("           角度算法本身已复现到位（Böhler 11 例平均误差 0.34°），")
    print("           卡住的是最后一环——B 点的自动定位规则不在轮廓几何里。")
    print("=" * 92)


if __name__ == "__main__":
    main()
