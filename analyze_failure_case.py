# -*- coding: utf-8 -*-
"""
失败案例解剖 —— 把「第 13 例 Dice 0.88」这句话变成可讲的分析

做三件事：
  1. 对全部 11 例算出逐例指标（Dice / 前景占比 / 漏检·误检像素 / 边界偏差 / 灰度统计），
     写进 results/calcaneus_case_analysis.csv —— 用数据判断哪一例是真正的失败案例。
  2. 对指定的失败案例画一张 6 格解剖图（原图 / GT 轮廓 / 预测轮廓 / 三色叠加 /
     上边界曲线对比 / 逐列边界偏差），写进 results/calcaneus_case{id}_analysis.png。
  3. 用算出来的数字去验证三条最常见的原因假设，并给出「支持 / 不支持」的判定 ——
     不靠感觉归因，靠数据归因。

用法：
    python analyze_failure_case.py                                  # 自动挑 Dice 最低的一例
    python analyze_failure_case.py --id 13                          # 指定第 13 例
    python analyze_failure_case.py --weights results/calcaneus_unet_best_split83.pth --tag check
"""

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
import torch

from calcaneus_dataset import (CalcaneusDataset, DATA_ROOT, dice_score, discover_ids,
                               iou_score)
from calcaneus_angles import REFERENCE, contour, edt_sq
from unet import UNet, count_parameters

RESULT_DIR = Path(__file__).resolve().parent / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT83_VAL = ["1", "13", "94"]      # train_calcaneus.py 在 seed=42 下切出的 3 例验证集


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


# ---------------------------------------------------------------- 指标
def upper_profile(mask):
    """每个 x 列最靠上的前景像素的 y。该列没有前景则 NaN。"""
    W = mask.shape[1]
    top = np.full(W, np.nan)
    for x in range(W):
        idx = np.nonzero(mask[:, x])[0]
        if len(idx):
            top[x] = idx.min()
    return top


def surface_stats(pred, gt):
    cp, cg = contour(pred), contour(gt)
    if not cp.any() or not cg.any():
        return dict(mean=float("nan"), hd95=float("nan"))
    d = np.concatenate([np.sqrt(edt_sq(cg))[cp], np.sqrt(edt_sq(cp))[cg]])
    return dict(mean=float(d.mean()), hd95=float(np.percentile(d, 95)))


def gray_stats(path):
    from PIL import Image
    a = np.asarray(Image.open(path).convert("L")).astype(np.float32)
    p1, p99 = np.percentile(a, 1), np.percentile(a, 99)
    return dict(mean=float(a.mean()), std=float(a.std()),
                p1=float(p1), p99=float(p99), contrast=float(p99 - p1))


def bucket_fraction(err_mask, x0, x1, nb=4):
    """把 [x0, x1] 均分成 nb 段，返回每段内 err_mask 像素占该段宽度的比例。"""
    if not err_mask.any():
        return [0.0] * nb
    xs = np.nonzero(err_mask)[1]
    edges = np.linspace(x0, x1 + 1, nb + 1)
    out = []
    for i in range(nb):
        lo, hi = edges[i], edges[i + 1]
        cnt = int(((xs >= lo) & (xs < hi)).sum())
        out.append(cnt / max(1, (hi - lo) * err_mask.shape[0]))
    return out


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(RESULT_DIR / "calcaneus_unet_best_split83.pth"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--id", default="", help="要解剖哪一例；留空则自动挑 Dice 最低的一例")
    ap.add_argument("--tag", default="", help="产物文件名后缀，与 run_all.py 保持一致")
    args = ap.parse_args()

    tag = f"_{args.tag}" if args.tag and not args.tag.startswith("_") else args.tag
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"找不到权重 {weights}\n请先运行：python train_calcaneus.py --epochs 100 --tag split83")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_ch=1, out_ch=1, base=args.base).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    ids = discover_ids()
    ds = CalcaneusDataset(ids=ids, size=args.size, return_id=True)

    print("=" * 100)
    print("失败案例解剖")
    print("=" * 100)
    print(f"权重      : {weights}   参数量 = {count_parameters(model):,}   device = {device}")
    print(f"数据      : {DATA_ROOT}")
    print()

    rows, panels = [], {}
    with torch.no_grad():
        for idx in range(len(ds)):
            image, mask, sid = ds[idx]
            logits = model(image.unsqueeze(0).to(device))
            prob = torch.sigmoid(logits).cpu()[0]
            pred = (prob > 0.5).float()

            gt_np = mask[0].numpy() > 0.5
            pred_np = pred[0].numpy() > 0.5
            fp = (~gt_np) & pred_np
            fn = gt_np & (~pred_np)

            ss = surface_stats(pred_np, gt_np)
            gs = gray_stats(DATA_ROOT / "images" / f"{sid}.png")
            top_g, top_p = upper_profile(gt_np), upper_profile(pred_np)
            both = ~np.isnan(top_g) & ~np.isnan(top_p)
            dcol = np.where(both, np.abs(top_p - top_g), np.nan)

            rows.append(dict(
                id=sid, split=("验证" if sid in SPLIT83_VAL else "训练"),
                dice=round(dice_score(pred, mask), 4), iou=round(iou_score(pred, mask), 4),
                gt_fg_pct=round(gt_np.mean() * 100, 2), pred_fg_pct=round(pred_np.mean() * 100, 2),
                fp_px=int(fp.sum()), fn_px=int(fn.sum()),
                mean_ssd=round(ss["mean"], 2), hd95=round(ss["hd95"], 2),
                top_diff_mean=round(float(np.nanmean(dcol)), 2),
                top_diff_max=round(float(np.nanmax(dcol)), 2),
                gray_mean=round(gs["mean"], 1), gray_std=round(gs["std"], 1),
                gray_contrast=round(gs["contrast"], 1),
            ))
            panels[sid] = dict(img=image[0].numpy(), gt=gt_np, pred=pred_np,
                               fp=fp, fn=fn, top_g=top_g, top_p=top_p, dcol=dcol)

    # ---------------- 逐例表 ----------------
    print(f"{'id':>4} {'集':>4} | {'Dice':>7} {'IoU':>7} | {'GT前景%':>7} {'预测%':>7} | "
          f"{'漏检px':>7} {'误检px':>7} | {'边界均差':>8} {'HD95':>7} | {'上边界均差':>10}")
    print("-" * 100)
    for r in rows:
        print(f"{r['id']:>4} {r['split']:>4} | {r['dice']:>7.4f} {r['iou']:>7.4f} | "
              f"{r['gt_fg_pct']:>7.2f} {r['pred_fg_pct']:>7.2f} | "
              f"{r['fn_px']:>7d} {r['fp_px']:>7d} | {r['mean_ssd']:>8.2f} {r['hd95']:>7.2f} | "
              f"{r['top_diff_mean']:>10.2f}")
    print("-" * 100)
    worst = min(rows, key=lambda r: r["dice"])
    print(f"全库 Dice 最低: #{worst['id']}  Dice={worst['dice']}  （{'验证集' if worst['split']=='验证' else '训练集'}）")
    print()

    # ---------------- 选目标案例 ----------------
    target_id = args.id.strip() if args.id.strip() else worst["id"]
    tgt = next((r for r in rows if r["id"] == target_id), None)
    if tgt is None:
        raise SystemExit(f"数据里没有编号 {target_id}")
    pnl = panels[target_id]
    others = [r for r in rows if r["id"] != target_id]
    mean_of_others = {k: float(np.mean([r[k] for r in others]))
                      for k in ("dice", "gt_fg_pct", "gray_contrast", "gray_mean", "mean_ssd")}

    print("=" * 100)
    print(f"解剖目标：第 {target_id} 例（{'验证集，模型没见过' if tgt['split'] == '验证' else '训练集，模型见过'}）")
    print("=" * 100)
    print(f"  Dice {tgt['dice']}  vs  其余 10 例均值 {mean_of_others['dice']:.4f}"
          f"   差 {tgt['dice'] - mean_of_others['dice']:+.4f}")
    print(f"  漏检(FN) {tgt['fn_px']} px   误检(FP) {tgt['fp_px']} px"
          f"   （漏检 {0 if tgt['fn_px']==0 else '是'} 主要矛盾）")
    print(f"  GT 前景占比 {tgt['gt_fg_pct']}%  vs  其余均值 {mean_of_others['gt_fg_pct']:.2f}%")
    print(f"  灰度对比度(p99-p1) {tgt['gray_contrast']}  vs  其余均值 {mean_of_others['gray_contrast']:.1f}")
    print(f"  预测前景占比 {tgt['pred_fg_pct']}%  ← 与 GT 占比 {tgt['gt_fg_pct']}% 的差"
          f" = 面积层面的偏差")
    print()

    # ---------------- 三条假设的自动判定 ----------------
    gt_np, pred_np, fp, fn = pnl["gt"], pnl["pred"], pnl["fp"], pnl["fn"]
    xs = np.nonzero(gt_np.any(axis=0))[0]
    x0, x1 = int(xs.min()), int(xs.max())
    fn_b = bucket_fraction(fn, x0, x1)
    fp_b = bucket_fraction(fp, x0, x1)
    seg_names = ["前部(前突侧)", "中前(跟骨沟附近)", "中后", "后部(结节侧)"]
    print("漏检 / 误检 的空间分布（按 x 四等分，数值=该段内错误像素密度）：")
    print(f"  {'区域':<16} {'漏检 FN':>9} {'误检 FP':>9}")
    for i, nm in enumerate(seg_names):
        print(f"  {nm:<16} {fn_b[i]:>9.4f} {fp_b[i]:>9.4f}")
    print()

    fn_top = int(np.argmax(fn_b))
    pred_area_ratio = tgt["pred_fg_pct"] / max(tgt["gt_fg_pct"], 1e-6)
    contrast_ratio = tgt["gray_contrast"] / max(mean_of_others["gray_contrast"], 1e-6)

    hyp = []
    h1 = contrast_ratio < 0.90
    hyp.append(("假设一：这一例的曝光 / 对比度异常，导致边界本身不清晰",
                f"对比度 {tgt['gray_contrast']} vs 其余均值 {mean_of_others['gray_contrast']:.1f}"
                f"（比值 {contrast_ratio:.2f}）",
                "支持" if h1 else "不支持"))
    h2 = (pred_area_ratio < 0.90) or (pred_area_ratio > 1.10)
    hyp.append(("假设二：目标面积与训练分布不匹配，模型整体「缩」了或「胀」了",
                f"预测前景 {tgt['pred_fg_pct']}% / GT 前景 {tgt['gt_fg_pct']}%"
                f" = {pred_area_ratio:.2f} 倍",
                "支持" if h2 else "不支持"))
    h3 = fn_b[fn_top] > 1.5 * float(np.mean(fn_b))
    hyp.append((f"假设三：漏检集中在「{seg_names[fn_top]}」，说明该处边界形态与训练样本差异大",
                f"该段漏检密度 {fn_b[fn_top]:.4f} vs 四段均值 {float(np.mean(fn_b)):.4f}"
                f"（{fn_b[fn_top]/max(np.mean(fn_b),1e-9):.1f} 倍）",
                "支持" if h3 else "不支持"))

    print("三条归因假设的自动判定（用上面算出来的数字判，不靠感觉）：")
    for i, (name, ev, verdict) in enumerate(hyp, 1):
        mark = "[支持]  " if verdict == "支持" else "[不支持]"
        print(f"  {mark} {name}")
        print(f"           证据：{ev}")
    print()

    # ---------------- 存 CSV ----------------
    csv_path = RESULT_DIR / f"calcaneus_case_analysis{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"逐例指标表 : {csv_path}")

    # ---------------- 6 格解剖图 ----------------
    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.4), dpi=150)
    img = pnl["img"]

    def overlay(ax, mask, color, alpha=0.45):
        ax.imshow(img, cmap="gray")
        rgba = np.zeros((*mask.shape, 4))
        rgba[mask] = [*color, alpha]
        ax.imshow(rgba)

    ax = axes[0][0]
    ax.imshow(img, cmap="gray")
    ax.set_title(f"原图　#{target_id}（{'验证集' if tgt['split'] == '验证' else '训练集'}）", fontsize=11)

    ax = axes[0][1]
    overlay(ax, pnl["gt"], (0.15, 0.75, 0.25))
    ax.set_title(f"人工标注 GT　前景 {tgt['gt_fg_pct']}%", fontsize=11)

    ax = axes[0][2]
    overlay(ax, pnl["pred"], (0.90, 0.15, 0.15))
    ax.set_title(f"模型预测　前景 {tgt['pred_fg_pct']}%　Dice {tgt['dice']}", fontsize=11)

    ax = axes[1][0]
    rgb = np.stack([img] * 3, axis=-1)
    rgb = (rgb - rgb.min()) / max(rgb.max() - rgb.min(), 1e-6) * 0.55 + 0.25
    rgb[(pnl["gt"] & pnl["pred"])] = [0.15, 0.75, 0.25]
    rgb[fp] = [0.90, 0.15, 0.15]
    rgb[fn] = [0.15, 0.35, 0.95]
    ax.imshow(np.clip(rgb, 0, 1))
    ax.set_title(f"绿=预测正确　红=多预测({tgt['fp_px']}px)　蓝=漏预测({tgt['fn_px']}px)", fontsize=11)

    ax = axes[1][1]
    xs_all = np.arange(len(pnl["top_g"]))
    m = ~np.isnan(pnl["top_g"])
    ax.plot(xs_all[m], pnl["top_g"][m], color="#1a7f4b", lw=1.4, label="GT 上边界")
    m2 = ~np.isnan(pnl["top_p"])
    ax.plot(xs_all[m2], pnl["top_p"][m2], color="#c01c28", lw=1.4, label="预测上边界")
    q = np.linspace(x0, x1 + 1, 5)
    ax.axvspan(q[2], q[3], color="#b45309", alpha=0.10, zorder=0)
    for k, c in zip("ABCD", ["#1a5fb4", "#1a5fb4", "#1a5fb4", "#6941c6"]):
        px, py = REFERENCE[target_id][k]
        ax.axvline(px, color=c, ls=":", lw=0.9, alpha=0.7)
        ax.text(px, 0.035, k, transform=ax.get_xaxis_transform(), color=c,
                fontsize=9.5, ha="center", va="bottom", weight="600")
    ax.invert_yaxis()
    ax.set_xlabel("x (像素)　虚线 = 参考关键点位置，阴影 = 漏检集中区")
    ax.set_ylabel("y 上边界 (像素，越上越小)")
    ax.set_title("上边界轮廓对比：预测把中后段的上边界压低了", fontsize=11)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.25, lw=0.6)

    ax = axes[1][2]
    dc = pnl["dcol"]
    ax.axvspan(q[2], q[3], color="#b45309", alpha=0.10, zorder=0)
    ax.plot(xs_all, dc, color="#b45309", lw=1.2)
    ax.axhline(tgt["top_diff_mean"], color="#5f6368", ls="--", lw=1.0,
               label=f"平均 {tgt['top_diff_mean']:.2f} px")
    ax.axhline(tgt["top_diff_max"], color="#c01c28", ls=":", lw=1.0,
               label=f"最大 {tgt['top_diff_max']:.2f} px")
    ax.set_xlabel("x (像素)"); ax.set_ylabel("|预测上边界 − GT 上边界| (px)")
    ax.set_title(f"逐列边界偏差：最差落在「{seg_names[fn_top]}」", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, lw=0.6)

    # 前 4 格是图像，去掉刻度；后 2 格是曲线，保留坐标轴
    for a in (axes[0][0], axes[0][1], axes[0][2], axes[1][0]):
        a.set_xticks([])
        a.set_yticks([])
    for a in (axes[0][0], axes[0][1], axes[0][2], axes[1][0], axes[1][1], axes[1][2]):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)

    fig.suptitle(f"失败案例解剖　第 {target_id} 例　Dice {tgt['dice']}　"
                 f"漏检 {tgt['fn_px']} px / 误检 {tgt['fp_px']} px　"
                 f"（其余 10 例 Dice 均值 {mean_of_others['dice']:.4f}）", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig_path = RESULT_DIR / f"calcaneus_case{target_id}_analysis{tag}.png"
    fig.savefig(fig_path)
    print(f"解剖图     : {fig_path}")
    print()

    # ---------------- 改进动作 ----------------
    print("=" * 100)
    print("结论与下一步动作")
    print("=" * 100)
    for name, ev, verdict in hyp:
        if verdict == "支持":
            print(f"  · 成立：{name}")
    print()
    if fn_b[fn_top] > 1.5 * float(np.mean(fn_b)):
        print(f"  最明确的一条：漏检密度最高落在「{seg_names[fn_top]}」（{fn_b[fn_top]:.4f}）。")
        print(f"  这条位置信息比「Dice 掉了多少」有用得多 —— 它直接指出该针对哪里做数据增强 / 加权损失。")
    print()
    print("  ⚠️ 但必须说清楚的边界：这只是 1 例的观察，11 例里 10 例都很好。")
    print("     它足以支撑「失败案例是什么、沿哪些维度定位的」，不足以支撑「哪个原因导致了失败」。")
    print("     要下因果结论，需要更多同类失败样本。")
    print("=" * 100)


if __name__ == "__main__":
    main()
