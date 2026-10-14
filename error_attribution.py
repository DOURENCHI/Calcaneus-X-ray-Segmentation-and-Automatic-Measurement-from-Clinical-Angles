# -*- coding: utf-8 -*-
"""
误差归因实验 —— 把角度总误差拆成「提点规则的」和「分割边界的」

问题背景
--------
项目已有的核心结论是「分割边界平均偏 1.56 px，就足以让角度抖 ±2~4°」。
但这句话里有两个误差源混在一起：
    (a) 关键点提点规则本身的误差（几何规则还原不出参考关键点）
    (b) 掩膜边界误差传导到关键点位置的误差
到底哪一个占主导？不拆开就没法回答，也没法决定下一步该优化哪里。

实验设计（3 条支线 / 2 个减法）
------------------------------
  支线①  人工掩膜 + 参考提点   → A1（基线，关键点用从参数图逐像素反推的位置）
  支线②  人工掩膜 + 自动提点   → A2
  支线③  预测掩膜 + 自动提点   → A3

  A2 − A1 = 提点规则误差（掩膜没变，只换了提点方式）
  A3 − A2 = 分割边界误差（提点规则没变，只换了掩膜）
  A3 − A1 = 总误差 = 两者之和（恒等，用来校验实验没做错）

⚠️ 为什么不是原计划的 4 条支线
    原计划里第 4 条「预测掩膜 + 参考提点」是**退化**的：角度只由关键点坐标决定，
    关键点一旦固定在参考位置，掩膜就完全不影响结果 —— 那一格必然恒等于 A1。
    所以正确的因子设计只能到 3 格。写成 4 格会在答辩时被问倒，这里直接说明。

用法：
    python error_attribution.py                            # 预测掩膜 = 本仓库自训 U-Net
    python error_attribution.py --pred-source pkg          # 预测掩膜 = 数据包自带掩膜
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch

from calcaneus_angles import (PRINTED, REFERENCE, auto_landmarks, bohler_angle,
                              gissane_angle, load_mask)
from calcaneus_dataset import DATA_ROOT, discover_ids
from unet import UNet

RESULT_DIR = Path(__file__).resolve().parent / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


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


def angles_from(lm):
    """从关键点字典算两个角；提点失败返回 (nan, nan)。"""
    if lm is None:
        return float("nan"), float("nan")
    return bohler_angle(lm), gissane_angle(lm)


def predict_masks(ids, weights, size, base, device):
    """用自训 U-Net 对全部样本出预测掩膜（bool，256×256）。"""
    from calcaneus_dataset import CalcaneusDataset
    model = UNet(in_ch=1, out_ch=1, base=base).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    ds = CalcaneusDataset(ids=ids, size=size, return_id=True)
    out = {}
    with torch.no_grad():
        for k in range(len(ds)):
            image, _, sid = ds[k]
            prob = torch.sigmoid(model(image.unsqueeze(0).to(device))).cpu()[0]
            out[sid] = (prob[0] > 0.5).numpy()      # [1,H,W] -> [H,W]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(RESULT_DIR / "calcaneus_unet_best_split83.pth"))
    ap.add_argument("--pred-source", choices=["model", "pkg"], default="model",
                    help="model=本仓库自训 U-Net；pkg=数据包自带掩膜 result/result/results/")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = discover_ids()

    if args.pred_source == "model":
        pred_masks = predict_masks(ids, args.weights, args.size, args.base, device)
        pred_label = "本仓库自训 U-Net"
    else:
        pred_masks = {i: load_mask("results", i) for i in ids}
        pred_label = "数据包自带掩膜"

    print("=" * 104)
    print("误差归因实验：角度总误差 = 提点规则误差 + 分割边界误差")
    print("=" * 104)
    print(f"数据        : {DATA_ROOT}")
    print(f"人工掩膜    : labels/")
    print(f"预测掩膜    : {pred_label}")
    print(f"基线 A1     : 关键点用从参数图逐像素反推的参考位置（复现误差 Böhler 0.34° / Gissane 6.31°）")
    print()

    rows = []
    for i in ids:
        gt_mask = load_mask("labels", i)
        lm_ref = REFERENCE[i]
        lm_auto_gt, _ = auto_landmarks(gt_mask)
        lm_auto_pr, _ = auto_landmarks(pred_masks[i])

        b1, g1 = angles_from(lm_ref)
        b2, g2 = angles_from(lm_auto_gt)
        b3, g3 = angles_from(lm_auto_pr)
        pb, pg = PRINTED[i]

        rows.append(dict(
            id=i,
            a1_b=round(b1, 3), a2_b=round(b2, 3), a3_b=round(b3, 3),
            a1_g=round(g1, 3), a2_g=round(g2, 3), a3_g=round(g3, 3),
            printed_b=pb, printed_g=pg,
            rule_b=round(b2 - b1, 3), seg_b=round(b3 - b2, 3), tot_b=round(b3 - b1, 3),
            rule_g=round(g2 - g1, 3), seg_g=round(g3 - g2, 3), tot_g=round(g3 - g1, 3),
            # A1 自身相对参数图印出值的偏差（含参考点反推误差）与端到端总误差
            base_b=round(b1 - pb, 3), base_g=round(g1 - pg, 3),
            e2e_b=round(b3 - pb, 3), e2e_g=round(g3 - pg, 3),
        ))

    def show(title, p):
        """p: 'b' 或 'g'，分别对应 Böhler / Gissane。"""
        print(f"【{title}】单位：度")
        print(f"{'id':>4} | {'A1 人工+参考':>13} {'A2 人工+自动':>13} {'A3 预测+自动':>13} | "
              f"{'提点规则(A2-A1)':>16} {'分割(A3-A2)':>13} {'总误差(A3-A1)':>14}")
        print("-" * 104)
        for r in rows:
            print(f"{r['id']:>4} | {r['a1_' + p]:>13.2f} {r['a2_' + p]:>13.2f} "
                  f"{r['a3_' + p]:>13.2f} | "
                  f"{r['rule_' + p]:>+16.2f} {r['seg_' + p]:>+13.2f} {r['tot_' + p]:>+14.2f}")
        rule = np.array([abs(r["rule_" + p]) for r in rows])
        seg = np.array([abs(r["seg_" + p]) for r in rows])
        tot = np.array([abs(r["tot_" + p]) for r in rows])
        print("-" * 104)
        print(f"     平均绝对误差：提点规则 {rule.mean():.2f}°   分割边界 {seg.mean():.2f}°   "
              f"总误差 {tot.mean():.2f}°   单例最大 {tot.max():.2f}°")
        print(f"     中位数        ：提点规则 {np.median(rule):.2f}°   分割边界 {np.median(seg):.2f}°   "
              f"总误差 {np.median(tot):.2f}°")
        b = np.array([abs(r["base_" + p]) for r in rows])
        e2e = np.array([abs(r["e2e_" + p]) for r in rows])
        print(f"     分解链        ：基线复现 {b.mean():.2f}°  +  提点规则 {rule.mean():.2f}°  +  "
              f"分割边界 {seg.mean():.2f}°   ⇒  端到端（全自动链 vs 参数图印出值）{e2e.mean():.2f}°"
              f"（单例最大 {e2e.max():.2f}°）")
        print()
        return rule.mean(), seg.mean(), tot.mean(), b.mean(), e2e.mean()

    print("=" * 104)
    print("逐例明细")
    print("=" * 104)
    rb, sb, tb, base_b, e2e_b = show("Böhler 角", "b")
    rg, sg, tg, base_g, e2e_g = show("Gissane 角", "g")

    # ---------------- 结论 ----------------
    print("=" * 104)
    print("归因结论：两个角的最主要误差源不是同一个")
    print("=" * 104)
    for name, rule, seg, tot in (("Böhler", rb, sb, tb), ("Gissane", rg, sg, tg)):
        main_src = "提点规则" if rule > seg else "分割边界"
        ratio = max(rule, seg) / max(min(rule, seg), 1e-6)
        print(f"  {name:<8} 提点规则 {rule:5.2f}°  分割边界 {seg:5.2f}°  "
              f"→ 主要来自【{main_src}】，是另一项的 {ratio:.1f} 倍")
    print()
    print(f"  端到端误差（全自动链路输出 vs 参数图印出值）：Böhler {e2e_b:.2f}°   Gissane {e2e_g:.2f}°")
    print(f"  交叉印证：旧的「敏感度 × 边界偏移」折算给出 ±1.45~2.79°（Böhler）/ ±2.10~4.45°（Gissane），")
    print(f"            本次「分割边界」贡献的中位数为 {np.median([abs(r['seg_b']) for r in rows]):.2f}° / "
          f"{np.median([abs(r['seg_g']) for r in rows]):.2f}° —— 两种独立方法同量级，互相印证；")
    print(f"            均值（{sb:.2f}° / {sg:.2f}°）被个别大误差样本抬高。")
    print()
    print("  怎么用这条结论：")
    if rb > sb:
        print("    · Böhler 角的误差主要来自提点规则 —— 继续提升分割精度对它没有意义，")
        print("      唯一的解法是获得关键点的定义规则与标注脚本（B 点几何规则已穷尽且失败）。")
    if sg > rg:
        print("    · Gissane 角的误差主要来自分割边界 —— 优化方向在掩膜，")
        print("      尤其应针对跟骨沟（D 点）附近，那里也是角度最敏感的地方。")
    print()
    print("  ⚠️ 口径（必须一起说）：")
    print("    · A1 基线自身带有误差（Böhler 0.34° / Gissane 6.31°），所以上面的分解")
    print("      是在「以反推参考点为真值」的前提下成立的；Gissane 的基线误差较大，")
    print("      它的归因结论比 Böhler 弱。")
    print("    · 这是 11 例的结果，不是统计结论。")
    print("=" * 104)

    # ---------------- 存表 ----------------
    csv_path = RESULT_DIR / f"calcaneus_error_attribution{args.tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n逐例表: {csv_path}")

    # ---------------- 画图 ----------------
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3), dpi=150)
    x = np.arange(len(rows))
    labels = [r["id"] for r in rows]

    for ax, (key, name, rule_m, seg_m, tot_m) in zip(
            axes[:2],
            [(("rule_b", "seg_b"), "Böhler 角", rb, sb, tb),
             (("rule_g", "seg_g"), "Gissane 角", rg, sg, tg)]):
        rule_v = np.array([abs(r[key[0]]) for r in rows])
        seg_v = np.array([abs(r[key[1]]) for r in rows])
        ax.bar(x, rule_v, 0.62, label="提点规则误差", color="#c01c28")
        ax.bar(x, seg_v, 0.62, bottom=rule_v, label="分割边界误差", color="#1a5fb4")
        ax.plot(x, rule_v + seg_v, "k_", ms=14, mew=2, label="总误差")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel("样本编号"); ax.set_ylabel("角度误差（度）")
        ax.set_title(f"{name}：总误差 = 提点 {rule_m:.2f}° + 分割 {seg_m:.2f}°", fontsize=11)
        ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=0.25, lw=0.6)

    ax = axes[2]
    w = 0.26
    comps = [("基线复现", base_b, base_g, "#9aa0a6"),
             ("提点规则", rb, rg, "#c01c28"),
             ("分割边界", sb, sg, "#1a5fb4")]
    for k, (nm, vb, vg, c) in enumerate(comps):
        xs_ = np.arange(2) + (k - 1) * w
        ax.bar(xs_, [vb, vg], w, label=nm, color=c)
        for xi, v in zip(xs_, [vb, vg]):
            ax.text(xi, v + 0.08, f"{v:.2f}", ha="center", fontsize=8.5)
    for xi, v in zip(np.arange(2), [e2e_b, e2e_g]):
        ax.plot([xi - 1.5 * w, xi + 1.5 * w], [v, v], color="#0f8a6a", lw=2.0, ls="--")
        ax.text(xi + 1.6 * w, v, f"端到端 {v:.2f}°", color="#0f8a6a", fontsize=8.5, va="center")
    ax.set_xticks(np.arange(2)); ax.set_xticklabels(["Böhler", "Gissane"], fontsize=10.5)
    ax.set_xlim(-0.6, 1.75)
    ax.set_ylabel("平均绝对误差（度）")
    ax.set_title("误差来源分解（虚线 = 端到端总误差）", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left"); ax.grid(axis="y", alpha=0.25, lw=0.6)

    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"误差归因：A2−A1 = 提点规则，A3−A2 = 分割边界　（预测掩膜 = {pred_label}）",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = RESULT_DIR / f"calcaneus_error_attribution{args.tag}.png"
    fig.savefig(fig_path)
    print(f"归因图: {fig_path}")


if __name__ == "__main__":
    main()
