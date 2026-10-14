# -*- coding: utf-8 -*-
"""
11 折留一交叉验证（Leave-One-Out）

为什么做这个：
    原来的 8 训练 / 3 验证只用了 3 例当测试，Dice 在 0.88~0.96 之间跳，
    说"模型多好"或"模型多差"都没有依据。留一验证让**每一例都当过一次测试样本**，
    于是能给出一个分布，而不是 3 个数字。

它解决什么、不解决什么（必须说清楚）：
    解决：不再依赖"这一次划分的运气"。任何一例被挑出来单独看，都有它的留一结果。
    不解决：11 例仍然只有 11 例。留一验证不产生泛化性能结论，
           它只是把"指标的不确定性"从暗处搬到明处。
    注意：留一每折用 10 例训练（8-3 划分只用了 8 例），训练集更大，
         所以留一的绝对数值会比 8-3 划分略好，两者不能直接比大小。

用法：
    python leave_one_out.py                    # 全部 11 折，约 2 分钟
    python leave_one_out.py --epochs 60        # 快速跑一遍
    python leave_one_out.py --repeats 2        # 每折跑 2 个种子，看种子间波动
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from calcaneus_dataset import CalcaneusDataset, dice_score, discover_ids, iou_score
from train_calcaneus import evaluate, set_seed, setup_matplotlib
from unet import UNet, count_parameters

RESULT_DIR = Path(__file__).resolve().parent / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


class SoftDiceLoss(nn.Module):
    """软 Dice 损失（对 sigmoid 后的概率算，不用先二值化）。"""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        num = 2.0 * (p * target).sum(dim=(2, 3)) + self.eps
        den = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + self.eps
        return (1.0 - num / den).mean()


def build_criterion(name):
    """name: bce（原配置） | dice_bce（0.5*BCE + 0.5*Dice，用来压训练崩溃）"""
    if name == "dice_bce":
        bce = nn.BCEWithLogitsLoss()
        dice = SoftDiceLoss()
        return lambda logits, target: 0.5 * bce(logits, target) + 0.5 * dice(logits, target)
    return nn.BCEWithLogitsLoss()


def run_fold(full_ids, test_id, args, device):
    """训练「除 test_id 以外的全部样本」、测试 test_id，返回该折的指标。

    注意：训练集必须从 full_ids 里减，不能用筛选后的列表 ——
    否则 --only 诊断时训练集会被一起筛掉。
    """
    train_ids = [i for i in full_ids if i != test_id]
    train_ds = CalcaneusDataset(ids=train_ids, size=args.size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    test_ds = CalcaneusDataset(ids=[test_id], size=args.size)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    model = UNet(in_ch=1, out_ch=1, base=args.base).to(device)
    criterion = build_criterion(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for _ in range(args.epochs):
        model.train()
        for image, mask in train_loader:
            image, mask = image.to(device), mask.to(device)
            loss = criterion(model(image), mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    test_loss, test_dice, test_iou = evaluate(model, test_loader, criterion, device)

    # 额外记录：测试例的前景占比，以及预测面积 / GT 面积（判断"缩"还是"胀"）
    image, mask = test_ds[0]
    with torch.no_grad():
        pred = (torch.sigmoid(model(image.unsqueeze(0).to(device))).cpu()[0] > 0.5).float()
    gt_fg = float(mask.mean())
    pr_fg = float(pred.mean())
    return dict(test_id=test_id, test_dice=round(test_dice, 4), test_iou=round(test_iou, 4),
                test_loss=round(test_loss, 4),
                gt_fg_pct=round(gt_fg * 100, 2), pred_fg_pct=round(pr_fg * 100, 2),
                area_ratio=round(pr_fg / max(gt_fg, 1e-6), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=1, help="每折重复几个种子，用来观察种子间波动")
    ap.add_argument("--loss", choices=["bce", "dice_bce"], default="bce",
                    help="bce=原配置；dice_bce=混合损失，用于压制训练崩溃")
    ap.add_argument("--only", type=str, default="",
                    help="只跑指定的测试例，逗号分隔，如 31,73 —— 用于诊断崩溃折")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_ids = discover_ids()          # 完整样本集，训练集永远从这里减
    all_ids = full_ids
    if args.only.strip():
        keep = [x.strip() for x in args.only.split(",") if x.strip()]
        all_ids = [i for i in all_ids if i in keep]

    print("=" * 96)
    print("11 折留一交叉验证（Leave-One-Out）")
    print("=" * 96)
    print(f"device = {device}   样本 {len(all_ids)} 例：{all_ids}")
    print(f"配置   : epochs={args.epochs} base={args.base} lr={args.lr} "
          f"seed={args.seed} repeats={args.repeats} loss={args.loss}")
    print(f"含义   : 每折用 {len(full_ids)-1} 例训练、1 例测试，逐折轮流 —— 每一例都会当一次测试样本")
    print()
    print(f"{'折':>3} {'测试例':>6} {'种子':>5} | {'Dice':>7} {'IoU':>7} | "
          f"{'GT前景%':>8} {'预测%':>7} {'面积比':>7}")
    print("-" * 72)

    rows, t0 = [], time.time()
    for fold, test_id in enumerate(all_ids, 1):
        for r in range(args.repeats):
            seed = args.seed + r * 1000
            set_seed(seed)                      # 每折都重置，保证可复现
            res = run_fold(full_ids, test_id, args, device)
            res.update(fold=fold, seed=seed, loss=args.loss)
            rows.append(res)
            print(f"{fold:>3} {test_id:>6} {seed:>5} | {res['test_dice']:>7.4f} "
                  f"{res['test_iou']:>7.4f} | {res['gt_fg_pct']:>8.2f} "
                  f"{res['pred_fg_pct']:>7.2f} {res['area_ratio']:>7.3f}")
    elapsed = time.time() - t0

    dices = np.array([r["test_dice"] for r in rows])
    print("-" * 72)
    print(f"用时 {elapsed:.1f} 秒")
    print()
    print("=" * 96)
    print("汇总（这是本仓库现在能给的最诚实的指标表述）")
    print("=" * 96)
    print(f"  折数            : {len(rows)}")
    print(f"  Dice  均值±标准差 : {dices.mean():.4f} ± {dices.std(ddof=1):.4f}")
    print(f"  Dice  中位数      : {np.median(dices):.4f}   最小 {dices.min():.4f} 最大 {dices.max():.4f}")
    print(f"  Dice  四分位区间  : {np.percentile(dices, 25):.4f} ~ {np.percentile(dices, 75):.4f}")
    print(f"  IoU   均值        : {np.mean([r['test_iou'] for r in rows]):.4f}")
    if args.repeats > 1:
        per_fold = {}
        for r in rows:
            per_fold.setdefault(r["test_id"], []).append(r["test_dice"])
        spreads = [max(v) - min(v) for v in per_fold.values()]
        print(f"  同折不同种子的 Dice 极差: 最大 {max(spreads):.4f}  平均 {np.mean(spreads):.4f}")
    print()
    print("  逐例留一 Dice（按编号）：")
    order = sorted(rows, key=lambda r: int(r["test_id"]))
    print("    " + "  ".join(f"#{r['test_id']}={r['test_dice']:.4f}" for r in order))
    print()
    worst = min(rows, key=lambda r: r["test_dice"])
    print(f"  留一最差的一例: #{worst['test_id']}  Dice {worst['test_dice']:.4f}  "
          f"（面积比 {worst['area_ratio']}）")
    hard = sum(1 for r in rows if r["test_dice"] < 0.95)
    print(f"  Dice < 0.95 的例数: {hard} / {len(rows)}")
    print()
    print("  ⚠️ 口径：这是 11 例留一的结果分布，不是泛化性能。")
    print("     它比 8-3 划分强的地方在于：不再有'哪 3 例当验证集'这个运气因素。")
    print("     但它仍然只有 11 例 —— 样本量小这个根本问题没有被解决，只是被量化了。")
    print("=" * 96)

    # ---------------- 存表 ----------------
    tag = args.tag
    csv_path = RESULT_DIR / f"calcaneus_loo_results{tag}.csv"
    fields = ["fold", "test_id", "seed", "loss", "test_dice", "test_iou", "test_loss",
              "gt_fg_pct", "pred_fg_pct", "area_ratio"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n逐折表: {csv_path}")

    # ---------------- 画图 ----------------
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2), dpi=150)
    x = np.arange(len(order))
    ids_sorted = [r["test_id"] for r in order]
    vals = [r["test_dice"] for r in order]

    colors = ["#c01c28" if v < 0.95 else "#1a5fb4" for v in vals]
    axes[0].bar(x, vals, 0.62, color=colors)
    axes[0].axhline(dices.mean(), color="#1a7f4b", ls="--", lw=1.2,
                    label=f"均值 {dices.mean():.4f}")
    axes[0].axhline(0.95, color="#b45309", ls=":", lw=1.2, label="0.95 参考线")
    axes[0].set_xticks(x); axes[0].set_xticklabels(ids_sorted, fontsize=9)
    axes[0].set_ylim(min(0.80, min(vals) - 0.02), 1.0)
    axes[0].set_xlabel("留出的测试样本编号"); axes[0].set_ylabel("Dice")
    axes[0].set_title("11 折留一：每一例当测试样本时的 Dice", fontsize=11)
    axes[0].legend(fontsize=8.5); axes[0].grid(axis="y", alpha=0.25, lw=0.6)
    for xi, v in zip(x, vals):
        axes[0].text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=7.5, color="#3c4043")

    axes[1].hist(dices, bins=7, color="#1a5fb4", alpha=0.85, edgecolor="white")
    axes[1].axvline(dices.mean(), color="#1a7f4b", ls="--", lw=1.4,
                    label=f"均值 {dices.mean():.4f}")
    axes[1].axvline(np.median(dices), color="#b45309", ls=":", lw=1.4,
                    label=f"中位数 {np.median(dices):.4f}")
    axes[1].set_xlabel("Dice"); axes[1].set_ylabel("折数")
    axes[1].set_title(f"留一 Dice 分布（均值 {dices.mean():.4f} ± {dices.std(ddof=1):.4f}）",
                      fontsize=11)
    axes[1].legend(fontsize=8.5); axes[1].grid(axis="y", alpha=0.25, lw=0.6)

    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig_path = RESULT_DIR / f"calcaneus_loo{tag}.png"
    fig.savefig(fig_path)
    print(f"图    : {fig_path}")


if __name__ == "__main__":
    main()
