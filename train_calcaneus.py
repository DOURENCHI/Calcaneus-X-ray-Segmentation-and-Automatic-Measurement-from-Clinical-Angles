# -*- coding: utf-8 -*-
"""
跟骨 U-Net 分割基线 —— 训练 + 验证

用它跑两件事，分开看，别混：
  1) 过拟合测试（--overfit）：在全部 11 例上训练，看训练集 Dice 能不能压到 0.95+。
     这一条只证明"你的代码链路是通的"，不证明模型有泛化能力。
  2) 留出验证（默认）：8 例训练 / 3 例验证。3 例的验证集统计上很不可靠，
     只能当趋势看，不能拿来说"模型性能如何"。

用法：
    python train_calcaneus.py --overfit --epochs 80
    python train_calcaneus.py --epochs 80
    python train_calcaneus.py --epochs 120 --size 256 --base 16 --lr 1e-3

产物（都写在 results/ 下）：
    calcaneus_unet_best.pth      验证 Dice 最好的权重
    calcaneus_unet_last.pth      最后一轮的权重
    calcaneus_train_log.csv      逐轮指标
    calcaneus_train_curve.png    训练曲线
"""

import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from calcaneus_dataset import CalcaneusDataset, dice_score, iou_score, discover_ids
from unet import UNet, count_parameters

RESULT_DIR = Path(__file__).resolve().parent / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """按整图算 Dice / IoU（不是按 batch 平均），11 例规模下这样更准。"""
    model.eval()
    total_loss, total_dice, total_iou, n = 0.0, 0.0, 0.0, 0
    for image, mask in loader:
        image, mask = image.to(device), mask.to(device)
        logits = model(image)
        loss = criterion(logits, mask)
        pred = (torch.sigmoid(logits) > 0.5).float()
        total_loss += loss.item()
        total_dice += dice_score(pred, mask)
        total_iou += iou_score(pred, mask)
        n += 1
    return total_loss / max(n, 1), total_dice / max(n, 1), total_iou / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256, help="输入边长，跟骨原始裁剪图就是 256")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=16, help="UNet 第一层通道数")
    ap.add_argument("--val-ids", type=str, default="", help="逗号分隔，如 88,91,94")
    ap.add_argument("--overfit", action="store_true", help="全部数据都训练（过拟合测试）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="", help="给产物文件名加后缀，方便保存多组实验")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"_{args.tag}" if args.tag else ""

    print("=" * 74)
    print(f"device = {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    print(f"size={args.size}  epochs={args.epochs}  batch={args.batch_size}  "
          f"lr={args.lr}  base={args.base}  overfit={args.overfit}  seed={args.seed}")
    print("=" * 74)

    # ---------------- 数据 ----------------
    all_ids = discover_ids()
    if args.overfit:
        train_ids, val_ids = list(all_ids), []
    else:
        if args.val_ids:
            val_ids = [s.strip() for s in args.val_ids.split(",") if s.strip()]
        else:
            rng = random.Random(args.seed)
            val_ids = sorted(rng.sample(all_ids, 3), key=lambda s: int(s))
        train_ids = [i for i in all_ids if i not in val_ids]

    print(f"全部样本 : {all_ids}")
    print(f"训练集   : {train_ids}  ({len(train_ids)} 例)")
    print(f"验证集   : {val_ids if val_ids else '（无，过拟合测试）'}")
    print()

    train_ds = CalcaneusDataset(ids=train_ids, size=args.size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = None
    if val_ids:
        val_ds = CalcaneusDataset(ids=val_ids, size=args.size)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # ---------------- 模型 ----------------
    model = UNet(in_ch=1, out_ch=1, base=args.base).to(device)
    n_param = count_parameters(model)
    print(f"U-Net 可训练参数量 = {n_param:,}")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---------------- 训练 ----------------
    log_rows = []
    best_val_dice = -1.0
    history = {"epoch": [], "train_loss": [], "train_dice": [], "val_loss": [], "val_dice": []}
    t0 = time.time()

    header = f"{'epoch':>5} | {'train_loss':>10} {'train_dice':>10} {'train_IoU':>10}"
    if val_loader:
        header += f" | {'val_loss':>9} {'val_dice':>9} {'val_IoU':>9}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        model.train()
        sum_loss = sum_dice = sum_iou = 0.0
        n = 0
        for image, mask in train_loader:
            image, mask = image.to(device), mask.to(device)
            logits = model(image)
            loss = criterion(logits, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = (torch.sigmoid(logits) > 0.5).float()
                sum_dice += dice_score(pred, mask)
                sum_iou += iou_score(pred, mask)
            sum_loss += loss.item()
            n += 1

        tr_loss, tr_dice, tr_iou = sum_loss / max(n, 1), sum_dice / max(n, 1), sum_iou / max(n, 1)
        row = {"epoch": epoch, "train_loss": tr_loss, "train_dice": tr_dice, "train_iou": tr_iou}

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_dice"].append(tr_dice)

        if val_loader:
            va_loss, va_dice, va_iou = evaluate(model, val_loader, criterion, device)
            row.update({"val_loss": va_loss, "val_dice": va_dice, "val_iou": va_iou})
            history["val_loss"].append(va_loss)
            history["val_dice"].append(va_dice)

            if va_dice > best_val_dice:
                best_val_dice = va_dice
                torch.save(model.state_dict(), RESULT_DIR / f"calcaneus_unet_best{tag}.pth")
        else:
            # 过拟合测试：没有独立验证集，用训练 Dice 当"最好"的判据
            if tr_dice > best_val_dice:
                best_val_dice = tr_dice
                torch.save(model.state_dict(), RESULT_DIR / f"calcaneus_unet_best{tag}.pth")

        log_rows.append(row)

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            line = f"{epoch:>5} | {tr_loss:>10.4f} {tr_dice:>10.4f} {tr_iou:>10.4f}"
            if val_loader:
                line += f" | {row['val_loss']:>9.4f} {row['val_dice']:>9.4f} {row['val_iou']:>9.4f}"
            print(line)

    elapsed = time.time() - t0
    torch.save(model.state_dict(), RESULT_DIR / f"calcaneus_unet_last{tag}.pth")

    # ---------------- 存日志 ----------------
    fields = ["epoch", "train_loss", "train_dice", "train_iou"]
    if val_loader:
        fields += ["val_loss", "val_dice", "val_iou"]
    csv_path = RESULT_DIR / f"calcaneus_train_log{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(log_rows)

    # ---------------- 画曲线 ----------------
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=150)

    axes[0].plot(history["epoch"], history["train_loss"], color="#1a5fb4", label="训练 Loss")
    if val_loader:
        axes[0].plot(history["epoch"], history["val_loss"], color="#c01c28", label="验证 Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].set_title("损失曲线", fontsize=11)
    axes[0].grid(alpha=0.25, linewidth=0.6)
    axes[0].legend(fontsize=9)

    axes[1].plot(history["epoch"], history["train_dice"], color="#1a5fb4", label="训练 Dice")
    if val_loader:
        axes[1].plot(history["epoch"], history["val_dice"], color="#c01c28", label="验证 Dice")
    axes[1].axhline(0.95, color="#888780", linestyle="--", linewidth=0.9)
    axes[1].text(1, 0.955, "0.95", fontsize=8, color="#5F5E5A")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("Dice")
    axis_title = "Dice 曲线（过拟合测试）" if not val_loader else "Dice 曲线（8 训练 / 3 验证）"
    axes[1].set_title(axis_title, fontsize=11)
    axes[1].grid(alpha=0.25, linewidth=0.6)
    axes[1].legend(fontsize=9)

    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout()
    curve_path = RESULT_DIR / f"calcaneus_train_curve{tag}.png"
    fig.savefig(curve_path)

    # ---------------- 总结 ----------------
    print()
    print("=" * 74)
    print(f"训练完成，用时 {elapsed:.1f} 秒")
    print(f"  训练集 Dice 末轮 = {history['train_dice'][-1]:.4f}   最好 = {max(history['train_dice']):.4f}")
    if val_loader:
        print(f"  验证集 Dice 末轮 = {history['val_dice'][-1]:.4f}   最好 = {best_val_dice:.4f}")
        print(f"  验证集 IoU  末轮 = {log_rows[-1]['val_iou']:.4f}")
    print(f"  参数量 = {n_param:,}")
    print()
    print("产物:")
    print(f"  {RESULT_DIR / f'calcaneus_unet_best{tag}.pth'}")
    print(f"  {RESULT_DIR / f'calcaneus_unet_last{tag}.pth'}")
    print(f"  {csv_path}")
    print(f"  {curve_path}")
    print()
    if not val_loader:
        verdict = "链路已通" if max(history["train_dice"]) > 0.95 else "链路通了但拟合不足，需要加 epoch 或调 lr"
        print(f"【判断】过拟合测试：{verdict}")
        print("        注意：这只能说明你的数据->模型->训练->预测链路是通的，")
        print("        不能说明模型在没见过的数据上表现如何。")
    else:
        print("【判断】8 训练 / 3 验证只是趋势参考。3 例验证集统计上不可靠，")
        print("        不要拿这几个数字去汇报'模型性能'。要结论就得先拿到更多标注样本。")
    print("=" * 74)


if __name__ == "__main__":
    main()
