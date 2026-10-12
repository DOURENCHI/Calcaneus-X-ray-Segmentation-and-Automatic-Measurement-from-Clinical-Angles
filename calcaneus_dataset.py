# -*- coding: utf-8 -*-
"""
跟骨 X 光分割 —— 数据集

只读 D:\\论文\\方向与规划\\项目\\跟骨\\result\\result 下的 images/ 与 labels/，
按"编号"配对（images/13.png <-> labels/13.png）。

设计说明：
1. 配对用的是文件名去掉扩展名，而不是目录里的排序位置 —— 这样即使两个目录
   里文件顺序不一致，也不会把 A 的图和 B 的掩膜配到一起。
2. mask 只做 NEAREST 缩放并二值化（>0.5 -> 1），因为它是标签，不能用插值。
3. 归一化默认沿用你在 dataset.py 里的 min-max 写法，保留记忆点。

单独运行本文件会做一次自检：打印配对结果 + 张量形状/取值范围。
    python calcaneus_dataset.py
"""

from pathlib import Path
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# =========================
# 数据路径（只读，不要改动这里面的文件）
# =========================
# 说明：这个目录被改过名（2026-10-12 从 D:\论文 改成 D:\工程），
# 所以这里不写死单一路径，而是按候选列表自动探测，也支持用环境变量覆盖：
#     set CALCANEUS_ROOT=D:\你的\路径\result\result

_CANDIDATE_ROOTS = [
    Path(r"D:\工程\方向与规划\项目\跟骨\result\result"),   # 2026-10-12 起的路径
    Path(r"D:\论文\方向与规划\项目\跟骨\result\result"),   # 旧路径
]


def _resolve_root():
    env_root = os.environ.get("CALCANEUS_ROOT")
    if env_root:
        return Path(env_root)
    for candidate in _CANDIDATE_ROOTS:
        if candidate.exists():
            return candidate
    return _CANDIDATE_ROOTS[0]


DATA_ROOT = _resolve_root()
IMAGE_DIR = DATA_ROOT / "images"
LABEL_DIR = DATA_ROOT / "labels"

# 全部 11 例的编号（从文件名自动扫描得到，这里只是默认值）
ALL_IDS = ["1", "13", "23", "31", "36", "61", "73", "75", "88", "91", "94"]


def discover_ids(image_dir: Path = IMAGE_DIR, mask_dir: Path = LABEL_DIR):
    """扫描两个目录，返回两边都有的编号（按数字大小排序）。"""
    image_stems = {p.stem for p in Path(image_dir).glob("*.png")}
    mask_stems = {p.stem for p in Path(mask_dir).glob("*.png")}

    only_image = sorted(image_stems - mask_stems)
    only_mask = sorted(mask_stems - image_stems)
    if only_image or only_mask:
        print(f"[警告] 未配对的图像: {only_image} / 未配对的掩膜: {only_mask}")

    both = image_stems & mask_stems
    return sorted(both, key=lambda s: int(s) if s.isdigit() else 1 << 30)


class CalcaneusDataset(Dataset):
    """跟骨 X 光 + 人工分割掩膜。

    返回
    ----
    image : torch.float32, shape [1, H, W], 取值 0~1
    mask  : torch.float32, shape [1, H, W], 取值 0 或 1
    (return_id=True 时额外返回样本编号字符串，方便追溯)
    """

    def __init__(self, ids=None, image_dir=IMAGE_DIR, mask_dir=LABEL_DIR,
                 size=256, norm="minmax", return_id=False):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.size = size
        self.norm = norm
        self.return_id = return_id
        self.ids = list(ids) if ids is not None else discover_ids(self.image_dir, self.mask_dir)

        missing = [i for i in self.ids
                   if not (self.image_dir / f"{i}.png").exists()
                   or not (self.mask_dir / f"{i}.png").exists()]
        if missing:
            raise FileNotFoundError(f"这些编号在 images/ 或 labels/ 里不存在: {missing}")

    def __len__(self):
        return len(self.ids)

    def _load_image(self, path):
        image = Image.open(path).convert("L")
        if self.size is not None:
            image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        image = np.asarray(image).astype(np.float32)

        if self.norm == "minmax":          # 沿用你 dataset.py 的写法
            image = image - image.min()
            if image.max() > 0:
                image = image / image.max()
        else:                              # "div255"
            image = image / 255.0
        return image

    def _load_mask(self, path):
        mask = Image.open(path).convert("L")
        if self.size is not None:
            mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        mask = np.asarray(mask).astype(np.float32) / 255.0
        # 防插值/压缩产生中间值：二值化
        mask = (mask > 0.5).astype(np.float32)
        return mask

    def __getitem__(self, index):
        sample_id = self.ids[index]
        image = self._load_image(self.image_dir / f"{sample_id}.png")
        mask = self._load_mask(self.mask_dir / f"{sample_id}.png")

        # [H, W] -> [1, H, W]
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        if self.return_id:
            return image, mask, sample_id
        return image, mask


def dice_score(pred_bin, target_bin, eps=1e-7):
    """二值 Dice。pred_bin / target_bin 是 0/1 张量，形状任意。"""
    pred_bin = pred_bin.float()
    target_bin = target_bin.float()
    inter = (pred_bin * target_bin).sum()
    return ((2 * inter + eps) / (pred_bin.sum() + target_bin.sum() + eps)).item()


def iou_score(pred_bin, target_bin, eps=1e-7):
    """二值 IoU。"""
    pred_bin = pred_bin.float()
    target_bin = target_bin.float()
    inter = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - inter
    return ((inter + eps) / (union + eps)).item()


if __name__ == "__main__":
    print("数据根目录:", DATA_ROOT)
    print("目录存在:", IMAGE_DIR.exists(), LABEL_DIR.exists())
    print()

    ids = discover_ids()
    print(f"扫描到 {len(ids)} 组配对样本: {ids}")
    print()

    ds = CalcaneusDataset(size=256, return_id=True)
    print("Dataset 长度:", len(ds))
    print()

    for idx in [0, len(ds) - 1]:
        image, mask, sample_id = ds[idx]
        print(f"--- 样本 {sample_id} ---")
        print("  image:", tuple(image.shape), image.dtype,
              f"min={image.min():.3f} max={image.max():.3f} mean={image.mean():.4f}")
        print("  mask :", tuple(mask.shape), mask.dtype,
              f"min={mask.min():.3f} max={mask.max():.3f}",
              "唯一值=", sorted(set(mask.unique().tolist())),
              f"前景占比={mask.mean().item() * 100:.2f}%")
    print()
    print("[自检通过] 配对、缩放、归一化、张量形状都正常。")
