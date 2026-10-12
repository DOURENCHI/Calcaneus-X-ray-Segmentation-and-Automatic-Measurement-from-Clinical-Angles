# -*- coding: utf-8 -*-
"""
跟骨角度管线：从掩膜到 Böhler / Gissane 角

这一版做四件事，每件都独立可验证：

  1) 角度公式 + 参考关键点
     用逐像素反推出来的 11 组关键点，复现参数图上印出的数值
     （Böhler 平均误差 0.34°、Gissane 约 1.9°）—— 这是整条链路的"标尺"。

  2) 分割边界误差
     预测掩膜 vs 人工掩膜的轮廓对称表面距离（mean SSD / HD95）。
     这是"分割做得好不好"的量化，也是第 3 步的输入。

  3) 角度敏感度 → 误差预算
     对每个关键点、每个方向做中心差分，算出 dθ/dp（度/像素）。
     再乘上第 2 步测到的边界偏移，得到"分割误差会传导成多少度角度误差"。
     —— 这一步不依赖关键点的定义来源，纯几何，结论是硬的。

  4) 自动提点（实验性）
     从掩膜上边界用几何规则自动定位 A/B/C/D，并逐点报告与参考点的误差。
     结论会告诉你：哪几个点可以自动提，哪个点不行。

用法：
    & "D:\\Miniconda\\envs\\medimg\\python.exe" calcaneus_angles.py
    & "D:\\Miniconda\\envs\\medimg\\python.exe" calcaneus_angles.py --mask results   # 用预测掩膜提点
    & "D:\\Miniconda\\envs\\medimg\\python.exe" calcaneus_angles.py --mask labels    # 用人工掩膜提点
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calcaneus_dataset import DATA_ROOT, discover_ids

RESULT_DIR = Path(__file__).resolve().parent / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# 参数图上印出的数值（我逐张放大读出来的，作为对照基准）
# =====================================================================
PRINTED = {
    "1":  (6.00, 164.19), "13": (13.93, 156.61), "23": (3.65, 156.03),
    "31": (11.87, 153.19), "36": (10.48, 160.91), "61": (18.99, 162.90),
    "73": (8.16, 146.82), "75": (9.09, 178.48), "88": (11.48, 147.71),
    "91": (3.82, 145.46), "94": (5.83, 163.21),
}

# =====================================================================
# 从参数图逐像素反推出来的 4 个关键点（A/B/C = 3 个绿点，D = 红点）
# 这些是"已知正确答案"，用来当标尺；也是 --mode reference 的输入
# =====================================================================
REFERENCE = {
    "1":  dict(A=(82.6, 136.7), B=(160.7, 95.3), C=(184.0, 97.0), D=(114, 132)),
    "13": dict(A=(71.5, 167.2), B=(120.1, 102.6), C=(177.0, 81.0), D=(102, 149)),
    "23": dict(A=(81.7, 133.5), B=(160.5, 110.8), C=(209.0, 107.0), D=(112, 138)),
    "31": dict(A=(69.6, 134.7), B=(140.7, 82.3), C=(198.0, 76.0), D=(102, 128)),
    "36": dict(A=(78.6, 127.9), B=(147.7, 77.3), C=(203.0, 68.0), D=(123, 114)),
    "61": dict(A=(97.9, 131.2), B=(183.2, 96.2), C=(216.8, 123.9), D=(144, 126)),
    "73": dict(A=(86.1, 131.0), B=(165.7, 87.3), C=(203.0, 88.0), D=(127, 133)),
    "75": dict(A=(39.7, 144.3), B=(129.4, 100.4), C=(179.0, 102.0), D=(82, 128)),
    "88": dict(A=(92.1, 122.0), B=(155.7, 87.3), C=(202.0, 89.0), D=(125, 123)),
    "91": dict(A=(88.6, 123.3), B=(140.7, 102.3), C=(161.0, 101.0), D=(119, 129)),
    "94": dict(A=(84.6, 132.9), B=(124.9, 105.3), C=(164.0, 90.0), D=(111, 126)),
}


# =====================================================================
# 1. 角度公式
# =====================================================================
def angle_at_vertex(vertex, p, q):
    """角 p-vertex-q，返回度数。注意 vertex 是顶点（中间那个字母）。"""
    v1 = np.asarray(p, float) - np.asarray(vertex, float)
    v2 = np.asarray(q, float) - np.asarray(vertex, float)
    c = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def bohler_angle(lm):
    """Böhler = 角 A-B-C（顶点 A），两条边 A→B 与 A→C。"""
    return angle_at_vertex(lm["A"], lm["B"], lm["C"])


def gissane_angle(lm):
    """Gissane = 角 D-A-C（顶点 D），两条边 D→A 与 D→C。"""
    return angle_at_vertex(lm["D"], lm["A"], lm["C"])


def bohler_textbook(lm):
    """教科书定义：顶点在后关节面(B)，两条边到 A 与 C。用于对比，不是脚本口径。"""
    return angle_at_vertex(lm["B"], lm["A"], lm["C"])


def gissane_textbook(lm):
    """教科书定义：顶点在跟骨沟(D)，两条边到 A 与 B。"""
    return angle_at_vertex(lm["D"], lm["A"], lm["B"])


# =====================================================================
# 2. 掩膜与轮廓
# =====================================================================
# 说明：这台机器的 medimg 环境里没有 scipy（MONAI 装了但没带 scipy），
# 所以距离变换和腐蚀都用 numpy 自己实现，避免额外依赖。

def load_mask(kind, sample_id, root=DATA_ROOT):
    path = Path(root) / kind / f"{sample_id}.png"
    return np.array(Image.open(path).convert("L")) > 127


def _erode(mask):
    """4 邻域腐蚀一格。"""
    m = mask.astype(bool)
    er = m.copy()
    er[1:, :] &= m[:-1, :]
    er[:-1, :] &= m[1:, :]
    er[:, 1:] &= m[:, :-1]
    er[:, :-1] &= m[:, 1:]
    return er


def contour(mask):
    """掩膜的内边界（1 像素宽）。"""
    m = mask.astype(bool)
    return m & ~_erode(m)


def edt_sq(binary):
    """精确平方欧氏距离变换：返回每个像素到最近 True 像素的平方距离。

    用 Felzenszwalb & Huttenlocher 的一维抛物线下包络算法，行、列各扫一遍。
    """
    INF = 1e20

    def dt1d(f):
        n = len(f)
        d = np.empty(n)
        v = np.zeros(n, dtype=int)
        z = np.empty(n + 1)
        k = 0
        v[0] = 0
        z[0] = -INF
        z[1] = INF
        for q in range(1, n):
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
            while s <= z[k]:
                k -= 1
                s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
            k += 1
            v[k] = q
            z[k] = s
            z[k + 1] = INF
        k = 0
        for q in range(n):
            while z[k + 1] < q:
                k += 1
            d[q] = (q - v[k]) ** 2 + f[v[k]]
        return d

    f = np.where(binary, 0.0, INF).astype(np.float64)
    for y in range(f.shape[0]):
        f[y] = dt1d(f[y])
    for x in range(f.shape[1]):
        f[:, x] = dt1d(f[:, x])
    return f


def surface_distance_stats(pred, gt):
    """预测轮廓与人工轮廓的对称表面距离（像素）。"""
    cp, cg = contour(pred), contour(gt)
    if not cp.any() or not cg.any():
        return dict(mean=float("nan"), hd95=float("nan"), max=float("nan"))
    d_pg = np.sqrt(edt_sq(cg))[cp]      # 预测轮廓每点到人工轮廓的距离
    d_gp = np.sqrt(edt_sq(cp))[cg]
    both = np.concatenate([d_pg, d_gp])
    return dict(mean=float(both.mean()),
                hd95=float(np.percentile(both, 95)),
                max=float(both.max()))


# =====================================================================
# 3. 敏感度 / 误差预算
# =====================================================================
def angle_gradient(angle_fn, lm, key, coord, h=0.5):
    """中心差分：关键点 key 的 coord('x'|'y') 挪 1 像素，角度变化多少度。"""
    def shifted(delta):
        out = {k: np.array(v, float) for k, v in lm.items()}
        out[key][0 if coord == "x" else 1] += delta
        return out
    return (angle_fn(shifted(+h)) - angle_fn(shifted(-h))) / (2 * h)


def sensitivity_table(lm, angle_fn):
    """返回 {点: (dθ/dx, dθ/dy, 梯度模长)}，单位 度/像素。"""
    out = {}
    for key in ("A", "B", "C", "D"):
        gx = angle_gradient(angle_fn, lm, key, "x")
        gy = angle_gradient(angle_fn, lm, key, "y")
        out[key] = (gx, gy, math.hypot(gx, gy))
    return out


# =====================================================================
# 4. 从掩膜上边界自动提点（实验性）
# =====================================================================
def _upper_profile(mask, win=9):
    H, W = mask.shape
    top = np.full(W, np.nan)
    for x in range(W):
        idx = np.nonzero(mask[:, x])[0]
        if len(idx):
            top[x] = idx.min()
    xs = np.nonzero(~np.isnan(top))[0]
    if len(xs) == 0:
        return None, None
    x0, x1 = int(xs.min()), int(xs.max())
    y = top[x0:x1 + 1]
    # 线性插值补洞
    nans = np.isnan(y)
    if nans.any():
        y[nans] = np.interp(np.nonzero(nans)[0], np.nonzero(~nans)[0], y[~nans])
    # 滑动平均
    k = np.ones(win) / win
    ys = np.convolve(np.pad(y, win // 2, mode="edge"), k, mode="valid")
    return ys, x0


def auto_landmarks(mask, win=9, spike_drop=15, slope_frac=0.35):
    """从掩膜上边界自动提取 A/B/C/D。返回 (landmarks, 说明dict)。

    规则（都写在注释里，便于你以后改成自己的口径）：
      A：上边界的"最前端"。若前 8 列内上边界骤降超过 spike_drop 像素，
         说明最前面是一根尖突，则取骤降之后的位置。
      C：上边界全局最高点（y 最小）—— 这条在小样本上很稳。
      D：跟骨沟。先把上边界求导，找上升最陡的位置，再向左回退到
         "陡升的起点"（|斜率| 落到峰值 35% 以内的那一列）。
      B：后关节面最高点。取 D 与 C 之间、上边界曲率（斜率变化）最大处。
    """
    ys, x0 = _upper_profile(mask, win)
    if ys is None:
        return None, {}
    n = len(ys)
    s = np.gradient(ys)                      # 斜率，负 = 轮廓在上升
    info = {}

    # ---- A ----
    ia = 0
    if n > 8 and (ys[0] - ys[7]) > spike_drop:
        hit = np.nonzero(ys[: max(9, n // 6)] <= ys[0] - spike_drop)[0]
        ia = int(hit[0]) if len(hit) else 0
    A = (x0 + ia, ys[ia])

    # ---- C ----
    ic = int(np.argmin(ys))
    C = (x0 + ic, ys[ic])

    # ---- D ----
    lo, hi = int(0.20 * n), int(0.80 * n)
    if hi <= lo:
        lo, hi = 0, n - 1
    i_steep = lo + int(np.argmin(s[lo:hi]))
    s_min = s[i_steep]
    id_ = i_steep
    if s_min < 0:
        while id_ > lo and abs(s[id_ - 1]) >= slope_frac * abs(s_min):
            id_ -= 1
    D = (x0 + id_, ys[id_])

    # ---- B ----
    i_lo, i_hi = max(id_ + 2, 0), max(ic - 1, id_ + 3)
    B = (x0 + i_hi, ys[i_hi])
    if i_hi > i_lo:
        curv = np.abs(np.gradient(s[i_lo:i_hi + 1]))
        ib = i_lo + int(np.argmax(curv))
        B = (x0 + ib, ys[ib])

    info = dict(x0=x0, x1=x0 + n - 1, iA=ia, iC=ic, iD=id_, D_slope=float(s_min))
    return dict(A=tuple(map(float, A)), B=tuple(map(float, B)),
                C=tuple(map(float, C)), D=tuple(map(float, D))), info


def nearest_contour_point(pt, cpts):
    """找离 pt 最近的轮廓点，返回 (点坐标, 距离)。cpts 形状 (N,2)，顺序为 (x,y)。"""
    d = np.hypot(cpts[:, 0] - pt[0], cpts[:, 1] - pt[1])
    i = int(d.argmin())
    return cpts[i], float(d[i])


def contour_points(mask):
    ys, xs = np.nonzero(contour(mask))
    return np.stack([xs, ys], axis=1).astype(float)


def local_displacement(lm, cp_pred, cp_gt):
    """每个关键点附近，预测轮廓相对人工轮廓偏移了多少（向量，像素）。"""
    out = {}
    for k, pt in lm.items():
        pp, _ = nearest_contour_point(pt, cp_pred)
        pg, _ = nearest_contour_point(pt, cp_gt)
        out[k] = pp - pg
    return out


# =====================================================================
# 主流程
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", choices=["results", "labels"], default="results",
                    help="用哪个掩膜做自动提点：results=模型预测，labels=人工标注")
    ap.add_argument("--out", type=str, default="calcaneus_angles")
    args = ap.parse_args()

    ids = discover_ids()
    print("=" * 96)
    print(f"数据根目录: {DATA_ROOT}")
    print(f"自动提点使用的掩膜: {args.mask}/")
    print("=" * 96)

    # ---------- 第一部分：参考关键点复现印出值 ----------
    print("\n【第一部分】用反推出来的参考关键点复现参数图上的数值")
    print(f"{'id':>4} | {'Böhler印':>9} {'Böhler算':>9} {'差':>7} | "
          f"{'Gissane印':>10} {'Gissane算':>10} {'差':>7} | {'Böhler教科书':>11} {'Gissane教科书':>12}")
    print("-" * 96)
    eb = eg = 0.0
    for i in ids:
        lm = REFERENCE[i]
        b, g = bohler_angle(lm), gissane_angle(lm)
        pb, pg = PRINTED[i]
        eb += abs(b - pb); eg += abs(g - pg)
        print(f"{i:>4} | {pb:>9.2f} {b:>9.2f} {b-pb:>+7.2f} | "
              f"{pg:>10.2f} {g:>10.2f} {g-pg:>+7.2f} | "
              f"{bohler_textbook(lm):>11.1f} {gissane_textbook(lm):>12.1f}")
    n = len(ids)
    print("-" * 96)
    print(f"     平均绝对误差: Böhler {eb/n:.2f}°   Gissane {eg/n:.2f}°")
    print("     最后两列是骨科教科书定义算出来的值 —— 和印出的数完全对不上，")
    print("     说明参考结果的顶点取法和教科书不一样，复现必须沿用同一口径。")

    # ---------- 第二部分：分割边界误差 ----------
    print("\n【第二部分】分割边界误差（预测掩膜 vs 人工掩膜，单位：像素）")
    print(f"{'id':>4} | {'mean SSD':>9} {'HD95':>8} {'最大值':>8}")
    print("-" * 40)
    sd_rows = {}
    for i in ids:
        sd = surface_distance_stats(load_mask("results", i), load_mask("labels", i))
        sd_rows[i] = sd
        print(f"{i:>4} | {sd['mean']:>9.2f} {sd['hd95']:>8.2f} {sd['max']:>8.2f}")
    mean_ssd = np.mean([sd_rows[i]["mean"] for i in ids])
    print("-" * 40)
    print(f"     11 例平均 mean SSD = {mean_ssd:.2f} 像素")

    # ---------- 第三部分：敏感度与误差预算 ----------
    print("\n【第三部分】角度敏感度（度/像素）与分割误差传导")
    print("  说明：dθ/dx、dθ/dy 是'关键点向 x 或 y 方向挪 1 像素，角度变化多少度'。")
    print("        误差预算 = 梯度模长 × 边界平均偏移（把分割误差线性折算成角度误差）。")
    print()
    hdr = (f"{'id':>4} | {'点':>3} | {'Böhler dθ/dx':>12} {'dθ/dy':>8} {'|∇θ|':>7} | "
           f"{'Gissane dθ/dx':>13} {'dθ/dy':>8} {'|∇θ|':>7} | {'预算B':>7} {'预算G':>7}")
    print(hdr)
    print("-" * len(hdr))
    budget_rows = []
    for i in ids:
        lm = REFERENCE[i]
        s_b = sensitivity_table(lm, bohler_angle)
        s_g = sensitivity_table(lm, gissane_angle)
        d = sd_rows[i]["mean"]
        pred_err_b = math.hypot(*[s_b[k][2] for k in ("A", "B", "C")]) * d
        pred_err_g = math.hypot(*[s_g[k][2] for k in ("A", "C", "D")]) * d
        budget_rows.append(dict(id=i, ssd=d, bohler_budget=pred_err_b, gissane_budget=pred_err_g,
                                **{f"grad_b_{k}": s_b[k][2] for k in "ABCD"},
                                **{f"grad_g_{k}": s_g[k][2] for k in "ABCD"}))
        for k in ("A", "B", "C", "D"):
            tail = f"{pred_err_b:>7.2f} {pred_err_g:>7.2f}" if k == "A" else ""
            print(f"{i:>4} | {k:>3} | {s_b[k][0]:>12.2f} {s_b[k][1]:>8.2f} {s_b[k][2]:>7.2f} | "
                  f"{s_g[k][0]:>13.2f} {s_g[k][1]:>8.2f} {s_g[k][2]:>7.2f} | {tail}")
        print("-" * len(hdr))

    mean_gb = {k: np.mean([r[f"grad_b_{k}"] for r in budget_rows]) for k in "ABCD"}
    mean_gg = {k: np.mean([r[f"grad_g_{k}"] for r in budget_rows]) for k in "ABCD"}
    print("     平均敏感度（度/像素）:")
    print("       对 Böhler : " + "  ".join(f"{k}={mean_gb[k]:.2f}" for k in "ABCD")
          + f"   → 最敏感的是 {max(mean_gb, key=mean_gb.get)}")
    print("       对 Gissane: " + "  ".join(f"{k}={mean_gg[k]:.2f}" for k in "ABCD")
          + f"   → 最敏感的是 {max(mean_gg, key=mean_gg.get)}")
    print(f"     平均误差预算: Böhler ±{np.mean([r['bohler_budget'] for r in budget_rows]):.2f}°  "
          f"Gissane ±{np.mean([r['gissane_budget'] for r in budget_rows]):.2f}°")

    # ---------- 第三·补：关键点附近的局部轮廓偏移 ----------
    print("\n【第三·补】关键点附近，预测轮廓相对人工轮廓偏移了多少（像素向量）")
    print("  这比全局平均更有意义：哪里的轮廓偏了，就影响哪个点。")
    print(f"{'id':>4} | " + " | ".join(f"{k:>16}" for k in "ABCD") + " |")
    print("-" * 84)
    disp_rows = []
    for i in ids:
        lm = REFERENCE[i]
        cp_pred = contour_points(load_mask("results", i))
        cp_gt = contour_points(load_mask("labels", i))
        disp = local_displacement(lm, cp_pred, cp_gt)
        s_b = sensitivity_table(lm, bohler_angle)
        s_g = sensitivity_table(lm, gissane_angle)
        # 线性折算：Δθ ≈ ∇θ · Δp
        db = sum(s_b[k][0] * disp[k][0] + s_b[k][1] * disp[k][1] for k in ("A", "B", "C"))
        dg = sum(s_g[k][0] * disp[k][0] + s_g[k][1] * disp[k][1] for k in ("A", "C", "D"))
        disp_rows.append(dict(id=i, delta_bohler=db, delta_gissane=dg,
                              **{f"disp_{k}": float(np.linalg.norm(disp[k])) for k in "ABCD"}))
        cells = " | ".join(
            f"({disp[k][0]:>+5.1f},{disp[k][1]:>+5.1f})" for k in "ABCD")
        print(f"{i:>4} | {cells} |")
    print("-" * 84)
    print(f"     线性折算出的角度扰动绝对值平均: Böhler {np.mean([abs(r['delta_bohler']) for r in disp_rows]):.2f}°  "
          f"Gissane {np.mean([abs(r['delta_gissane']) for r in disp_rows]):.2f}°")
    print("     ⇒ 这就是'分割差一点 → 角度差多少'的量化答案。")


    # ---------- 第四部分：自动提点 ----------
    print(f"\n【第四部分】从 {args.mask}/ 掩膜自动提点（实验性）")
    print(f"{'id':>4} | {'点':>3} {'参考(x,y)':>14} {'自动(x,y)':>14} {'误差(px)':>9}")
    print("-" * 60)
    auto_rows = []
    for i in ids:
        am, info = auto_landmarks(load_mask(args.mask, i))
        if am is None:
            print(f"{i:>4} | 提取失败")
            continue
        errs = {}
        for k in ("A", "B", "C", "D"):
            ref = np.array(REFERENCE[i][k], float)
            got = np.array(am[k], float)
            e = float(np.linalg.norm(ref - got))
            errs[k] = e
            flag = "" if e <= 6 else ("  ←偏大" if e <= 15 else "  ←失败")
            print(f"{i:>4} | {k:>3} ({ref[0]:>5.0f},{ref[1]:>5.0f}) ({got[0]:>5.0f},{got[1]:>5.0f}) {e:>9.1f}{flag}")
        auto_rows.append(dict(id=i, **{f"err_{k}": errs[k] for k in "ABCD"},
                              bohler_auto=bohler_angle(am), gissane_auto=gissane_angle(am),
                              bohler_ref=bohler_angle(REFERENCE[i]),
                              gissane_ref=gissane_angle(REFERENCE[i])))
        print("-" * 60)
    if auto_rows:
        print("     逐点平均误差（像素）: " + "  ".join(
            f"{k}={np.mean([r[f'err_{k}'] for r in auto_rows]):.1f}" for k in "ABCD"))
        print("     角度平均误差（度）  : "
              f"Böhler {np.mean([abs(r['bohler_auto']-r['bohler_ref']) for r in auto_rows]):.1f}  "
              f"Gissane {np.mean([abs(r['gissane_auto']-r['gissane_ref']) for r in auto_rows]):.1f}")

    # ---------- 存表 ----------
    csv_path = RESULT_DIR / f"{args.out}_{args.mask}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "bohler_printed", "bohler_ref", "gissane_printed", "gissane_ref",
                     "mean_ssd_px", "bohler_budget_deg", "gissane_budget_deg",
                     "grad_b_B", "grad_g_D",
                     "disp_A_px", "disp_B_px", "disp_C_px", "disp_D_px",
                     "delta_bohler_deg", "delta_gissane_deg"])
        for i in ids:
            lm = REFERENCE[i]
            r = next(x for x in budget_rows if x["id"] == i)
            dp = next(x for x in disp_rows if x["id"] == i)
            wr.writerow([i, PRINTED[i][0], round(bohler_angle(lm), 3),
                         PRINTED[i][1], round(gissane_angle(lm), 3),
                         round(r["ssd"], 3), round(r["bohler_budget"], 3),
                         round(r["gissane_budget"], 3),
                         round(r["grad_b_B"], 3), round(r["grad_g_D"], 3),
                         round(dp["disp_A"], 2), round(dp["disp_B"], 2),
                         round(dp["disp_C"], 2), round(dp["disp_D"], 2),
                         round(dp["delta_bohler"], 3), round(dp["delta_gissane"], 3)])
    print(f"\n表已存: {csv_path}")

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

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), dpi=150)
        x = np.arange(len(ids))

        axes[0].bar(x - 0.2, [sd_rows[i]["mean"] for i in ids], 0.4, label="mean SSD", color="#1a5fb4")
        axes[0].bar(x + 0.2, [sd_rows[i]["hd95"] for i in ids], 0.4, label="HD95", color="#c0dd97")
        axes[0].set_xticks(x); axes[0].set_xticklabels(ids, fontsize=8)
        axes[0].set_ylabel("像素"); axes[0].set_title("分割边界误差（预测 vs 人工）", fontsize=11)
        axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.25, linewidth=0.6)

        axes[1].bar(x - 0.2, [r["bohler_budget"] for r in budget_rows], 0.4,
                    label="Böhler", color="#1a5fb4")
        axes[1].bar(x + 0.2, [r["gissane_budget"] for r in budget_rows], 0.4,
                    label="Gissane", color="#c01c28")
        axes[1].set_xticks(x); axes[1].set_xticklabels(ids, fontsize=8)
        axes[1].set_ylabel("度"); axes[1].set_title("分割误差传导出的角度误差预算", fontsize=11)
        axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.25, linewidth=0.6)

        names = ["A 前突", "B 后关节面", "C 结节上缘", "D 跟骨沟"]
        axes[2].bar(np.arange(4) - 0.2, [np.mean([r[f"grad_b_{k}"] for r in budget_rows]) for k in "ABCD"],
                    0.4, label="对 Böhler", color="#1a5fb4")
        axes[2].bar(np.arange(4) + 0.2, [np.mean([r[f"grad_g_{k}"] for r in budget_rows]) for k in "ABCD"],
                    0.4, label="对 Gissane", color="#c01c28")
        axes[2].set_xticks(np.arange(4)); axes[2].set_xticklabels(names, fontsize=8)
        axes[2].set_ylabel("度/像素"); axes[2].set_title("各关键点的角度敏感度", fontsize=11)
        axes[2].legend(fontsize=8); axes[2].grid(axis="y", alpha=0.25, linewidth=0.6)

        for ax in axes:
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig_path = RESULT_DIR / f"{args.out}_{args.mask}.png"
        fig.savefig(fig_path)
        print(f"图已存: {fig_path}")
    except Exception as e:
        print("画图失败:", e)


if __name__ == "__main__":
    main()
