# 代码笔记

关键函数、参数、数据结构与原理。

## 1. U-Net（`unet.py`）

### `DoubleConv`

```python
(Conv2d 3×3, padding=1, bias=False) → BatchNorm2d → ReLU
(Conv2d 3×3, padding=1, bias=False) → BatchNorm2d → ReLU
```

`bias=False` 是因为后面紧跟 `BatchNorm`——BN 的 `beta` 已经承担了偏置，
再加一份是冗余参数。

### `UNet(in_ch=1, out_ch=1, base=16)`

| 阶段 | 结构 | 分辨率 | 通道（base=16） |
|---|---|---|---|
| Encoder | enc1 → pool → enc2 → pool → enc3 → pool → enc4 | 256 → 128 → 64 → 32 → 16 | 16 / 32 / 64 / 128 |
| Bottleneck | `DoubleConv(b*8, b*16)` | 8 | 256 |
| Decoder | up4 → cat(s4) → dec4 … up1 → cat(s1) → dec1 | 16 → 32 → 64 → 128 → 256 | 128 / 64 / 32 / 16 |
| Head | `Conv2d(b, out_ch, kernel_size=1)` | 256 | 1 |

参数量（实测）：

| base | 参数量 |
|---|---|
| 8 | 486,409 |
| **16（默认）** | **1,942,289** |
| 32 | 7,762,465 |

**两个关键约定**：

1. **输出是 logits，不在模型里加 sigmoid** —— 训练配 `BCEWithLogitsLoss`（数值稳定），
   预测时自己 `sigmoid > 0.5`
2. **skip 用 `torch.cat` 而不是相加** —— 拼接保留编码器的高分辨率特征通道，
   相加会把它们混在一起。边界精度直接决定角度精度，这里不能省

```python
x = self.dec4(torch.cat([self.up4(x), s4], dim=1))
```

## 2. 角度计算（`calcaneus_angles.py`）

### `angle_at_vertex(vertex, p, q)`

```python
u = p - vertex
v = q - vertex
cos = (u·v) / (|u|·|v|)
θ = arccos(clip(cos, -1, 1)) * 180 / π
```

`clip` 是必须的：浮点误差可能让 `cos` 略微超过 ±1，`arccos` 会返回 `nan`。

### `bohler_angle(lm)` / `gissane_angle(lm)`

| 函数 | 顶点 | 两臂 |
|---|---|---|
| `bohler_angle` | **A** | A→B、A→C |
| `gissane_angle` | D | D→A、D→C |

配套还有 `bohler_textbook` / `gissane_textbook` 两个"教科书口径"版本（Böhler 顶点用 B），
用来**证明口径差异有多大**——教科书口径算出来是 118°~168°，与本项目的 3.65°~19° 完全不同。

## 3. 距离变换（`edt_sq`）

**自己用 numpy 实现精确欧氏距离变换，不依赖 scipy**。
算法来自 Felzenszwalb & Huttenlocher 的**一维抛物线下包络**：

```python
f = np.where(binary, 0.0, INF)   # 前景距离 0，背景无穷
for y in range(H):  f[y]     = dt1d(f[y])     # 行扫
for x in range(W):  f[:, x]  = dt1d(f[:, x])  # 列扫
```

`dt1d` 维护三样东西：

| 变量 | 作用 |
|---|---|
| `v[]` | 每个抛物线的顶点位置 |
| `z[]` | 相邻抛物线的交点（分界） |
| `k` | 当前有效抛物线个数 |

每个新位置 `q` 的抛物线若把前面的完全遮住，就把前面的 `k` 弹出（`while s <= z[k]: k -= 1`），
最后按交点查表得到 `d[q] = (q - v[k])² + f[v[k]]`。

**复杂度 O(n)**，比暴力 O(n²) 快得多，且结果是**精确**的（不是近似）。

为什么值得自己写：`requirements.txt` 里没有 scipy 与 OpenCV，
整个项目只依赖 torch / numpy / Pillow / matplotlib。

### `surface_distance_stats(pred, gt)`

```python
d_pg = sqrt(edt_sq(cg))[cp]     # 预测轮廓每点 → 人工轮廓的距离
d_gp = sqrt(edt_sq(cp))[cg]     # 反向
both = concatenate([d_pg, d_gp])
```

返回 `mean` / `hd95`（95 百分位）/ `max`。

**为什么要双向**：单向只算"预测点离真值多远"，漏掉"真值上有点没被覆盖"的情况。
对称距离才能同时反映过分割与欠分割。

## 4. 敏感度（`angle_gradient`）

```python
def angle_gradient(angle_fn, lm, key, coord, h=0.5):
    # 中心差分：把某点的 x 或 y 挪 ±h，看角度变化
```

用**数值微分**而不是手推偏导，原因：角度公式含 `arccos`，
手推链式法则容易出错，而且一旦口径改了（顶点换人）公式就得重推。
中心差分只要 4 行，且精度足够（误差 O(h²)）。

`h = 0.5` 像素——取 1 像素会把结果平滑掉，取太小会被浮点噪声淹没。

## 5. 上边界与关键点（本项目的核心算法）

### `_upper_profile(mask, win=9)`

```python
for x in range(W):
    idx = np.nonzero(mask[:, x])[0]
    if len(idx):
        top[x] = idx.min()          # 每列最靠上的前景像素
```

三步处理：

| 步 | 做法 | 为什么 |
|---|---|---|
| 取列最小值 | `min(idx)` | 上边界 = 每列最上面的前景点 |
| 线性插值补洞 | `np.interp` | 掩膜可能有空洞，导致某列为 `nan` |
| 滑动平均 | `np.convolve` + `edge` 填充 | 抑制单像素锯齿，`win=9` |

`pad(mode="edge")` 很关键：不加的话边界会被 0 拉低，窗口越大边角越歪。

### `auto_landmarks(mask, win=9, spike_drop=15, slope_frac=0.35)`

三个参数都在函数签名里，便于改成自己的口径：

| 参数 | 默认 | 作用 |
|---|---|---|
| `win` | 9 | 上边界平滑窗口 |
| `spike_drop` | 15 | 判定"前端是尖突"的骤降阈值（像素） |
| `slope_frac` | 0.35 | 回退到陡升起点时，斜率的相对阈值 |

四个点的规则：

| 点 | 规则 | 实测误差 |
|---|---|---|
| **A** | 默认取第 0 列；**若 `ys[0] - ys[7] > 15`**（前面是根尖突），则找首个 `ys <= ys[0]-15` 的位置 | 6.9 px |
| **C** | `argmin(ys)` —— 上边界全局最高点 | 10.2 px |
| **D** | 在 `[0.2n, 0.8n]` 找最陡上升（`argmin(s)`），再向左回退到 `\|s\| >= 0.35*\|s_min\|` 的列 | **3.9 px（可靠）** |
| **B** | 在 `[iD+2, iC-1]` 之间找 `\|gradient(s)\|` 最大处（曲率最大） | **23.9 px（失败）** |

**A 为什么要判尖突**：跟骨前突在影像上可能呈现为一根细尖，
直接取最左列会落在尖端上，而不是解剖学意义上的"上缘最高点"。

**D 为什么要回退**：最陡上升处是"沟的斜坡中段"，而解剖上的跟骨沟顶点
在斜坡**起点**，所以要沿斜率回退到坡度降到峰值 35% 的位置。

**B 为什么失败**：无论用"曲率最大""陡升终点""后关节面段最高点"都还原不出来，
说明 B 点的定义**不完全由轮廓几何决定**（可能依赖阅片人对关节面边界的判断）。
这是本仓库最重要的已知缺口——**需要向数据提供方索取标注脚本，而不是继续调规则**。

## 6. 损失与训练

### `SoftDiceLoss`

```python
dice = (2 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)
loss = 1 - dice
```

`smooth`（拉普拉斯平滑）防止前景为空时分母为 0。

### `build_criterion(name)`

| name | 返回 |
|---|---|
| `bce` | `BCEWithLogitsLoss()` |
| `dice_bce` | `0.5 × BCE + 0.5 × SoftDice` |

混合损失的作用见 debug_log：**把训练崩溃的概率压下去**，
而不是"更准"（同种子对照里 BCE 换种子也能到 0.94~0.96）。

### `set_seed(seed=42)`

除了常规的 `torch.manual_seed` / `np.random.seed` / `random.seed`，
**默认开启 `torch.backends.cudnn.deterministic = True`**。

不开的话，cuDNN 会自动挑卷积算法，同一份数据两次运行可能走到不同实现上，
误差逐轮累积——实测同种子最好验证 Dice 能差 0.3 个点（详见 debug_log）。

## 7. 参数速查

| 参数 | 值 | 位置 |
|---|---|---|
| 输入尺寸 | 256×256 灰度 | dataset |
| `base` | 16（默认） | unet |
| 参数量（base=16） | 1,942,289 | — |
| 损失 | `BCEWithLogitsLoss` | train |
| 二值化阈值 | `sigmoid > 0.5` | predict |
| 上边界平滑窗口 | 9 | angles |
| 尖突判定阈值 | 15 px | angles |
| 回退斜率比 | 0.35 | angles |
| 中心差分步长 | 0.5 px | angles |
| 种子 | 42 | train / loo |
| 归一化 | 逐图 min-max | dataset |
| 留一折数 | 11 | loo |

## 8. 原理补充

### 为什么用余弦定理而不是斜率差

Böhler 角是"两条线的夹角"，可以用 `arctan` 斜率差算，
但当某条线接近竖直时斜率趋于无穷，数值不稳定。
向量夹角用 `arccos` 对所有方向一致，没有奇异点。

### 为什么归一化用逐图 min-max（已知局限）

逐图 min-max 会把不同片子的曝光差异**抹平**——一张偏暗的片子和一张偏亮的片子
归一化后看起来一样。好处是模型不学曝光，坏处是丢失了"骨密度"这类信息。
更稳妥的做法是固定窗宽窗位或分位数截断，已在 README 局限里列出。

### 为什么"角度复现"能到 0.34°，而"自动提点"做不到

复现用的是**逐像素反推的参考关键点**——即从已有结果图上把点找回来，
所以只剩"角度公式"这一项误差，自然很小（0.34°）。

自动提点要从轮廓**规则性地重新定位**这些点，而 B 点的定义不完全是几何的，
于是误差一下跳到 23.9 px。**这两件事必须分开说，不能混成一个"精度"。**
