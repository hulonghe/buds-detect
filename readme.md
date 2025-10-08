# Tea Buds Target Detection

| 卷积类型        | 🧠 作用机制 / 效果说明                                                         | 🧩 PyTorch 示例                                                                                                                     | 📐 输入 → 输出形状                        | ⚙️ 核心参数说明                                                                                                                          |
|-------------|------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| **标准卷积**    | 卷积核提取局部特征（边缘/纹理），输出通道混合全部输入通道，表达能力强但计算大。                               | `nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)`                                                                            | `[B, 3, 64, 64] → [B, 16, 64, 64]`  | `in_channels`：输入通道数（必须设）<br>`out_channels`：输出特征数（可调，影响表达维度）<br>`kernel_size`：感受野大小（常用 3）<br>`stride`：步长（控制下采样）<br>`padding`：是否保留尺寸 |
| **深度可分离卷积** | 分为两步：每通道独立卷积（depthwise）+ 通道融合（pointwise 1×1）。大幅降低 FLOPs 和参数量，适合轻量网络。   | `nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, groups=16), nn.Conv2d(16, 32, 1))`                                                 | `[B, 16, 64, 64] → [B, 32, 64, 64]` | `groups=Cin`：深度卷积（通道分组 = 输入通道）<br>`kernel_size=3`：提取空间信息<br>`1x1卷积`：调整输出通道（可调）                                                     |
| **空洞卷积**    | 将卷积核“拉伸”加空洞，扩大感受野，参数不变但能捕捉远程信息，常用于语义分割。                                | `nn.Conv2d(16, 32, kernel_size=3, padding=2, dilation=2)`                                                                         | `[B, 16, 64, 64] → [B, 32, 64, 64]` | `dilation`：空洞率（越大感受野越大）<br>`padding`：应设为 `dilation` 对齐特征边界<br>其余参数同标准卷积（可调）                                                        |
| **组卷积**     | 将输入通道划分为 `groups` 组，各组独立卷积，降低通道间耦合，节省计算，提升效率。                          | `nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=4)`                                                                           | `[B, 32, 64, 64] → [B, 32, 64, 64]` | `groups`：决定每组多少通道独立卷积（`groups=1`是标准卷积）<br>必须满足：`in_channels % groups == 0`<br>可调但需满足 shape 一致性                                     |
| **可变形卷积**   | 学习卷积核偏移量 offset，使其采样更灵活，能适应目标形变、旋转等复杂结构。                               | `DeformConv2d(16, 32, kernel_size=3, padding=1)`（需安装 mmcv 或 detectron2 扩展）                                                        | `[B, 16, 64, 64] → [B, 32, 64, 64]` | `offset` 由额外分支生成（自动学习）<br>`kernel_size/padding` 同标准卷积<br>Deformable 版本（v1/v2）还可学习 modulation scalar（可调）                            |
| **1×1 卷积**  | 只作用于通道维度，不改变空间维度。用于升降维、通道压缩、非线性转换、残差融合。                                | `nn.Conv2d(64, 16, kernel_size=1)`                                                                                                | `[B, 64, 64, 64] → [B, 16, 64, 64]` | `in_channels/out_channels` 可调控制通道压缩/扩展比<br>`kernel_size=1` 固定（不需 padding）                                                          |
| **转置卷积**    | 用于上采样，反卷积操作（非线性可学习）。可恢复空间尺寸，但易出现“棋盘效应”（需注意 kernel/stride 对齐）。          | `nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)`                                                                             | `[B, 32, 32, 32] → [B, 16, 64, 64]` | `stride` 控制上采样倍数<br>`kernel_size` 应与 `stride` 对齐（通常等于 stride）<br>`padding/output_padding` 调节输出尺寸精度                                 |
| **卷积注意力模块** | 加入全局或局部注意力机制，提升显著区域响应，常配合 1×1 卷积控制通道权重（如 SE、CBAM）。不会改变特征尺寸，仅增强关键通道或区域。 | `SE = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(64, 16, 1), nn.ReLU(), nn.Conv2d(16, 64, 1), nn.Sigmoid())`<br>`x * SE(x)` | `[B, 64, 64, 64] → [B, 64, 64, 64]` | `hidden_dim`（瓶颈通道数）控制压缩比<br>是否使用空间/通道注意力可调<br>输出与输入尺寸一致，仅改变特征激活分布                                                                  |

-------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------

| 类型              | 🧠 作用机制 / 效果说明                                                     | ⚙️ 关键特性               | 🧩 PyTorch 示例                                 | 📐 输入 → 输出形状                    | 🎛️ 可训练参数 |
|-----------------|--------------------------------------------------------------------|-----------------------|-----------------------------------------------|---------------------------------|-----------|
| **ReLU**        | 将所有负值置 0，保持正值不变。<br>✔ 引入非线性，防止梯度消失，训练稳定。                           | 非线性、单边抑制              | `nn.ReLU()`<br>`F.relu(x)`                    | `[B, C, H, W] → [B, C, H, W]`   | ❌ 无       |
| **LeakyReLU**   | 改进版 ReLU，允许负值通过较小斜率泄漏。<br>✔ 缓解 ReLU 死亡问题。                          | 负区间线性泄漏（如 slope=0.01） | `nn.LeakyReLU(0.01)`                          | `[B, C, H, W] → [B, C, H, W]`   | ❌ 无       |
| **Sigmoid**     | 将值映射到 `[0, 1]`，常用于概率输出或门控机制。<br>✔ 易饱和，梯度小，不适合深层结构。                 | 输出压缩、平滑               | `nn.Sigmoid()`<br>`torch.sigmoid(x)`          | `[B, C, H, W] → [B, C, H, W]`   | ❌ 无       |
| **Tanh**        | 输出范围 `[-1, 1]`，中心化的 Sigmoid。<br>✔ 用于需要负值信息的场景，如 RNN。               | 有符号压缩                 | `nn.Tanh()`<br>`torch.tanh(x)`                | `[B, C, H, W] → [B, C, H, W]`   | ❌ 无       |
| **Softmax**     | 归一化为概率分布，常用于分类任务最后一层。                                              | 输出所有值和为 1（按 dim 归一）   | `nn.Softmax(dim=1)`                           | `[B, C] → [B, C]`               | ❌ 无       |
| **BatchNorm2d** | 对每个通道做归一化（均值为 0，方差为 1），然后线性变换（可训练 γ, β）。<br>✔ 加速收敛、缓解梯度爆炸。常用于卷积之后。 | 通道维度归一化、支持动态 batch    | `nn.BatchNorm2d(num_features=64)`             | `[B, 64, H, W] → [B, 64, H, W]` | ✅ 有（γ, β） |
| **LayerNorm**   | 对每个样本的所有通道 + 空间归一化，常用于 NLP 或 transformer。                          | 输入维度灵活，跨通道归一          | `nn.LayerNorm([C, H, W])`                     | `[B, C, H, W] → [B, C, H, W]`   | ✅ 有（γ, β） |
| **GroupNorm**   | 将通道分成 G 组做归一化，适用于 batch size 很小时。<br>✔ 替代 BatchNorm，更稳定。           | 无 batch 依赖，可跨设备稳定运行   | `nn.GroupNorm(num_groups=8, num_channels=64)` | `[B, 64, H, W] → [B, 64, H, W]` | ✅ 有（γ, β） |
| **Dropout**     | 随机将部分神经元置零，防止过拟合。<br>✔ 训练时启用，测试时关闭。                                | 随机抑制、提升泛化能力           | `nn.Dropout(p=0.5)`                           | `[B, D] → [B, D]`               | ❌ 无       |

# Tea Bud Image Datasets

This repository provides three datasets — **A**, **B**, and **C** — designed for research on tea bud detection,
classification.  
Each dataset contains high-resolution images of tea buds captured under different lighting and environmental conditions,
suitable for training and evaluating computer vision models.

## Dataset Overview

- **Dataset A**: The most complex, consisting of raw images directly captured from real-world environments without special processing, containing a large number of small objects.
- **Dataset B**: Moderately difficult, with most objects being of medium size and some small objects present.
- **Dataset C**: The simplest, primarily containing medium to large objects, which are relatively easy to recognize.

## Access

You can download the datasets from the following link:

🔗 **Download Link1（A，B，C）:** [Baidu Cloud](https://pan.baidu.com/s/1UgKeorPjWE6YxlakTMkOow)  
🔑 **Extraction Code:** `r8ya`

🔗 **Download Link2（B）:** [Roboflow](https://universe.roboflow.com/ycy-9asp0/tearob)

🔗 **Download Link3（C）:** [Roboflow](https://universe.roboflow.com/project-n7lcw/object-detection-for-tea-bud)

# Environment and Usage Instructions

This section describes the recommended system environment, software dependencies, and usage workflow for training and
evaluating tea bud recognition models.

## 1. System Requirements

- **Operating System:** Windows 10/11, Ubuntu 20.04+
- **Hardware Configuration:**
    - GPU: NVIDIA RTX 5070 or higher (≥12 GB VRAM recommended)
    - CPU: Intel i7 / AMD Ryzen 7 or better
    - RAM: ≥32 GB
    - Storage: ≥100 GB of free disk space

## 2. Software Environment

- **Programming Language:** Python 3.10
- **Deep Learning Framework:** PyTorch = 2.9.0
- **Essential Libraries:**
    - OpenCV = 4.12.0.88
    - albumentations ≥ 2.0.8
    - NumPy
    - Pandas
    - Matplotlib
- **Annotation Tools (optional):** LabelImg, X-AnyLabeling

## 3. Installation and Setup

```bash
# Clone the project repository
git clone https://github.com/hulonghe/buds-detect.git
cd buds-detect

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # (Linux/macOS)
venv\Scripts\activate      # (Windows)

# Install dependencies
pip install -r requirements.txt
```

## 4. Training and Evaluation Workflow

Download and extract the tea bud datasets (A, B, C). Start training:

```bash
python train_reg.py
```

Evaluate the trained model:

```bash
python validaate_reg.py
```

Training logs, metrics (accuracy, precision, recall, mAP), and model checkpoints will be saved in the ./runs directory.

## 5. Recommendations

Adjust batch size and learning rate according to GPU memory limits.
Use mixed precision training (torch.amp) to speed up convergence.

# Usage

Please cite this dataset appropriately if you use it in your research or publications.  
For academic and non-commercial use only.

# Acknowledgment and Contact

Thank you for using this dataset and environment guide.  
If you have any questions, suggestions, or collaboration requests, please feel free to contact us:

- **Author:** Wuyan  
- **Email:** [hlhcmp@gmail.com]

We appreciate your feedback and contribution to improving tea bud recognition research.