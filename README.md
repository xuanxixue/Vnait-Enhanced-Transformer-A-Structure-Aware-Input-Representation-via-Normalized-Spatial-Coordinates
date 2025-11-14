# Vnait-Enhanced Transformer: A Structure-Aware Input Representation via Normalized Spatial Coordinates

基于归一化空间坐标的结构感知输入表示：Vnait 增强型 Transformer

## 🎯 概述

传统 Transformer 依赖离散位置索引或固定正弦编码建模词序，但存在两大关键局限：
- **缺乏跨句长归一化能力**
- **结构连续性不足**

这导致在低资源场景下样本效率低下。Vnait 提出一种新型输入表示框架，以**可学习、归一化、单调的实值空间坐标**替代传统位置嵌入，实现更高效、更鲁棒、更具泛化能力的输入表示。

## 🚀 核心特性

| 特性 | 优势 |
|------|------|
| **参数高效** | 消除位置嵌入表（节省 $L_{\max} \times d$ 参数） |
| **数据高效** | 强结构先验加速低资源场景收敛 |
| **长度泛化** | 归一化 $S_i \in [0,100]$ 支持跨长度比较 |
| **可解释性** | $S_i$ 提供连续位置语义（如 $S_i=50$ ≈ 中心位置） |
| **兼容性** | 即插即用，兼容任何基于 Transformer 的架构 |

## 📐 核心约束

给定长度为 $n$ 的句子，为每个词元分配实值标量 $S_i \in \mathbb{R}$，满足：

$$
\boxed{
\begin{aligned}
&\text{(1) 单调性:} && S_1 < S_2 < \cdots < S_n \\
&\text{(2) 归一性:} && \sum_{i=1}^{n} S_i = 100 \\
&\text{(3) 连续性:} && S_i \in \mathbb{R} \quad \text{(支持小数、负数)}
\end{aligned}
}
$$

## 🛠️ 方法实现

### 1. 空间坐标可学习生成

引入无约束参数 $\mathbf{z} = [z_1, \dots, z_n] \in \mathbb{R}^n$：

$$
\begin{aligned}
\tilde{S}_i &= \sum_{k=1}^{i} \underbrace{\log(1 + e^{z_k})}_{\text{softplus}(z_k)} \quad \text{(保证严格递增)} \\
S_i &= 100 \cdot \frac{\tilde{S}_i}{\tilde{S}_n} \quad \text{(归一化总和为100)}
\end{aligned}
$$

### 2. 空间向量构建

通过多层感知机将标量 $S_i$ 映射为 $d$ 维空间向量：

$$
\begin{aligned}
\mathbf{u}_i^{(1)} &= \mathbf{W}^{(1)} S_i + \mathbf{b}^{(1)} \in \mathbb{R}^{d_1} \\
\mathbf{v}_i^{(1)} &= \sigma_1(\mathbf{u}_i^{(1)}) \\
&\vdots \\
\mathbf{s}_i &= \mathbf{W}^{(L)} \mathbf{v}_i^{(L-1)} + \mathbf{b}^{(L)} \in \mathbb{R}^{d}
\end{aligned}
$$

### 3. 语义向量提取

使用预训练 Word2Vec 嵌入：

$$
\mathbf{e}_i = \text{Word2Vec}(w_i)
$$

### 4. 显式空间-语义关系建模

计算三类关系特征：

$$
\begin{aligned}
\Delta_i &= \mathbf{e}_i - \mathbf{s}_i \quad \text{(差值向量)} \\
\rho_i &= \frac{\mathbf{e}_i^\top \mathbf{s}_i}{\|\mathbf{e}_i\| \cdot \|\mathbf{s}_i\|} \in [-1, 1] \quad \text{(相似值)} \\
\mathbf{a}_i &= \text{MLP}_{\text{rel}}\left( [\mathbf{e}_i; \mathbf{s}_i; \Delta_i; \rho_i \cdot \mathbf{1}_d] \right) \in \mathbb{R}^{d} \quad \text{(关联向量)}
\end{aligned}
$$

### 5. 最终输入表示

融合语义、空间与关系信息：

$$
\boxed{\mathbf{x}_i = \mathbf{e}_i + \mathbf{s}_i + \mathbf{a}_i}
$$

**关键创新**：此向量完全替代传统 Transformer 中的 $\text{Embed}(w_i) + \text{PosEmb}(i)$，无需任何位置嵌入表或正弦编码。

## 🔄 与标准 Transformer 集成

Vnait 仅修改输入层，主干架构完全兼容：

$$
\mathbf{H} = \text{TransformerEncoder}(\mathbf{X})
$$

其中 $\mathbf{X} = [\mathbf{x}_1, \dots, \mathbf{x}_n]$，$\text{TransformerEncoder}$ 包含标准组件：
- 多头自注意力 (MHSA)
- 位置前馈网络 (FFN) 
- 残差连接与层归一化

## 🎯 训练目标

总损失结合任务监督与结构正则：

$$
\mathcal{L} = \underbrace{\mathcal{L}_{\text{task}}}_{\text{下游任务}}
+ \gamma_1 \underbrace{\sum_{i=1}^{n-1} \text{Huber}\left( (S_{i+1} - S_i) - \frac{100}{n} \right)}_{\text{平滑正则}}
+ \gamma_2 \underbrace{\max(0, -S_1)^2 + \max(0, S_n - 100)^2}_{\text{边界约束}}
$$

## 📊 理论分析

### 参数效率
- 传统方法：需要 $L_{\max} \times d$ 的位置嵌入参数
- Vnait：仅需 MLP 参数，通常远少于位置嵌入表

### 长度泛化
- 归一化坐标 $S_i \in [0,100]$ 确保跨长度位置语义一致
- $S_i=50$ 始终代表"中间"位置，无论句子长度

### 兼容性
- 即插即用，兼容 BERT、GPT、T5 等架构
- 不修改 Transformer 核心代码

## 🎯 适用场景

特别适用于：
- 小样本学习
- 跨句长任务
- 低资源自然语言处理
- 需要强结构先验的应用

## 🔮 未来工作

1. **扩展到解码器架构**：为空间坐标添加因果约束
2. **与预训练模型结合**：集成到 LLM 并在低资源任务验证
3. **空间坐标可视化**：分析 $S_i$ 与语言结构的关系

---

> **注**：本方法在不改变 Transformer 主干结构的前提下，通过引入归一化空间坐标，实现了更高效、更鲁棒的结构感知输入表示。
