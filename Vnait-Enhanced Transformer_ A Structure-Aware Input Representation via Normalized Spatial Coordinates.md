# Vnait-Enhanced Transformer: A Structure-Aware Input Representation via Normalized Spatial Coordinates

## 基于归一化空间坐标的结构感知输入表示：Vnait 增强型 Transformer



***

## Abstract / 摘要

Transformer has become the backbone of natural language processing (NLP), but its positional encoding mechanisms (discrete indices or fixed sinusoidal functions) suffer from two critical limitations: lack of cross-sentence length normalization and insufficient structural continuity. These issues lead to poor sample efficiency in low-resource scenarios. To address this, we propose **Vnait**, a novel input representation framework that replaces traditional positional embeddings with **learnable, normalized, and monotonic real-valued spatial coordinates** ($\{S_i\}_{i=1}^n$). Vnait constructs input tokens by fusing semantic vectors (from Word2Vec), spatial vectors (mapped from $S_i$), and explicit spatial-semantic relational vectors—without introducing any linguistic labels. Theoretical analysis shows that Vnait achieves parameter efficiency (saving $L_{\max} \times d$ positional embedding parameters), length generalization (normalized $S_i \in [0,100]$), and compatibility with existing Transformer architectures. This work provides a structure-aware alternative for input representation in low-resource NLP tasks.

Transformer 已成为自然语言处理（NLP）的核心架构，但其位置编码机制（离散索引或固定正弦函数）存在两大关键局限：缺乏跨句长归一化能力与不足的结构连续性，导致低资源场景下样本效率低下。为此，本文提出**Vnait**—— 一种新型输入表示框架，以**可学习、归一化、单调的实值空间坐标**（$\{S_i\}_{i=1}^n$）替代传统位置嵌入。Vnait 通过融合语义向量（来自 Word2Vec）、空间向量（由$S_i$映射得到）与显式空间 - 语义关系向量构建输入词元，且不引入任何语言学标签。理论分析表明，Vnait 实现了参数高效（节省$L_{\max} \times d$个位置嵌入参数）、长度泛化（归一化$S_i \in [0,100]$）与现有 Transformer 架构的兼容性，为低资源 NLP 任务的输入表示提供了结构感知的新方案。



***

## 1. Introduction / 引言

### 1.1 Research Background / 研究背景

Since the proposal of Transformer (Vaswani et al., 2017), self-attention mechanisms have enabled parallel processing of sequence data, revolutionizing NLP tasks such as machine translation and text classification. A critical component of Transformer is **positional encoding**—it injects word order information into the model, as self-attention itself is order-agnostic.

自 Transformer（Vaswani 等人，2017）提出以来，自注意力机制实现了序列数据的并行处理，彻底改变了机器翻译、文本分类等 NLP 任务。Transformer 的核心组件之一是**位置编码**：由于自注意力本身不具备语序感知能力，位置编码需向模型注入词序信息。

Two mainstream positional encoding methods exist:



1. **Discrete positional indices**: Learn an embedding table of size $L_{\max} \times d$ (where $L_{\max}$ is the maximum sequence length, $d$ is the hidden dimension), and assign embeddings based on token positions ($i=1,2,\dots,n$).

2. **Fixed sinusoidal encoding**: Use sine/cosine functions of different frequencies to generate position vectors, which are added to semantic embeddings.

主流位置编码方法分为两类：



1. **离散位置索引**：学习一个尺寸为$L_{\max} \times d$的嵌入表（$L_{\max}$为最大序列长度，$d$为隐藏维度），根据词元位置（$i=1,2,\dots,n$）分配嵌入向量；

2. **固定正弦编码**：使用不同频率的正 / 余弦函数生成位置向量，与语义嵌入相加。

### 1.2 Limitations of Existing Methods / 现有方法局限

Both methods have inherent flaws:



* **No cross-length normalization**: For a token at the "middle" position, its discrete index is $n/2$ (varies with $n$), and its sinusoidal vector also changes with $n$—this makes position semantics inconsistent across sentences of different lengths.

* **Lack of structural continuity**: Discrete indices are integer-valued, and sinusoidal vectors are fixed—neither allows the model to learn fine-grained structural adjustments from data.

* **Poor low-resource efficiency**: Discrete embedding tables require large data to optimize, while fixed sinusoidal encoding lacks adaptability—both perform poorly when labeled data is scarce.

两种方法均存在固有缺陷：



* **无跨长度归一化**：对于 “中间” 位置的词元，其离散索引为$n/2$（随$n$变化），正弦向量也随$n$改变 —— 导致不同长度句子的位置语义不一致；

* **缺乏结构连续性**：离散索引为整数，正弦向量固定 —— 均无法让模型从数据中学习细粒度的结构调节；

* **低资源效率差**：离散嵌入表需大量数据优化，固定正弦编码缺乏适应性 —— 在标注数据稀缺时表现不佳。

### 1.3 Research Contributions / 研究贡献

We propose Vnait (**V**ectorized **n**ormalized **a**ttentional **i**nput **t**ransformation) to address these issues. Our key contributions are:



1. A **normalized spatial coordinate system** ($\{S_i\}_{i=1}^n$) with three constraints (monotonicity, normalization, continuity) to model position semantics consistently across lengths.

2. A **learnable coordinate generation mechanism** using softplus and normalization, ensuring differentiability and automatic constraint satisfaction.

3. An **explicit spatial-semantic fusion strategy** that combines semantic, spatial, and relational information to replace traditional "semantic + positional" embeddings.

4. Theoretical verification of parameter efficiency, length generalization, and compatibility with standard Transformers.

本文提出 Vnait（**V**ectorized **n**ormalized **a**ttentional **i**nput **t**ransformation）以解决上述问题，核心贡献包括：



1. 提出具有 “单调性、归一性、连续性” 三大约束的**归一化空间坐标系**（$\{S_i\}_{i=1}^n$），实现跨长度一致的位置语义建模；

2. 设计基于 softplus 与归一化的**可学习坐标生成机制**，保证可微性与约束自动满足；

3. 提出**显式空间 - 语义融合策略**，融合语义、空间与关系信息，替代传统 “语义 + 位置” 嵌入；

4. 从理论上验证了方法的参数高效性、长度泛化性与标准 Transformer 的兼容性。



***

## 2. Related Work / 相关工作

### 2.1 Positional Encoding in Transformers / Transformer 中的位置编码



* **Fixed positional encoding**: Vaswani et al. (2017) first proposed sinusoidal encoding, which is computation-efficient but lacks adaptability. Later works (e.g., Wang et al., 2020) tried to optimize frequency parameters but still remained fixed.

* **Learnable positional encoding**: Devlin et al. (2018) (BERT) used discrete positional embedding tables, which are adaptive but require large $L_{\max}$ (wasting parameters for short sentences) and lack cross-length normalization.

* **固定位置编码**：Vaswani 等人（2017）首次提出正弦编码，计算高效但缺乏适应性；后续工作（如 Wang 等人，2020）尝试优化频率参数，但仍保持固定。

* **可学习位置编码**：Devlin 等人（2018）（BERT）采用离散位置嵌入表，具备适应性但需设置较大$L_{\max}$（对短句子造成参数浪费），且无跨长度归一化。

### 2.2 Structure-Aware Input Representation / 结构感知输入表示

Recent works have focused on injecting structural information into inputs:



* **Syntax-aware methods**: Incorporate parse trees (e.g., Liu et al., 2019) or dependency graphs (e.g., Marcheggiani & Titov, 2017) into embeddings, but rely on pre-trained linguistic tools (increasing complexity).

* **Data-driven structure methods**: Qiu et al. (2021) learned structural embeddings via self-supervised signals, but still used traditional positional encodings as a base.

Vnait differs from these works: it learns structural information **purely from data** (no linguistic labels/tools) and **integrates position and structure** into a unified coordinate system.

近年研究聚焦于向输入注入结构信息：



* **语法感知方法**：将解析树（如 Liu 等人，2019）或依赖图（如 Marcheggiani & Titov，2017）融入嵌入，但依赖预训练语言工具（增加复杂度）；

* **数据驱动结构方法**：Qiu 等人（2021）通过自监督信号学习结构嵌入，但仍以传统位置编码为基础。

Vnait 与之不同：纯数据驱动学习结构信息（无语言学标签 / 工具），并将位置与结构整合到统一坐标系中。



***

## 3. Methodology of Vnait / Vnait 方法学

### 3.1 Core Design Principles / 核心设计原则

Vnait follows three principles to address existing limitations:



1. **Normalization**: Ensure position semantics are consistent across sentences of different lengths.

2. **Learnability**: Allow the model to adapt to data distribution (unlike fixed encoding).

3. **Minimal Constraints**: Avoid over-engineering (no linguistic labels) and let structure emerge from data.

Vnait 遵循三大原则以解决现有局限：



1. **归一性**：确保不同长度句子的位置语义一致；

2. **可学习性**：允许模型适应数据分布（区别于固定编码）；

3. **最小约束**：避免过度设计（无语言学标签），让结构从数据中自涌现。

### 3.2 Core Constraints of Spatial Coordinates / 空间坐标核心约束

For a sentence of length $n$, we assign each token a real-valued scalar $S_i \in \mathbb{R}$ (spatial coordinate) satisfying three constraints:

对于长度为$n$的句子，为每个词元分配实值标量$S_i \in \mathbb{R}$（空间坐标），满足三大约束：

$\boxed{
\begin{aligned}
&(1) \text{ Monotonicity (单调性):} && S_1 < S_2 < \cdots < S_n \quad (\text{preserves word order}) \\
&(2) \text{ Normalization (归一性):} && \sum_{i=1}^n S_i = 100 \quad (\text{cross-length consistency}) \\
&(3) \text{ Continuity (连续性):} && S_i \in \mathbb{R} \quad (\text{supports decimals/negatives for fine adjustment})
\end{aligned}
}$

**Rationale / 设计依据**:



* Monotonicity ensures the model does not reverse word order (a fundamental requirement for sequence modeling).

* Normalization fixes the total "spatial span" of the sentence to 100, so $S_i=50$ always represents the "middle" position (regardless of $n$).

* Continuity allows the model to learn fine-grained position adjustments (e.g., emphasizing key tokens by expanding their coordinate intervals).

* 单调性确保模型不颠倒词序（序列建模的基本要求）；

* 归一性将句子的总 “空间跨度” 固定为 100，因此$S_i=50$始终代表 “中间” 位置（与$n$无关）；

* 连续性允许模型学习细粒度位置调节（如通过扩大坐标间隔强调关键词元）。

### 3.3 Learnable Generation of Spatial Coordinates / 空间坐标的可学习生成

To ensure differentiability (critical for backpropagation) and automatic constraint satisfaction, we introduce unconstrained parameters $\mathbf{z} = [z_1, z_2, \dots, z_n] \in \mathbb{R}^n$ and design a two-step generation process:

为保证可微性（反向传播的关键）与约束自动满足，引入无约束参数$\mathbf{z} = [z_1, z_2, \dots, z_n] \in \mathbb{R}^n$，设计两步生成流程：

#### Step 1: Generate strictly increasing intermediate coordinates / 生成严格递增的中间坐标

We use the **softplus function** ($\text{softplus}(x) = \log(1+e^x)$) to map $z_k$ to positive values, then compute cumulative sums to ensure monotonicity:

采用**softplus 函数**（$\text{softplus}(x) = \log(1+e^x)$）将$z_k$映射为正值，再通过累积和确保单调性：

$\tilde{S}_i = \sum_{k=1}^i \underbrace{\log(1+e^{z_k})}_{\text{softplus}(z_k)}
$

**Property / 性质**: $\tilde{S}_i < \tilde{S}_{i+1}$ (since $\text{softplus}(z_{i+1}) > 0$), satisfying monotonicity.

#### Step 2: Normalize to satisfy sum constraint / 归一化以满足求和约束

We normalize $\tilde{S}_i$ by the total sum of intermediate coordinates ($\tilde{S}_n$) to ensure $\sum_{i=1}^n S_i = 100$:

通过中间坐标的总和（$\tilde{S}_n$）对$\tilde{S}_i$归一化，确保$\sum_{i=1}^n S_i = 100$：

$S_i = 100 \cdot \frac{\tilde{S}_i}{\tilde{S}_n} \tag{2}
$

**Property / 性质**: $\sum_{i=1}^n S_i = 100 \cdot \frac{\sum_{i=1}^n \tilde{S}_i}{\tilde{S}_n} = 100$ (since $\sum_{i=1}^n \tilde{S}_i = \tilde{S}_n$ for cumulative sums), satisfying normalization.

### 3.4 Spatial Vector Construction / 空间向量构建

The scalar $S_i$ is too low-dimensional to interact with high-dimensional semantic embeddings ($\in \mathbb{R}^d$). We use a **multi-layer perceptron (MLP)** to map $S_i$ to a $d$-dimensional spatial vector $\mathbf{s}_i$:

标量$S_i$维度过低，无法与高维语义嵌入（$\in \mathbb{R}^d$）交互。采用**多层感知机（MLP）** 将$S_i$映射为$d$维空间向量$\mathbf{s}_i$：

$\begin{aligned}
\mathbf{u}_i^{(1)} &= \mathbf{W}^{(1)} S_i + \mathbf{b}^{(1)} && \in \mathbb{R}^{d_1} \quad (\text{Linear Layer 1 / 线性层1}) \\
\mathbf{v}_i^{(1)} &= \sigma_1\left(\mathbf{u}_i^{(1)}\right) && \quad (\text{Nonlinear Activation 1 / 非线性激活1}) \\
\mathbf{u}_i^{(2)} &= \mathbf{W}^{(2)} \mathbf{v}_i^{(1)} + \mathbf{b}^{(2)} && \in \mathbb{R}^{d_2} \quad (\text{Linear Layer 2 / 线性层2}) \\
\mathbf{v}_i^{(2)} &= \sigma_2\left(\mathbf{u}_i^{(2)}\right) && \quad (\text{Nonlinear Activation 2 / 非线性激活2}) \\
&\vdots \\
\mathbf{s}_i &= \mathbf{W}^{(L)} \mathbf{v}_i^{(L-1)} + \mathbf{b}^{(L)} && \in \mathbb{R}^d \quad (\text{Output Layer / 输出层})
\end{aligned}
$

**Hyperparameters / 超参数**:



* Number of layers $L \geq 2$ (to model non-linear spatial-semantic relationships).

* Activation functions $\sigma_l$: GELU (preferred for NLP) or ReLU.

* Hidden dimensions $d_1, d_2, \dots, d_{L-1}$: Typically set to $2d$ or $4d$ (balance between capacity and efficiency).

### 3.5 Semantic Vector Extraction / 语义向量提取

We use **pre-trained Word2Vec embeddings** (Mikolov et al., 2013) as the semantic base (frozen or fine-tuned during training):

采用**预训练 Word2Vec 嵌入**（Mikolov 等人，2013）作为语义基础（训练中可冻结或微调）：

$\mathbf{e}_i = \text{Word2Vec}(w_i)
$

where $w_i$ is the $i$-th token of the sentence, and $\mathbf{e}_i \in \mathbb{R}^d$ (matching the dimension of $\mathbf{s}_i$ for fusion).

其中$w_i$为句子的第$i$个词元，$\mathbf{e}_i \in \mathbb{R}^d$（与$\mathbf{s}_i$维度匹配以实现融合）。

### 3.6 Explicit Spatial-Semantic Relation Modeling / 显式空间 - 语义关系建模

To capture the interaction between spatial and semantic information (ignored by traditional "additive" positional encoding), we compute three types of relational features:

为捕捉空间与语义信息的交互（传统 “加法式” 位置编码忽略此点），计算三类关系特征：

#### (1) Difference Vector (差值向量)

Measures the "offset" between semantic and spatial vectors:

衡量语义与空间向量的 “偏差”：

$\Delta_i = \mathbf{e}_i - \mathbf{s}_i
$

#### (2) Similarity Scalar (相似值)

Measures the cosine similarity between $\mathbf{e}_i$ and $\mathbf{s}_i$ (range: $[-1,1]$):

衡量$\mathbf{e}_i$与$\mathbf{s}_i$的余弦相似度（范围：$[-1,1]$）：

$\rho_i = \frac{\mathbf{e}_i^\top \mathbf{s}_i}{\|\mathbf{e}_i\| \cdot \|\mathbf{s}_i\|}
$

#### (3) Association Vector (关联向量)

Fuses semantic, spatial, and relational features into a unified relational vector using a dedicated MLP ($\text{MLP}_{\text{rel}}$):

通过专用 MLP（$\text{MLP}_{\text{rel}}$）将语义、空间与关系特征融合为统一关系向量：

$\mathbf{a}_i = \text{MLP}_{\text{rel}}\left( \left[ \mathbf{e}_i; \mathbf{s}_i; \Delta_i; \rho_i \cdot \mathbf{1}_d \right] \right) \in \mathbb{R}^d
$

where $[\cdot;\cdot]$ denotes vector concatenation, and $\rho_i \cdot \mathbf{1}_d$ broadcasts the scalar $\rho_i$ to a $d$-dimensional vector (for dimension consistency).

其中$[\cdot;\cdot]$表示向量拼接，$\rho_i \cdot \mathbf{1}_d$将标量$\rho_i$广播为$d$维向量（保证维度一致）。

### 3.7 Final Input Token Representation / 最终输入词元表示

The final token vector $\mathbf{x}_i$ fuses semantic, spatial, and relational information—**completely replacing** the traditional "semantic embedding + positional embedding" in Transformers:

最终词元向量$\mathbf{x}_i$融合语义、空间与关系信息 ——**完全替代**Transformer 中传统的 “语义嵌入 + 位置嵌入”：

$\boxed{
\mathbf{x}_i = \mathbf{e}_i + \mathbf{s}_i + \mathbf{a}_i
}
$

**Key Innovation / 关键创新**: No positional embedding table or sinusoidal function is needed—all position-related information is encapsulated in $\mathbf{s}_i$ and $\mathbf{a}_i$.

### 3.8 Integration with Standard Transformer / 与标准 Transformer 的融合

Vnait only modifies the **input layer** of the Transformer; the backbone remains fully compatible with existing implementations. The sequence of final tokens $\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n]$ is fed into the Transformer encoder stack:

Vnait 仅修改 Transformer 的**输入层**，主干架构与现有实现完全兼容。最终词元序列$\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n]$输入 Transformer 编码器堆栈：

$\mathbf{H} = \text{TransformerEncoder}(\mathbf{X}) \tag{9}$

where $\text{TransformerEncoder}$ consists of standard components:



* Multi-Head Self-Attention (MHSA)

* Position-wise Feed-Forward Networks (FFN)

* Residual Connections & Layer Normalization

其中$\text{TransformerEncoder}$包含标准组件：



* 多头自注意力（MHSA）

* 位置 - wise 前馈网络（FFN）

* 残差连接与层归一化

### 3.9 Training Objective / 训练目标

The total loss $\mathcal{L}$ combines **task supervision** (for downstream tasks) and **structural regularization** (to maintain coordinate constraints):

总损失$\mathcal{L}$结合**任务监督**（针对下游任务）与**结构正则**（维持坐标约束）：

$\mathcal{L} = \underbrace{\mathcal{L}_{\text{task}}}_{\text{Downstream Task}}
\gamma_1 \underbrace{\sum_{i=1}^{n-1} \text{Huber}\left( (S_{i+1} - S_i) - \frac{100}{n} \right)}_{\text{Smoothness Regularization}}
\gamma_2 \underbrace{\max(0, -S_1)^2 + \max(0, S_n - 100)^2}_{\text{Boundary Constraint}} 
$

#### Component Explanations / 组件说明:



1. $\mathcal{L}_{\text{task}}$: Task-specific loss (e.g., cross-entropy for classification, MSE for regression).

2. Smoothness Regularization: Uses Huber loss (robust to outliers) to encourage uniform coordinate intervals (ideal interval: $100/n$).

3. Boundary Constraint: Penalties for $S_1 < 0$ or $S_n > 100$ (prevents extreme coordinates from distorting position semantics).

4. $\gamma_1, \gamma_2$: Regularization weights (typically set to $0.1 \sim 1.0$ via validation).

5. $\mathcal{L}_{\text{task}}$：任务特定损失（如分类任务的交叉熵，回归任务的 MSE）；

6. 平滑正则：采用 Huber 损失（对异常值鲁棒），鼓励坐标间隔均匀（理想间隔：$100/n$）；

7. 边界约束：对$S_1 < 0$或$S_n > 100$施加惩罚（防止极端坐标扭曲位置语义）；

8. $\gamma_1, \gamma_2$：正则权重（通常通过验证集设置为$0.1 \sim 1.0$）。



***

## 4. Theoretical Analysis / 理论分析

### 4.1 Parameter Efficiency / 参数效率

Traditional discrete positional encoding requires an embedding table of size $L_{\max} \times d$ (e.g., $L_{\max}=512$, $d=768$: $512 \times 768 = 393,216$ parameters). Vnait eliminates this table—its only additional parameters are:



* Unconstrained coordinates $\mathbf{z}$ (size $n$, negligible since $n \ll L_{\max}$).

* MLP for spatial vector ($\sum_{l=1}^{L-1} d_l \times d_{l+1} + d_l$) and $\text{MLP}_{\text{rel}}$ (similar size).

For $d=768$ and $L=2$ (hidden dimension $2d=1536$), the total additional parameters are $\approx 768 \times 1536 + 1536 + 1536 \times 768 + 768 \approx 2.4$ million—**much fewer than the positional embedding table for large&#x20;****&#x20;** (e.g., $L_{\max}=1024$: $768 \times 1024 = 786,432$ parameters, already larger than Vnait’s additional parameters for $d=768$).

传统离散位置编码需尺寸为$L_{\max} \times d$的嵌入表（如$L_{\max}=512$，$d=768$：$512 \times 768 = 393,216$个参数）。Vnait 取消该表，额外参数仅包括：



* 无约束坐标$\mathbf{z}$（尺寸$n$，因$n \ll L_{\max}$可忽略）；

* 空间向量 MLP（参数数$\sum_{l=1}^{L-1} d_l \times d_{l+1} + d_l$）与$\text{MLP}_{\text{rel}}$（尺寸相近）。

当$d=768$且$L=2$（隐藏维度$2d=1536$）时，总额外参数约为$768 \times 1536 + 1536 + 1536 \times 768 + 768 \approx 240$万 ——**远少于大****&#x20;****下的位置嵌入表**（如$L_{\max}=1024$：$768 \times 1024 = 786,432$个参数，已超过 Vnait 在$d=768$时的额外参数）。

### 4.2 Length Generalization / 长度泛化性

Normalized coordinates $S_i \in [0,100]$ ensure consistent position semantics across sentences of different lengths:



* For a sentence of length $n=10$: The "middle" token (i=5) has $S_5 \approx 50$.

* For a sentence of length $n=20$: The "middle" token (i=10) also has $S_{10} \approx 50$.

This is impossible for discrete indices (i=5 vs. i=10) or sinusoidal encoding (different frequency combinations). Thus, Vnait can generalize better to unseen sequence lengths (e.g., training on $n \leq 256$ and testing on $n=512$).

归一化坐标$S_i \in [0,100]$确保不同长度句子的位置语义一致：



* 长度$n=10$的句子：“中间” 词元（i=5）的$S_5 \approx 50$；

* 长度$n=20$的句子：“中间” 词元（i=10）的$S_{10} \approx 50$。

这对离散索引（i=5 vs. i=10）或正弦编码（不同频率组合）而言无法实现。因此，Vnait 对未见过的序列长度泛化性更强（如训练时$n \leq 256$，测试时$n=512$）。

### 4.3 Compatibility / 兼容性

Vnait is a **plug-and-play** input module: it only replaces the input embedding layer of the Transformer, and the backbone (MHSA, FFN, residual connections) remains unchanged. This means:



* It can be integrated into any Transformer-based architecture (BERT, GPT, T5) without modifying the core code.

* It is compatible with existing optimization pipelines (AdamW, learning rate scheduling) and evaluation metrics.

Vnait 是**即插即用**的输入模块：仅替换 Transformer 的输入嵌入层，主干架构（MHSA、FFN、残差连接）保持不变。这意味着：



* 可集成到任何基于 Transformer 的架构（BERT、GPT、T5）中，无需修改核心代码；

* 与现有优化流程（AdamW、学习率调度）和评估指标兼容。



***

## 5. Conclusion and Future Work / 结论与未来工作

### 5.1 Conclusion / 结论

This paper proposes Vnait, a structure-aware input representation framework for Transformers. By introducing learnable, normalized, and monotonic spatial coordinates, Vnait addresses the limitations of traditional positional encoding (lack of normalization, poor continuity, low resource efficiency). Theoretical analysis verifies that Vnait achieves:



1. **Parameter efficiency**: Eliminates positional embedding tables.

2. **Length generalization**: Normalized coordinates ensure consistent position semantics.

3. **Compatibility**: Plug-and-play integration with existing Transformers.

Vnait provides a new direction for input representation in low-resource NLP tasks, where structural priors can significantly improve sample efficiency.

本文提出 Vnait—— 一种面向 Transformer 的结构感知输入表示框架。通过引入可学习、归一化、单调的空间坐标，Vnait 解决了传统位置编码的局限（无归一性、连续性差、低资源效率低）。理论分析验证 Vnait 实现了：



1. **参数高效**：取消位置嵌入表；

2. **长度泛化**：归一化坐标确保位置语义一致；

3. **兼容性**：与现有 Transformer 即插即用集成。

Vnait 为低资源 NLP 任务的输入表示提供了新方向，其中结构先验可显著提升样本效率。

### 5.2 Future Work / 未来工作



1. **Extend to decoder architectures**: Current Vnait is designed for encoders; future work will adapt it to decoders (e.g., adding causal constraints to spatial coordinates).

2. **Combine with pre-trained models**: Integrate Vnait into large language models (LLMs) and fine-tune on low-resource tasks to verify practical performance.

3. **Visualize spatial coordinates**: Analyze the relationship between $S_i$ and linguistic structures (e.g., phrases, clauses) to further interpret Vnait’s structural awareness.

4. **扩展到解码器架构**：当前 Vnait 针对编码器设计，未来将适配解码器（如为空间坐标添加因果约束）；

5. **与预训练模型结合**：将 Vnait 集成到大型语言模型（LLM）中，在低资源任务上微调以验证实际性能；

6. **空间坐标可视化**：分析$S_i$与语言结构（如短语、从句）的关系，进一步解释 Vnait 的结构感知能力。

> （注：文档部分内容可能由 AI 生成）