# 第 1 章：Attention 如何改变序列建模

**本章问题：每个位置怎样直接从其他位置取信息，而不把一切都压进递归状态？** 前置是矩阵乘法、softmax 和 [RNN 梯度传播](../RNN/chapters/01_梯度传播_门控与注意力的起点.md)。原始 Transformer 是机器翻译的 encoder–decoder；今天常见的 decoder-only 语言模型是下一章的另一种配置。

输入矩阵里的 token 向量从哪里来、怎样收到梯度，以及它与上下文隐藏状态有什么区别，见[Embedding 前置章](chapters/01_Embedding与表示学习.md)。本章接着解释这些位置的向量怎样互相读取。

先想一句三词短句：“小猫 / 追逐 / 毛线”。在读“追逐”时，模型可能需要从“小猫”取出“谁在做动作”的信息；在读“毛线”时，可能需要从“追逐”取出“发生了什么”的信息。**注意力**是一套按当前位置的需要，给其他位置分配读取权重并汇总内容的计算。它不直接告诉我们模型最终理解了什么，先把读取的来源、权重和结果算清。

## 1. 从固定摘要到按需读取

早期 seq2seq 将源句压进一个固定向量，解码时长期依赖这份摘要。Bahdanau attention 让解码器在每一步对源端隐藏状态计算权重，因此可以按当前需要读取不同位置。Transformer 进一步用 self-attention 建立同一序列内的交互，训练时各位置的表示可以成批计算。[Bahdanau 等](https://arxiv.org/abs/1409.0473)、[Transformer 原文](https://arxiv.org/abs/1706.03762v1)

注意力不是“模型知道哪些词重要”的定义，而是一个可微分的数据读取算子。权重来自查询与键的匹配，值才是实际传递的内容。注意力图有助于观察模型，却不能单凭颜色把某个头解释成可靠的因果机制。

## 2. 先看形状，再看公式

把一条句子排成矩阵 $X\in\mathbb R^{T\times d}$：$T$ 是 token 位置数，$d$ 是每个位置的表示维度；先忽略一次处理多条句子的 batch。一次独立的读取计算叫一个**注意力头**。它把每个位置的表示投影成三种向量：**查询** $Q$ 表示当前位置想找什么，**键** $K$ 表示各位置可按什么被匹配，**值** $V$ 是真正被读取的内容。一个头计算

$$Q=XW_Q,\quad K=XW_K,\quad V=XW_V,$$

其中 $W_Q,W_K,W_V$ 是待学习的投影矩阵，$Q,K\in\mathbb R^{T\times d_k}$，$V\in\mathbb R^{T\times d_v}$；$d_k,d_v$ 分别是键和值的维度。令 $M$ 为位置可见性约束矩阵：禁止读取的位置加上负无穷。查询与键相乘得到两两匹配分数，再按行用 softmax 变成权重：

$$A=\operatorname{softmax}_{\text{行}}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right),\qquad H=AV.$$

$A_{ij}$ 是位置 $i$ 读取位置 $j$ 的权重；每个允许读取的行权重和为 1；$H$ 是汇总后的表示。若键和查询分量独立、均值为零且方差约为 1，点积方差约为 $d_k$；除以平方根可减轻 softmax 饱和。这是设计动机的近似分析，并非训练后分量仍独立的保证。

手算例子：某行允许两个位置，其缩放后分数为 $[0,\log 3]$，对应权重 $[1/4,3/4]$；若两个值为 $[2,6]$，输出是 5。把第二个位置屏蔽后，输出变成 2。实现要先加 mask 再 softmax；先 softmax 再把未来权重清零会破坏归一化。

多头注意力让多个投影空间并行读取：

$$\operatorname{MHA}(X)=\operatorname{Concat}(H_1,\ldots,H_h)W_O.$$

常取 $d_k=d_v=d/h$，所以“更多头”不等于在固定 $d$ 下增加同等比例的参数。实际张量通常整理为 `[B,h,T,d_head]`，分数为 `[B,h,T,T]`。实现 reshape 后必须确认位置轴和头轴没有交换错。

## 3. 两种 mask 与三种 attention

因果 mask 在 $j>i$ 时设为负无穷，使位置 $i$ 只能读当前位置及过去。padding mask 屏蔽补齐位置；它与因果约束解决不同问题。若一行全部被屏蔽，普通 softmax 可能产生 NaN，不能把这种输入悄悄当成合法样本。

原论文有三种读取：encoder self-attention 可读取完整源句；decoder self-attention 只能读取目标前缀；cross-attention 的 query 来自解码器，key/value 来自编码器。后者的分数形状是 `[B,h,T_target,T_source]`，不要求两个长度相等。

训练时给定完整目标序列，可以并行计算各位置的 next-token loss，但因果 mask 保证不能偷看答案。自回归生成时，下一个 token 依赖刚生成的 token，所以通常仍逐步生成。“训练位置可并行”不能写成“所有生成 token 可一次并行得到”。

## 4. Attention 只是一个模块

原论文还包含位置编码、逐位置前馈网络、残差、LayerNorm、dropout、特定学习率调度及标签平滑。FFN 对每个位置使用相同参数，但不会在此步骤直接混合不同位置。原始块是 post-LN；本项目教学实验使用 pre-LN，必须在讲解时说明差别。

完整层不只有 $T^2d$ 的注意力成本，还包括投影和 FFN 的约 $Td^2$ 成本。长序列时注意力矩阵可能成为瓶颈，小序列大隐藏维度时则不能忽略投影与 FFN。计算式帮助提出假设，实际吞吐还取决于内存访问和算子实现。

## 5. 深入：手写 Attention 的前向、反向与完整块

### 从一个读取算子到可执行核心

下面的函数接收已经投影并分头的 Q/K/V，形状分别为 `[B,h,Tq,dk]`、`[B,h,Tk,dk]`、`[B,h,Tk,dv]`。`allowed` 是能广播到 `[B,h,Tq,Tk]` 的布尔数组，True 表示可见；它是本函数自己的契约，不代表所有库接口都使用同一真假含义。这里要求每个 query 至少有一个可见 key，不含 dropout。

```python
import math
import torch

def attention_core(q, k, v, allowed):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    visible = torch.broadcast_to(allowed, scores.shape)
    if not visible.any(dim=-1).all():
        raise ValueError("存在没有可见 key 的 query")
    scores = scores.masked_fill(~visible, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v, weights
```

训练使用完整因果三角区域；跨注意力则依据源端有效位置设 mask。上游投影、分头、输出投影和 loss 仍由模型定义。若某 API 用 True 表示屏蔽而这里表示可见，直接搬 mask 会反转读取关系，必须用两位置的极小输入核对。

### softmax 的导数说明如何学习读取

令 $S=QK^\top/\sqrt{d_k}+M$、$A=\operatorname{softmax}(S)$、$O=AV$。对一行 softmax 有 $\partial A_j/\partial S_k=A_j(\mathbf1\{j=k\}-A_k)$。设输出上游梯度为 $G=\partial L/\partial O$，则

$$
\nabla_VL=A^\top G,\qquad D_A=GV^\top,
$$

$$
(D_S)_{ij}=A_{ij}\left((D_A)_{ij}-\sum_kA_{ik}(D_A)_{ik}\right),
$$

$$
\nabla_QL=D_SK/\sqrt{d_k},\qquad
\nabla_KL=D_S^\top Q/\sqrt{d_k}.
$$

这里的矩阵公式先忽略 batch/head，再逐批逐头使用。屏蔽位置的 A 为零，在合法非空行下其 score 梯度为零；参与读取的 V 则按读取权重获得梯度。匹配分数改变与内容向量改变是两条路径，因此不能把 attention 权重直接当作最终输出的完整解释。

例如一个 query q=1，两个 key `[0,ln3]`，两个 value `[2,6]`，$d_k=1$，全部可见；输出 O=5。令本次数值校验标量 L=O，故上游 G=1。得到 value 梯度 `[1/4,3/4]`、score 梯度 `[-3/4,3/4]`、query 梯度 $3\ln3/4$，key 梯度 `[-3/4,3/4]`。这不是实际语言模型的损失，只是用于检验局部反向。

### 残差和归一化放在哪里

LayerNorm 对每个位置的特征向量 x 计算均值和总体方差，并用可学习 gain/bias 调整：

$$
\operatorname{LN}(x)=\gamma\odot\frac{x-\mu(x)}{\sqrt{\sigma^2(x)+\varepsilon}}+\beta.
$$

这里统计轴是特征轴，$\varepsilon>0$ 防止除零，$\odot$ 表示逐分量乘法。pre-LN 子层为 $y=x+F(\operatorname{LN}(x))$，局部 Jacobian 含 $I+J_FJ_{LN}$；post-LN 为 $y=\operatorname{LN}(x+F(x))$，Jacobian 为 $J_{LN}(I+J_F)$。pre-LN 保留显式恒等路径，是理解梯度传播的一个依据；深层稳定性仍受初始化、尺度和优化影响，不能由一个 I 保证。[归一化位置研究](https://arxiv.org/abs/2002.04745)

### 参数与 FLOPs 要从运算数出来

采用标准等宽 MHA，无 bias，Q/K/V/O 各有 $d^2$ 参数；两层 FFN 宽度 m 有 $2dm$ 参数。取 m=4d，主体参数为 $12d^2$，固定 d 时把头数翻倍并不会翻倍这些投影参数。忽略 norm、非线性、mask 和 softmax，一次长度 T、单条序列前向的主要 FLOPs 约为

$$
8Td^2+4T^2d+4Tdm=24Td^2+4T^2d\quad(m=4d).
$$

按一次乘加为 2 FLOPs；前两项来自四个投影和 QK/AV，最后来自两次 FFN 矩阵乘。$T/d$ 较小时投影与 FFN 很重要；长序列配对项变大。要把参数存储、激活存储和 FLOPs 分开算，再用实测说明墙钟成本。

## 6. 完整自测与面试问答

### 题 1：多头注意力的张量形状与缩放因子

**题目：**输入 $X$ 为 `[B,T,d]=[2,5,16]`，头数 h=4，每头查询、键和值维度均为 4。各投影先输出 `[B,T,d]`，再分头。写出分头后的 Q/K/V、$QK^\top$、注意力权重、每头输出、拼接输出的形状；softmax 沿哪个轴，为什么除以 $\sqrt{d_{head}}$？

**参考答案：**Q/K/V 为 `[2,4,5,4]`；分数与权重为 `[2,4,5,5]`；每头输出为 `[2,4,5,4]`；转回位置轴后拼接为 `[2,5,16]`。Softmax 沿最后的键位置轴，使一个查询对可见位置的权重和为 1。若查询与键分量独立、均值零、方差为 1，点积方差随头维度增长，除以平方根使其量级更稳定；这是设计动机的假设分析。

**面试追问：**Query、key 与 value 为什么是三种投影？

**追问回答：**Query 表达读取请求，key 用于匹配，value 承载被汇总的内容。分开投影允许匹配空间与内容空间不同；不是为每个 token 人工规定重要程度。

### 题 2：怎样检查模型没有偷看未来？

**题目：**一个确定性 decoder-only 模型处于 eval 模式，关闭 Dropout。输入 S=`[BOS,1,2,3,4]`，仅将最后一项改为 9 得 S2，其他权重、位置编号与 mask 相同。第 i 个 logits 使用输入至 i，预测下一个 token。请指出哪些 logits 必须保持一致，并说明应怎样使用这个测试。

**参考答案：**输入索引 0—3 的 logits 必须在数值容差内一致，因为其允许前缀没有变化；索引 4 可以变化。比较两次前向的对应前缀，能发现未来泄漏的一类错误。可加入故意移除 mask 的反例；但若特定权重恰好不利用未来，该反例也可能碰巧不变化，故单个测试不是对所有输入的证明。

**面试追问：**为什么训练可并行，生成仍通常逐 token？

**追问回答：**训练已给完整目标序列，因果约束下可同时计算各前缀的预测；生成时后续前缀尚未确定，需要先产生上一 token。

### 题 3：看到当前输入，为什么仍可合法预测下一 token？

**题目：**完整序列为 `[BOS,我,爱,数学,EOS]`。手动移位一次，输入 `[BOS,我,爱,数学]`，目标 `[我,爱,数学,EOS]`。因果注意力允许当前位置读取自己和更早输入。逐位置写出预测目标，说明当前位置可见自己是否泄漏答案，并区分因果与 loss mask。

**参考答案：**BOS 位置预测“我”，“我”预测“爱”，“爱”预测“数学”，“数学”预测 EOS。当前输入与被预测的下一 token 不同，可见自己合法；若移位漏做，当前输入恰为标签就可能形成复制泄漏。因果 mask 控制信息可见性，loss mask 控制哪些目标监督。

**面试追问：**Self-attention 与 cross-attention 的来源差在哪？

**追问回答：**Self-attention 的 Q/K/V 来自同一序列表示；cross-attention 的 Q 来自当前解码序列，K/V 来自另一个来源，如编码器输出。两种序列长度可以不同。

### 题 4：手推一个 attention 读取的反向

**题目：**q=1，key 为 `[0,ln3]`、value 为 `[2,6]`，单头维度为 1，两个位置都可见。softmax 沿 key 轴，输出 $O=\sum_jA_jv_j$，本题标量 L=O，上游导数为 1。求 A、O、对 value、score、q 和 key 的梯度；写出函数核对解析反向与 autograd。

**参考答案：**A=`[1/4,3/4]`，O=5；value 梯度同 A。score 梯度为 `A_j(v_j-O)`，即 `[-3/4,3/4]`。q 梯度为 $3\ln3/4$，key 梯度为 `[-3/4,3/4]`。提交局部导数推导后，用 float64 实现上述核心，对独立小输入核对每个梯度；不要用两次同一 autograd 当独立证明。

**面试追问：**若第二个 key 被屏蔽，结果怎样变化？

**追问回答：**唯一可见位置权重为 1，输出 2，value 梯度 `[1,0]`，score/q/key 的梯度在该读取中均为零；整行都屏蔽则不是合法 softmax，应按约定拒绝或显式处理。

### 题 5：固定隐藏维度，多头数量影响哪些成本？

**题目：**标准 MHA 的 Q/K/V/O 投影均为 d×d，FFN 两层宽度 4d，不含 bias、norm 参数与其他层。取 d=64，比较 h=4 和 h=8 的主体参数量；再按本章 FLOPs 模型计算 T=64 和 T=512 的一次单序列前向。说明注意力矩阵激活的头轴如何影响存储。

**参考答案：**两种头数主体参数均为 $12×64^2=49,152$。T=64 时主要 FLOPs 为 7,340,032；T=512 时为 117,440,512。每头维度变化使 QK/AV 总点积成本在固定 d 下保持同口径，但若显式保存所有头的权重，元素数为 hT²，随 h 增长。实际实现可能融合、重算或不保存该完整矩阵。

**面试追问：**参数一样能证明表达能力和吞吐一样吗？

**追问回答：**不能。分头改变子空间分解，内核形状与存储也可能不同；需要固定其余结构与预算分别比较任务质量和系统指标。

### 题 6：给定 mask bug，设计能发现它的测试

**题目：**一个实现先对三项分数 `[0,0,0]` 做 softmax，再把第三项的权重清零而不重新归一化；value 为 `[0,0,9]`，第三项应屏蔽。先给出权重和权重和，再将 value 改为 `[3,3,9]`，比较错误实现与 mask-before-softmax 的输出。设计因果和 padding 两类回归检查。

**参考答案：**错误权重 `[1/3,1/3,0]`，和为 2/3；第一组输出碰巧都为零，不能暴露缩放错误。第二组错误输出 2，正确权重 `[1/2,1/2,0]`，输出 3。检查权重在可见 key 上和为 1；改未来 token 时前缀 logits 不变；追加右 padding 时有效输出不变，均需 eval 和固定位置编号。

**面试追问：**只检查“未来位置权重是零”为什么不够？

**追问回答：**零权重不保证其余权重正确归一化，也不覆盖标签错位、其他跨位置模块或缓存位置错位；应同时核对信息流与输出。

本章手推可用 [attention_math_lab.py](experiments/attention_math_lab.py) 的解析 VJP 与 autograd 对照核验；在项目根目录运行 `python scripts/run_lesson.py attention`。先独立计算再对照输出，核验范围见[实验说明](experiments/README.md)。

继续读 [语言模型训练](chapters/02_DecoderOnly与语言模型训练.md)。省流笔记见 [原论文实验与边界](papers/notes/2017_Transformer.md)，不要把 2017 年翻译结果当成现代通用 LLM 的实验证据。
