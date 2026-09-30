# Transformer 与基座模型：从序列表示到可训练的语言模型

**新手先读：[从零上手：一句话怎样变成可以训练的语言模型](从零上手.md)。** 从中文 token、编号和向量开始，手算三位置注意力，再运行微型语言模型，按中文解释理解 mask、标签、损失与真实结果。

这部分衔接 [RNN](../RNN/README.md) 与 [后训练](../post_train/README.md)。读完应能回答：一个批次如何产生概率和梯度；为什么训练、生成的成本不同；更大的模型为什么需要数据和系统共同支撑。

本卷从一句短文本推进：先把词切成 token（文本单位），让每个位置按需要读取前文，这就是第 1 章的注意力；再让模型在每个位置预测下一个 token，形成第 2 章的训练目标。能训练后，才问位置怎样表示、重复生成怎样节省计算，以及大规模训练时数据、计算、显存和通信各受什么约束。第 3—5 章分别回答这些后续问题。初读时先用三位置例子检查信息能否传到该到的位置，再读效率公式。

| 顺序 | 章节 | 学习后交付 |
| --- | --- | --- |
| 1 | [Attention Is All You Need](<Attention Is All You Need.md>) | 手算注意力、画出形状、解释两种 mask |
| 2 | [Decoder-only 与语言模型训练](chapters/02_DecoderOnly与语言模型训练.md) | 标签移位、有效 token 平均、一个训练循环 |
| 3 | [位置编码、KV Cache 与效率](chapters/03_位置编码_KVCache与注意力效率.md) | 推导 RoPE 的相对位置性质，估算缓存 |
| 4 | [预训练数据与 Scaling](chapters/04_预训练数据与Scaling.md) | 数据说明和固定预算下的对照设计 |
| 5 | [MoE、MLA 与训练系统](chapters/05_MoE_MLA与训练系统.md) | 区分参数、计算、通信和显存四种成本 |

第 1 月完成前两章和 [最小语言模型实验](experiments/README.md)；第 7–8 月再系统读后两章，不必等所有基础读完才开始 SFT。第三章随实际生成与显存问题补齐。书籍按 [参考地图](../references/参考书籍与课程地图.md) 选择章节。

论文原文与省流：[Transformer](papers/notes/2017_Transformer.md)、[Chinchilla](papers/notes/2022_Chinchilla.md)、[ZeRO](papers/notes/2020_ZeRO.md)、[DeepSeek-V3](papers/notes/2024_DeepSeek_V3.md)。它们分别是结构、算力分配、训练系统和综合技术报告，不能用同一种“跑分高低”来判断价值。

进入后训练前，至少能独立解释一个 `[B,T,V]` logits 张量、一个被忽略的 label、一次反向传播，以及为什么 padding 和未来位置不能泄漏。做不到时回到 [框架基础](../pytorch&mindspore/README.md)，无需重读全部深度学习。
