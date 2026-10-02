# 训练框架：从数学目标到可检查的训练程序

**想做完整自测或准备面试，打开 [各专题题目索引](../references/面试复习与题目索引.md)。** 题目均给出完整条件、参考答案、常见追问与回答；基础讲解、手算与实验排错按专题连接。

沿用入门课中“输入 2，目标 5”的模型，本册依次追踪五件事：数据怎样排成张量，预测怎样产生损失，梯度怎样传回参数，优化器怎样更新参数，更新后怎样验证。第 1 章处理形状和求导，第 2 章连成训练循环；第 3 章才比较两种框架怎样表达同一计算，第 4 章教你用证据排查失败。**张量**在这里就是带形状和数据类型的多维数组，**自动微分**是沿已执行的计算自动应用链式法则。读代码时先标出每个轴代表什么，再看函数名。

**新手先读：[从零上手：亲手算清一次参数更新](从零上手.md)。** 从 Python 与 torch 的区别、列表和张量开始，用完整中文算例和实跑代码连通前向、损失、反向、更新与验证；不要求先读英文教材。

读完第一次数值更新，做[三个短程序实验](短程序实验.md)：先运行 20 行左右的形状和梯度例子，再用三点数据训练一条直线。PyTorch 与 MindSpore 使用同一数学任务，输出可以逐项对照。完整的训练、保存与恢复实验留到第 4 章后做。

这一册服务于 [GOAL 的第 1 月与项目 A](../GOAL.md)：把张量、梯度、损失和优化器连成一个能训练、验证、保存和恢复的程序。数学基础帮助你推导梯度；框架能力则要求确认程序实际计算了这个梯度。完成后进入 [Transformer 与语言模型](../transformer/README.md) 的小型语言模型实践，把这里的线性预测器换成 decoder-only 网络，训练闭环继续复用。

PyTorch 是本年度项目的主要实践框架，MindSpore 用于学习概念迁移和执行模式差异。先把一个框架的训练循环解释清楚，再对照另一套表达；当前不要求同时维护两套完整 LLM 项目。

## 阅读顺序与验收

| 顺序 | 章节 | 可检查的结果 |
|---|---|---|
| 1 | [张量、广播与自动微分](chapters/01_张量_广播与自动微分.md) | 写清张量每个轴的语义；手算、autograd、有限差分三种梯度一致 |
| 2 | [损失函数、优化器与训练循环](chapters/02_损失函数_优化器与训练循环.md) | 能解释交叉熵、有效 token 归约、清零梯度和参数更新 |
| 3 | [PyTorch 与 MindSpore 概念对照](chapters/03_PyTorch与MindSpore概念对照.md) | 按数学契约比较前向、梯度、参数与模式，不靠替换 API 名称迁移 |
| 4 | [训练排错与可复现实验](chapters/04_训练排错与可复现实验.md) | 能定位一个广播/梯度/标签问题，验证保存恢复与对照公平性 |

建议把本册压缩为第 1 月的第一段实践：先做梯度实验，再拿一个小批次过拟合，随后接入项目 A 的真实 token 数据。已经能独立完成的部分用验收压缩复习，不必逐项抄写 API。

## 本次环境与版本边界

2026-09-26 做了只读检查，未安装任何依赖：

| 解释器 | 本次核验 |
|---|---|
| `C:/Python313/python.exe` | Python 3.13.5；未发现 `torch` 或 `mindspore` 包 |
| `D:/anaconda/envs/pytorch_env/python.exe` | Python 3.11.15；PyTorch 2.7.1+cu118 可导入，CUDA 可用；未发现 MindSpore |
| `D:/anaconda/envs/d2l/python.exe` | Python 3.9.25；发现 torch 包，未在此环境执行训练；未发现 MindSpore |

“未发现”仅指上面检查过的解释器，不代表机器上所有环境都没有。下面实验使用现有 `pytorch_env`，明确运行于 **CPU、float64**，不以 CUDA 可用作为 GPU 训练已验证的证据。PyTorch 代码依据实跑的 2.7.1 与官方 2.7 文档；MindSpore 对照固定查阅 2.4.0 官方文档，属于未在本机执行的教学示例，不声称是当前最新版或兼容所有设备。

## 直接运行实验

在仓库根目录执行；目录名称包含 `&`，PowerShell 中路径必须加引号：

```powershell
& 'D:/anaconda/envs/pytorch_env/python.exe' 'pytorch&mindspore/experiments/linear_train_lab.py' --output-dir 'pytorch&mindspore/experiments/results'
```

脚本检查广播反例、手推/autograd/有限差分梯度，然后用 SGD 拟合已知线性关系，验证独立输入点预测、权重保存重载，以及带动量优化器状态的中断续训。它写出 [summary.json](experiments/results/summary.json)、[训练曲线](experiments/results/learning_curve.csv)和本机生成的检查点。同一路径重复运行会覆盖实验自己的同名输出，改变实验时请使用新输出目录。

这是框架与训练机制验收，**不等于项目 A 已完成**。项目 A 仍需分词、因果 Attention、next-token 标签、语言模型训练、独立验证和生成样例。MindSpore 迁移实跑也仍是后续练习。

## 精确阅读入口

| 资料与版本 | 读哪里 | 用来回答什么 |
|---|---|---|
| D2L 1.0.3 | [2.5 Automatic Differentiation](https://d2l.ai/chapter_preliminaries/autograd.html) | 为什么梯度会累积，计算图怎样产生 |
| D2L 1.0.3 | [3.4 Linear Regression Implementation from Scratch](https://d2l.ai/chapter_linear-regression/linear-regression-scratch.html) | 模型、损失、优化器怎样组成训练循环 |
| D2L 1.0.3 | [5.3 Forward/Backward Propagation](https://d2l.ai/chapter_multilayer-perceptrons/backprop.html) | 共享计算图如何复用链式法则 |
| Goodfellow 等，《Deep Learning》 | [第 6 章 Deep Feedforward Networks](https://www.deeplearningbook.org/contents/mlp.html) | 网络如何定义函数，反向传播求的是什么 |
| 同上 | [第 8 章 Optimization](https://www.deeplearningbook.org/contents/optimization.html) | 优化目标、随机梯度与数值条件如何影响训练 |
| PyTorch 2.7 | [Autograd mechanics](https://docs.pytorch.org/docs/2.7/notes/autograd.html)、[Reproducibility](https://docs.pytorch.org/docs/2.7/notes/randomness.html) | 框架的实际语义与可复现边界 |
| MindSpore 2.4.0 | [PyNative](https://www.mindspore.cn/docs/en/r2.4.0/model_train/program_form/pynative.html)、[Graph](https://www.mindspore.cn/docs/en/r2.4.0/model_train/program_form/static_graph.html) | 执行与编译方式怎样影响开发和排错 |

这些入口用于按问题查阅，正文是原创教学说明，不是教材逐章转录。后续读训练论文时，先把目标函数写成带形状的伪码，再讨论模型规模与性能。
