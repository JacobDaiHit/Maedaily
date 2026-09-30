# 第 3 章　PyTorch 与 MindSpore：迁移计算契约

换框架时，最值得迁移的是你对目标函数、张量轴、参数状态和求导路径的理解。API 名字相近并不保证默认初始化、loss 归约与执行方式相同；结果不一致也不一定说明某一框架错了。本章以同一个数学问题建立对照，再讨论动态图与静态图。

前两章已明确“输入 2，预测 5”的张量形状、损失和更新顺序。现在让两套框架实现同一计算：先对齐输入、参数初值、损失归约和梯度，再比较代码组织方式。这样读下面的 API 表时，每个函数名都有一个已定义的数学对象与之对应。

[三个短程序](../短程序实验.md)把同一组数写成 PyTorch 与 MindSpore 两版：形状都应是 `[3,3]` 与 `[3]`，梯度都应是 $(-6,-3)$，训练应学到约 $(w,b)=(2,1)$。**先比较这些可见数值，再比较语法**。本次只实跑 PyTorch；MindSpore 程序按固定 2.4.0 文档编写，留给有适配环境时核对。

## 3.1 先声明版本与验证范围

本章 PyTorch 代码对应本机实跑的 2.7.1，文档查阅固定为 2.7。MindSpore 使用 **2.4.0 官方文档**，未在本次检查的 Python 环境中发现安装，因此下面 MindSpore 片段是按文档写的教学示例，尚未执行；不据此保证它在其他版本、Windows 安装组合或设备后端上可运行。

版本限制不是说概念失效，而是把“数学关系”“文档描述”和“本机验证”分开。未来安装或切换环境时，首先检查对应版本支持的 Python、操作系统和硬件，再运行最小梯度例子，不能仅凭某个设备有 GPU 就推出框架支持它。

## 3.2 两套框架表达同一张计算图

设预测为 $\hat Y=XW^\top+b$，其中 $X$ 为 `[B,d]`，存储权重 $W$ 为 `[h,d]`，输出 $\hat Y$ 为 `[B,h]`，$B$ 是批大小、$d$ 是输入维度、$h$ 是输出维度。这是 PyTorch `Linear` 与 MindSpore `Dense` 常用的权重布局；从手写的 $XW$ 迁移时要先确认自己使用的 $W$ 是否已经转置。[MindSpore Dense 2.4.0](https://www.mindspore.cn/docs/en/r2.4.0/api_python/nn/mindspore.nn.Dense.html)给出其公式与形状。

| 数学或工程职责 | PyTorch 2.7 常见表达 | MindSpore 2.4.0 常见表达 |
|---|---|---|
| 带状态的模型 | `nn.Module`，定义 `forward` | `nn.Cell`，定义 `construct` |
| 可训练参数 | `nn.Parameter` | `Parameter` |
| 仿射映射 | `nn.Linear` | `nn.Dense` |
| 求导 | `backward` 或 `autograd.grad` | `value_and_grad` 等函数变换 |
| 训练/评估行为 | `train()` / `eval()` | `set_train(True/False)` |
| 执行形态 | eager，可按需编译 | PyNative、Graph 与局部 JIT |

这个表是职责映射，不是逐行替换规则。例如 PyTorch 的 `.backward()` 默认把叶子参数梯度累加到 `.grad`；MindSpore 的 `value_and_grad` 返回一个同时产出值与梯度的新函数。迁移时必须重新明确“梯度保存在哪里，谁接收它，何时更新参数”。模型前向与训练模式接口可查 [Cell 2.4.0 文档](https://www.mindspore.cn/docs/en/r2.4.0/api_python/nn/mindspore.nn.Cell.html)。

## 3.3 一个足够小的跨框架校验

固定函数 $L(w)=\tfrac12(2w-5)^2$，在 $w=1$ 处应有 $L=4.5$、$dL/dw=-6$。先检查这个函数，再迁移完整网络。

```python
# PyTorch 2.7 表达；完整同类梯度校验已在实验脚本实跑。
import torch
w = torch.tensor([1.0], requires_grad=True)
loss = ((2.0 * w - 5.0) ** 2).sum() / 2.0
loss.backward()
print(loss.item(), w.grad)  # 预期 4.5，[-6]
```

```python
# MindSpore 2.4.0 文档对应教学示例；本机未执行。
import mindspore as ms
from mindspore import ops
ms.set_context(mode=ms.PYNATIVE_MODE)
def loss_fn(w):
    return ops.sum((2.0 * w - 5.0) ** 2) / 2.0
loss, gradient = ms.value_and_grad(loss_fn)(ms.Tensor([1.0], ms.float32))
print(loss, gradient)  # 数学预期 4.5，[-6]
```

若函数额外返回预测值作为日志信息，要明确这些辅助输出不参与求导。MindSpore 2.4.0 的 `has_aux=True` 表示只有首个输出参与梯度；对权重求导则使用 `grad_position=None` 并传入权重集合。具体返回结构要对照 [value_and_grad 文档](https://www.mindspore.cn/docs/en/r2.4.0/api_python/mindspore/mindspore.value_and_grad.html)，不要凭输出元组位置猜测。

## 3.4 动态图与静态图改变什么

PyNative 模式便于按 Python 执行顺序调试和查看中间张量；Graph 模式先把支持的程序结构组织成图，再编译执行，可能获得跨算子优化机会。首轮编译成本、支持的 Python 语法、形状变化是否触发重新编译，都会影响体验与耗时。这里的“静态”不等于只能固定 batch，也不等于永远不能表达数据相关控制流；能力取决于版本、图表示和具体用法。

动态与静态也不是两个互斥阵营。MindSpore 2.4.0 文档给出了动态图中局部 `jit` 的方式，PyTorch 则可以对 eager 程序按需编译。转换失败、回退与重编译等机制要按具体版本核查。首次迁移应先让两端的输入、前向输出、梯度一致，再比较编译性能；不要一开始同时改变框架、精度、模型与设备。

## 3.5 从最小例子迁移到项目 A

完整迁移按五层验证：同一输入和参数的输出；同一标量目标的梯度；同一步优化后的参数；同一短训练的曲线；独立验证集的行为。初始化不同会使第一步就不一致，因此先显式复制数值，而不是期待相同 seed 在不同框架中产生相同随机张量。

语言模型还要核对 LayerNorm 的 epsilon、注意力 mask 真假语义、dropout、词表轴和损失归约。同一个 `[B,T,V]` 张量并不足以说明标签与时间步对应正确。检查点通常也不能仅通过改后缀跨框架读取，需要参数名、布局、精度和模型配置的显式映射。

## 3.6 练习与阅读目的

1. 同一 seed、同一网络宽度，为何跨框架初始输出不同？提示：随机数实现、初始化分布与调用顺序都可能不同。
2. 如果前向一致而梯度差一倍，先检查哪里？提示：平方损失是否带 $1/2$、归约是 sum 还是 mean、是否重复计入辅助输出。
3. Graph 模式首轮慢，能否直接认为吞吐更低？提示：分开报告编译时间、预热与稳定阶段耗时。

精读 [MindSpore 2.4.0 PyNative](https://www.mindspore.cn/docs/en/r2.4.0/model_train/program_form/pynative.html)与[Graph 文档](https://www.mindspore.cn/docs/en/r2.4.0/model_train/program_form/static_graph.html)，目的在于理解执行边界；回看 [D2L 5.3 计算图](https://d2l.ai/chapter_multilayer-perceptrons/backprop.html)，确认两套 API 背后仍是同一条链式法则。
