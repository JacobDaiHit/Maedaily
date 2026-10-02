# 激活函数总览：先看曲线，再看梯度

第一次阅读请先走[从零上手](从零上手.md)的直线、ReLU 和 sigmoid 三个例子。本篇按函数分类供查阅；每个新函数先确认定义、输出范围与导数，再根据使用位置讨论训练效果。

先接上[入门第一课](../入门/第一课_从预测到学习.md)的预测例子：一层 $y=wx+b$ 只能画出直线。多层若每层仍只做线性或仿射运算，最后也能合并成一层；要让中间表示出现弯曲关系，就需要非线性操作。**激活函数**是作用在神经元中间数值上的函数，例如 ReLU 把负数变成 0，sigmoid 把实数压进 0 到 1。下面每看一个函数，先问输出范围、局部导数和适用位置，再讨论训练现象。

激活函数（Activation Function）是神经网络里非常核心的组件。它的本质作用可以概括为：

> **给神经网络引入非线性，让网络能够拟合复杂函数；同时通过不同形状的函数控制信息如何流动。**

如果没有激活函数，多层神经网络实际上仍然只是一个线性变换：

\[
W_3(W_2(W_1x+b_1)+b_2)+b_3
\]

最终仍然可以化简成：

\[
Wx+b
\]

因此，单靠叠加仿射层不会增加函数表达的非线性；但线性层本身仍有用，关键是它怎样与激活、残差等操作组合。

---

# 1. 激活函数分类总览

大致可以分：

| 类型 | 代表函数 | 主要用途 |
|-|-|-|
| 阶跃型 | Step | 早期神经元模型 |
| Sigmoid型 | Sigmoid | 二分类输出、门控 |
| 双曲型 | Tanh | RNN隐藏状态 |
| ReLU族 | ReLU、Leaky ReLU、GELU | 深度网络隐藏层 |
| 概率归一化 | Softmax | 多分类输出 |
| 平滑激活 | Swish、Mish | 某些前馈模块，需结合完整结构比较 |

---

# 2. Step Function（阶跃函数）

最早的神经网络思想来自感知机（Perceptron）。

公式：

\[
f(x)=
\begin{cases}
1,&x>0\\
0,&x\leq0
\end{cases}
\]


图像：

```
1 |       ______
  |
0 |______
        x
```


## 意义

像一个开关：

如果输入超过阈值：

打开。

否则：

关闭。


例如：

\[
w^Tx+b>0
\]

输出：

\[
1
\]


---

## 缺点

最大问题：

在零点不可导，其他位置导数为零。


训练需要：

\[
\frac{\partial L}{\partial w}
\]


但是阶跃函数：

几乎处处：

\[
f'(x)=0
\]


普通局部导数不能提供有效学习信号；感知机等仍可使用另行定义的更新规则。

平滑函数如 sigmoid 提供了可用导数；不同模型与训练方法并存。


---

# 3. Sigmoid

公式：

\[
\sigma(x)=\frac1{1+e^{-x}}
\]


范围：

\[
(0,1)
\]


曲线：

```
1        ______
        /
0.5 ----
      /
0 _____
```


---

## 特点

### 优点：

输出可以理解为概率。

例如：

\[
\sigma(x)=0.9
\]

表示：

若作为定义明确的二元概率模型输出，它给出的预测概率为 90%；是否校准为实际 90% 频率需在独立数据检查。门控数值不自动是事件概率。


---

### 缺点：

梯度消失。

因为：

\[
\sigma'(x)=\sigma(x)(1-\sigma(x))
\]


最大：

\[
0.25
\]


很容易越来越小。


---

## 使用位置

现在主要：

### ① 二分类输出层

例如：

垃圾邮件：

\[
P(spam)=\sigma(z)
\]


---

### ② LSTM/GRU门

例如：

遗忘门：

\[
f_t=\sigma(...)
\]


因为需要：

0~1控制比例。


---

# 4. Tanh（双曲正切）

公式：

\[
tanh(x)=
\frac{e^x-e^{-x}}
{e^x+e^{-x}}
\]


范围：

\[
(-1,1)
\]


图像：

```
1       ______
       /
0 -----
     /
-1___
```


---

## 和Sigmoid区别


Sigmoid：

\[
0\rightarrow1
\]


Tanh：

\[
-1\rightarrow1
\]


所以：

Tanh中心在0。


---

## 优点

函数关于零对称；输入分布相应对称等条件下，输出均值才为零：

\[
E[\tanh(X)]=0\quad\text{若 }X\text{ 的分布关于零对称}
\]


零中心输出可影响优化，但不能仅由函数对称性保证训练更容易。


---

## 缺点

仍然梯度消失。


---

## 使用

经典RNN：

\[
h_t=tanh(Wx+Uh_{t-1})
\]


LSTM候选记忆：

\[
\tilde c_t=tanh(...)
\]


---

# 5. ReLU（常用的分段线性激活）

Rectified Linear Unit

公式：

\[
ReLU(x)=max(0,x)
\]


图：

```
y

|
|       /
|      /
|_____/
      x
```


---

## 为什么革命性？

以前：

Sigmoid：

正负都压缩。


ReLU：

正数保持。


例如：

\[
x=100
\]

输出：

\[
100
\]


梯度：

\[
1
\]


---

## 优点

### ① 缓解梯度消失

正区间：

\[
f'(x)=1
\]


梯度可以传播。


---

### ② 计算简单

只需要：

\[
max(0,x)
\]


---

## 缺点

死亡ReLU：

如果：

\[
x<0
\]


输出：

\[
0
\]


梯度：

\[
0
\]


相关输入长期处于负区间且没有其他更新路径时，可能持续失活；一次负输入不足以判永久关闭。见 [条件与诊断](Dead_Neuron.md)。


---

## 使用

大量CNN：

- AlexNet
- VGG
- ResNet


早期Transformer FFN也常用。


---

# 6. Leaky ReLU

负区间保留局部斜率，缓解硬零导数造成的失活，但不保证整条梯度或优化正常。


公式：

\[
f(x)=
\begin{cases}
x,&x>0\\
0.01x,&x<0
\end{cases}
\]


图：

```
       /
      /
-----/
    /
```


负区间保留一点梯度。


---

使用：

- 一些CNN
- GAN


---

# 7. PReLU

Leaky ReLU升级。


公式：

\[
f(x)=
\begin{cases}
x,&x>0\\
ax,&x<0
\end{cases}
\]


区别：

Leaky：

\[
a=0.01
\]


固定。


PReLU：

\[
a
\]

是学习参数。


---

# 8. ELU

Exponential Linear Unit


公式：

\[
f(x)=
\begin{cases}
x,&x>0\\
\alpha(e^x-1),&x<0
\end{cases}
\]


特点：

负数区域平滑。


优点：

负区间允许负输出，在某些输入分布下可使输出均值更接近0；这不是任意数据上的零均值保证。


---

使用：

一些深度网络。


---

# 9. Softmax（非常重要）

Softmax不是普通激活函数。

它用于：

> 把多个数字转换成概率分布。


公式：

\[
softmax(x_i)
=
\frac{e^{x_i}}
{\sum_j e^{x_j}}
\]


例如：

模型输出：

\[
[2,1,0]
\]


Softmax：

\[
[0.67,0.24,0.09]
\]


加起来：

\[
1
\]


---

用途：

多分类输出。


例如：

猫狗鸟：

```
Linear层
 ↓
Softmax
 ↓
猫 0.7
狗 0.2
鸟 0.1
```


---

# 10. GELU（大模型核心）

Gaussian Error Linear Unit


公式：

\[
GELU(x)=x\Phi(x)
\]


其中：

\[
\Phi(x)
\]

是标准高斯分布CDF。


近似：

\[
GELU(x)
\approx
0.5x(1+tanh(\sqrt{2/\pi}(x+0.044715x^3)))
\]


---

## 为什么重要？

Transformer里的FFN大量使用。


例如：

GPT-2 的前馈层可作为 GELU 的一个具体例子：

```
Attention
 ↓
Linear
 ↓
GELU
 ↓
Linear
```


---

## 直觉

ReLU：

硬切：

\[
x<0\rightarrow0
\]


GELU：

软选择。


不是：

"负数全部不要"


而是：

"根据大小决定保留多少"


---

# 11. Swish

Google提出。


公式：

\[
Swish(x)=x\sigma(x)
\]


例如：

x越大：

接近x。


x很负：

接近0。


---

特点：

类似GELU。


使用：

- EfficientNet
- 一些Transformer


---

# 12. Mish

公式：

\[
Mish(x)=x\tanh(softplus(x))
\]


其中：

\[
softplus(x)=\log(1+e^x)
\]


Mish 与 Swish 都是光滑函数；“更平滑”需给具体度量，不能作为无需证据的性能结论。


使用：

YOLO系列。


---

# 13. 为什么大模型喜欢GELU？

以Transformer为例：

结构：

```
Input
 |
Embedding
 |
Attention
 |
FFN
 |
Output
```


FFN：

\[
FFN(x)=W_2\sigma(W_1x)
\]


这里需要：

- 非线性
- 平滑梯度
- 稳定训练


GELU 在若干 Transformer 中使用；优劣应由相同资源和配方的对照判断，不能从平滑性推出所有大规模训练更优。


所以：

使用 GELU 的 Transformer 前馈层：

\[
GELU
\]


BERT：

\[
GELU
\]


LLaMA：

\[
SwiGLU
\]


---

# 14. SwiGLU（现代LLM重点）

现在很多大模型使用。

例如：

LLaMA、PaLM。


结构：

不是：

\[
W_2GELU(W_1x)
\]


而是：

\[
W_3\left[(W_1x)\odot Swish(W_2x)\right]
\]


其中：

\[
\odot
\]

表示逐元素乘。


直觉：

两个通道：

一个负责信息：

\[
x
\]


一个负责控制：

\[
gate
\]


同样有相乘分支，但 Swish 门不限制在 0 到 1，也不包含 LSTM 的跨时间 cell state。


---

# 15. 总结表

|函数|数学形式|主要用途|
|-|-|-|
|Step|0/1|感知机|
|Sigmoid|\(\frac1{1+e^{-x}}\)|二分类、LSTM门|
|Tanh|\(\frac{e^x-e^{-x}}{e^x+e^{-x}}\)|RNN状态|
|ReLU|max(0,x)|CNN、早期深度网络|
|Leaky ReLU|负区间保留梯度|CNN/GAN|
|PReLU|可学习Leaky|CNN|
|ELU|平滑ReLU|深度网络|
|Softmax|概率归一化|分类输出|
|GELU|xΦ(x)|Transformer|
|Swish|xσ(x)|现代网络|
|SwiGLU|门控激活|大语言模型|

---

从发展路线看：

\[
\boxed{
Step
\rightarrow
Sigmoid/Tanh
\rightarrow
ReLU
\rightarrow
GELU
\rightarrow
SwiGLU
}
\]

上面的箭头是概念阅读路线，历史有并行分支，不代表新函数全面淘汰旧函数。学习重点是：

> 从“把神经元打开或关闭”，发展到“更精细地控制信息通过多少”。

理解具体模型时，应查看它的前馈层定义：GELU 是逐元素激活，SwiGLU 则包含两路投影、逐元素乘法与输出投影。不能仅凭模型系列名称推断采用哪一种。
