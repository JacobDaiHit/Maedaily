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

所以深度网络失去意义。

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
| 新型平滑激活 | Swish、Mish | 大模型Transformer |

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

不可导。


训练需要：

\[
\frac{\partial L}{\partial w}
\]


但是阶跃函数：

几乎处处：

\[
f'(x)=0
\]


无法反向传播。

所以后来被Sigmoid替代。


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

90%概率。


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

数据平均更接近0：

\[
E(x)\approx0
\]


优化更容易。


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

# 5. ReLU（目前最重要）

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


神经元可能永久关闭。


---

## 使用

大量CNN：

- AlexNet
- VGG
- ResNet


早期Transformer FFN也常用。


---

# 6. Leaky ReLU

解决死亡ReLU。


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

输出均值更接近0。


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

GPT系列：

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


比Swish更平滑。


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


GELU比ReLU更适合大规模训练。


所以：

GPT：

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
(W_1x)\odot Swish(W_2x)
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


类似LSTM门控。


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

背后的核心变化是：

> 从“把神经元打开或关闭”，发展到“更精细地控制信息通过多少”。

这也是为什么现代大模型（GPT、LLaMA等）的激活函数越来越像门控系统，而不是简单的非线性函数。