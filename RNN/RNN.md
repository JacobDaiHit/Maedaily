RNN（Recurrent Neural Network，循环神经网络）的核心数学思想其实非常简单：

> **把一个序列问题转化成一个动态系统：当前状态 = 上一个状态的信息 + 当前输入的信息。**

从数学角度看，RNN 本质上是在学习一个**非线性状态空间模型（nonlinear state-space model）**。

---

# 1. 为什么需要 RNN？

传统神经网络：

\[
y=f(x)
\]

输入和输出是一一对应的。

例如：

图片分类：

\[
\text{image}\rightarrow \text{cat}
\]

但是很多数据天然是序列：

- 文字：

\[
x_1,x_2,x_3,\cdots,x_T
\]

例如：

> "I love machine learning"

每个词的意义依赖前面的词。

- 股票：

\[
p_1,p_2,\cdots,p_T
\]

今天价格依赖历史走势。

- 语音：

\[
audio_1,audio_2,\cdots,audio_T
\]


所以我们需要一个模型：

\[
\boxed{
当前输出 = f(当前输入, 历史信息)
}
\]


故引出问题：历史信息怎么表示？

---

# 2. RNN的核心思想：隐藏状态（hidden state）

RNN 引入一个变量：

\[
h_t
\]

叫：

> 隐藏状态（hidden state）

它表示：

\[
\boxed{
\text{到时间 }t\text{ 为止，模型记住的信息}
}
\]


比如句子：

```
The movie was not very good
```

读到：

```
not
```

的时候：

RNN状态：

\[
h_t
\]

里面应该包含：

> 后面可能出现否定意义。


---

数学表达：

第 t 步：

输入：

\[
x_t
\]


之前记忆：

\[
h_{t-1}
\]


产生新记忆：

\[
\boxed{
h_t=f(x_t,h_{t-1})
}
\]


这就是 RNN 的灵魂。


---

# 3. 最简单RNN公式

通常写：

\[
h_t=\sigma(W_xx_t+W_hh_{t-1}+b)
\]


其中：

## 当前输入

\[
x_t
\]


经过：

\[
W_x
\]

得到：

\[
W_xx_t
\]


表示：

> 当前词带来的信息


---

## 历史状态

\[
h_{t-1}
\]


经过：

\[
W_h
\]

得到：

\[
W_hh_{t-1}
\]


表示：

> 过去记忆对现在的影响


---

二者相加：

\[
W_xx_t+W_hh_{t-1}+b
\]


然后激活：

\[
\sigma
\]


得到：

\[
h_t
\]


---

画成数学结构：

```
        h(t-1)
          |
          |
        W_h
          |
          v

x(t)--W_x--> (+) ---> σ ---> h(t)

```

---

# 4. 从数学角度看：RNN就是迭代函数

展开：

\[
h_t=f(x_t,h_{t-1})
\]


代入：

\[
h_{t-1}=f(x_{t-1},h_{t-2})
\]


得到：

\[
h_t
=
f(x_t,
f(x_{t-1},
f(x_{t-2},...)
))
\]


也就是说：

\[
\boxed{
h_t
=
F(x_1,x_2,\cdots,x_t)
}
\]


RNN实际上学习：

\[
\text{过去所有输入}
\rightarrow
\text{一个压缩状态}
\]


这和动态系统非常像。

---

# 5. RNN和马尔可夫过程的关系

经典马尔可夫：

\[
P(x_t|x_{t-1},x_{t-2},...)
\]


假设：

\[
P(x_t|x_{t-1})
\]


也就是：

未来只依赖现在。


RNN类似：

\[
\boxed{
h_t=f(h_{t-1},x_t)
}
\]


区别：

马尔可夫状态是人工定义。

RNN：

\[
h_t
\]

是神经网络自己学习出来的。

---

所以：

> RNN = 可学习的马尔可夫状态模型。


---

# 6. 为什么叫 Recurrent（循环）？

因为同一个函数不断重复：

第1步：

\[
h_1=f(x_1,h_0)
\]


第2步：

\[
h_2=f(x_2,h_1)
\]


第3步：

\[
h_3=f(x_3,h_2)
\]


参数完全一样：

\[
W_x,W_h
\]

不会变。


展开：

```
        W
        |
x1 ---> RNN ---> h1
              |
x2 ---> RNN ---> h2
              |
x3 ---> RNN ---> h3

```

数学上：

这是一个**参数共享的深层网络**。

---

# 7. RNN训练：为什么会有BPTT？

普通神经网络：

反向传播：

\[
\frac{\partial L}{\partial W}
\]


但是RNN：

参数重复使用：

\[
W_h
\]

影响：

\[
h_1,h_2,...,h_T
\]


所以：

loss:

\[
L
\]


对参数：

\[
W_h
\]


梯度：

\[
\frac{\partial L}{\partial W_h}
\]


需要累加所有时间：

\[
\boxed{
\sum_t
\frac{\partial L_t}{\partial W_h}
}
\]


这叫：

> Back Propagation Through Time

时间反向传播。

---

# 8. 最大数学问题：梯度消失

看：

\[
h_t=f(h_{t-1})
\]


连续展开：

\[
\frac{\partial h_t}{\partial h_1}
=
\prod_{k=2}^{t}
\frac{\partial h_k}{\partial h_{k-1}}
\]


如果：

\[
\left|
\frac{\partial h_k}{\partial h_{k-1}}
\right|<1
\]


那么：

\[
\prod
\rightarrow0
\]


于是：

早期信息消失。


例如：

一句话：

> "The movie that I watched yesterday which was directed by ... was amazing"


RNN读到：

amazing

的时候：

前面的movie信息可能已经丢失。

---

# 9. LSTM为什么出现？

LSTM的核心数学思想：

> 不让信息每次都经过非线性压缩，而增加一条近似线性的记忆通道。


普通RNN：

\[
h_t=f(h_{t-1},x_t)
\]


LSTM：

增加：

\[
c_t
\]

cell state。


核心：

\[
\boxed{
c_t
=
forget \times c_{t-1}
+
input \times new
}
\]


即：

\[
c_t=f_tc_{t-1}+i_t\tilde c_t
\]


这里：

- \(f_t\)：忘记多少
- \(i_t\)：写入多少


如果：

\[
f_t\approx1
\]


梯度可以长期传播。


---

# 10. RNN和Transformer的本质区别

这是理解Attention的关键。


## RNN：

强制：

\[
x_1\rightarrow x_2\rightarrow...\rightarrow x_T
\]


信息必须经过：

\[
h_1,h_2,...h_T
\]


路径：

\[
O(T)
\]


---

## Transformer：

直接：

\[
x_i
\rightarrow x_j
\]


通过：

\[
Attention(Q,K,V)
\]


任意两个token直接交流。


路径：

\[
O(1)
\]


所以长距离依赖强很多。


---

# 11. 从更高数学视角总结

RNN可以看成：

## (1) 非线性动力系统

\[
h_t=F(h_{t-1},x_t)
\]


---

## (2) 状态空间模型

类似：

\[
\begin{cases}
h_t=f(h_{t-1},x_t)\\
y_t=g(h_t)
\end{cases}
\]


---

## (3) 一个学习到的有限维记忆机

把：

\[
(x_1,\dots,x_t)
\]


压缩成：

\[
h_t\in R^d
\]


---

## (4) 参数共享的无限深网络

展开以后：

\[
f\circ f\circ f\circ...
\]


---

一句话总结：

> **RNN的数学思想，就是用一个可学习的状态变量 \(h_t\)，递归地压缩过去的信息，使序列问题变成一个动态系统；它本质上是在学习“如何更新记忆”。**

从这个角度看，后面的 LSTM、GRU、Transformer，其实都是在回答同一个问题：

**如何设计一个更好的记忆机制。**