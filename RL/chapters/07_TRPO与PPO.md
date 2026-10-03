# 第 7 章　TRPO 与 PPO：如何限制一次策略更新

> **前置知识：**策略梯度、GAE、概率比与 KL。
>
> **本次必读：**old/current、重要性比、clip 分支与 PPO 更新过程；第 1–3 题。
>
> **第二遍选读：**自然梯度、信赖域及平台反例；第 4–6 题。
>
> **运行命令：**在仓库根目录执行 `python scripts/run_lesson.py posttrain`；先预测再查看本次目录的 `结果说明.txt`。
>
> **完成标准：**手算正负优势的 clip 分支，并说明 clip 不能直接保证真实 KL 小；题目先写出自己的答案，再展开参考答案。

> **实验范围：**`posttrain` 检查 clip 的标量分支；实际采样与更新可接后训练 `rlvr-model`，本章完整 Actor–Critic PPO 仍需按数据与价值模型契约实施。

有了优势估计，并不代表可以沿梯度走任意远。轨迹由旧策略采样，更新后策略访问的状态与行动可能改变。TRPO 用信赖域约束策略变化，PPO 用更易优化的替代目标抑制部分过大更新。本章特别区分理论中的改进界与实际程序的经验稳定性。

沿用 `start` 的投入选择：一次采样表明投入有利，不意味着下一次就该把投入概率直接推到 100%。旧数据是在旧概率下得到的，改动太大后它可能不再代表新策略会遇到的情况。**信赖域**是限制新旧策略差别的范围；TRPO 直接把这类范围写成约束，PPO 用裁剪后的目标减少部分过大的更新。两者都要先说清“差别”怎样测量。

如果旧策略选投入的概率是 0.5，新策略给 0.6，同一动作的概率比是 $0.6/0.5=1.2$；若新策略给 0.95，比率就是 1.9。PPO 的裁剪目标会限制某些样本从过大的比率中继续获得更新收益，但这不保证整体策略绝不会走远。先弄懂这个比值，后文的 `clip` 公式才有具体对象。

## 7.1 为什么旧数据上的改善未必是真改善

记旧策略为 $\pi_{\mathrm{old}}$，新策略为 $\pi_\theta$，其性能分别为 $J(\pi_{\mathrm{old}})$ 与 $J(\pi_\theta)$。沿用第 5 章归一化折扣占用分布 $d_\pi$。在有限折扣 MDP 中，性能差异恒等式为

$$
J(\pi_\theta)-J(\pi_{\mathrm{old}})
=\frac{1}{1-\gamma}\mathbb E_{s\sim d_{\pi_\theta},a\sim\pi_\theta}
[A^{\pi_{\mathrm{old}}}(s,a)].
$$

困难在于右边需要新策略的状态分布。用旧状态分布近似，并对行动分布做重要性采样，得到局部替代目标。定义概率比

$$
r_t(\theta)=\frac{\pi_\theta(A_t\mid S_t)}{\pi_{\mathrm{old}}(A_t\mid S_t)},\qquad
L^{\mathrm{PG}}(\theta)=\mathbb E_{\mathrm{old}}[r_t(\theta)\widehat A_t].
$$

要求旧策略对新策略可能选的行动有支持，否则分母为零。这个行动比率并没有完整校正状态分布变化；它在旧策略附近的价值最清楚。实践中按 rollout 时间步取平均，还需与第 5 章的折扣加权约定区分。

## 7.2 TRPO 的理论动机与实际近似

KL 散度衡量两个分布的差别：$D_{\mathrm{KL}}(p\|q)=\sum_a p(a)\log[p(a)/q(a)]$，它非负但不对称。实际 TRPO 近似求解

$$
\max_\theta L^{\mathrm{PG}}(\theta)
\quad\text{使}\quad
\mathbb E_{s\sim d_{\mathrm{old}}}
D_{\mathrm{KL}}(\pi_{\mathrm{old}}(\cdot\mid s)\|\pi_\theta(\cdot\mid s))\le\delta,
$$

其中 $\delta>0$ 是更新预算。局部二阶展开 KL 得到曲率矩阵，再近似求自然梯度方向，配合线搜索寻找合适步长。理论分析使用控制所有状态差异的界与精确优势等条件；实际算法采用样本平均 KL、有限数据、近似求解，这些步骤并不自动继承严格单调改进保证。[TRPO 原论文](https://proceedings.mlr.press/v37/schulman15.html)也明确区分理论方案与实用近似。

## 7.3 PPO clip 的准确含义

PPO 的一个常用版本最大化裁剪目标

$$
L^{\mathrm{CLIP}}(\theta)=\mathbb E_{\mathrm{old}}\left[
\min\left(r_t\widehat A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\widehat A_t\right)\right],
$$

其中 $\epsilon>0$ 为裁剪范围，$\operatorname{clip}$ 把输入压到给定区间。当优势为正，继续把比率增到 $1+\epsilon$ 以上不再得到该项的额外奖励；当优势为负，继续把比率减到 $1-\epsilon$ 以下也不再得到额外奖励。反方向的坏更新仍会被惩罚。

取 $\epsilon=0.2$。若 $\widehat A=2,r=1.3$，普通项为 2.6，裁剪项为 2.4，取最小得到 2.4。若 $\widehat A=-2,r=0.7$，两个项分别为 -1.4 与 -1.6，结果为 -1.6。若优势为正而 $r=0.7$，结果仍是 $0.7\widehat A$，梯度并不会因为落在区间外就必定消失。

clip 不会把真实概率比强制锁在区间里。共享参数、其他样本和多轮更新仍可把比率推得更远。因此 PPO clip 既不是硬 KL 约束，也不保证真实回报单调提高。[PPO 原论文](https://arxiv.org/abs/1707.06347)提供算法与控制任务实验，应把其经验表现与理论保证分开阅读。

## 7.4 一批数据怎样被重复使用

```text
用当前策略收集 rollout，保存旧 logprob、奖励、终止信息和价值
用固定的旧价值预测计算 GAE 与 critic 目标
重复 K 轮，每轮打乱并划分 minibatch：
    重新计算当前策略 logprob
    ratio = exp(new_logprob - old_logprob)
    actor_loss = -mean(min(ratio*A, clip(ratio)*A))
    加上 critic 回归损失，可选加入熵奖励
    做梯度更新，记录 KL、裁剪比例与梯度范数
    若预设 KL 阈值触发，则提前结束本批更新
丢弃这一批数据，使用更新后的策略重新采样
```

旧对数概率在整个批次内必须固定。若每次更新后把分母换成当前策略，概率比会被重置到接近 1，失去相对采样策略的含义。近似 KL 的单批样本估计可以波动；熵、KL、优势的缩放与序列长度处理也必须记录。KL early stopping 是额外工程控制，不是 clip 公式自带的保证。

## 7.5 与语言模型后训练的连接

语言模型 PPO 至少有三种策略角色：正在训练的策略、采样时冻结的旧策略、提供行为约束的参考策略。PPO 概率比的分母是旧策略；RLHF 奖励中的 KL 正则常相对参考策略。两者可以暂时参数相同，但目的不同。价值模型估计未来训练奖励，奖励模型评价回答，两者也不能混叫“critic”。

长回答常按 token 计算比率，再按某种序列或 token 规则汇总。整个回答的概率比是各 token 比率的乘积，可能极端不稳定；因此不能把 token 目标与序列重要性采样直接等同。

## 7.6 深入：自然梯度的局部解与 clip 平台

在旧参数附近令 $\Delta\theta$ 为步长，surrogate 一阶变化为 $g^\top\Delta\theta$，KL 二阶近似为 $\frac12\Delta\theta^\top F\Delta\theta$，F 为对应 Fisher/局部 KL Hessian。若 F 正定、g非零，拉格朗日求解得到

$$\Delta\theta=\sqrt{\frac{2\delta}{g^\top F^{-1}g}}F^{-1}g.$$

这个方向考虑分布曲率，实际大模型不能显式求逆，常用 Hessian-vector product、共轭梯度、阻尼和线搜索。F 奇异时要处理不可识别方向与正则化；局部二阶近似不保证大步仍满足真实约束。

二动作 Bernoulli 策略 $p=\sigma(\theta)$、奖励[0,2]，旧p=.5，则 g=.5、F=.25。delta=.005给出 $\Delta\theta=.2$；新p≈.549834，真实 $KL(old\|new)=\log\cosh(.1)\approx.004991689$。小例子中近似很准，必须仍检查真实KL和surrogate，而非假设所有模型都如此。

PPO 平台反例：旧两动作等概率，优势分别为[1,-1]，epsilon=.2。新p(好)=.9时比率[1.8,.2]，好坏两项都进入改善方向的平坦分支；clip目标为 $.5(1.2-.8)=.2$，p=.6也达到同一个目标。真实KL(old‖new)在p=.9时约.510826，远大于小更新。目标不再奖励继续移动，不等于禁止移动或给唯一小步解。

实现验收先固定 old 概率、优势和mask，逐分支检查标量及梯度；再在小策略做真实采样与多轮更新。监控旧状态分布上的平均KL、重要桶KL、熵、ratio与训练外回报；旧状态上正常也不能排除新访问状态失效。

## 7.7 完整自测与面试问答

### 题 1：PPO 对负优势怎样裁剪？

**题目：**单样本估计优势 $\widehat A=-2$，概率比 $\rho=\pi_{new}(a|s)/\pi_{old}(a|s)=1.4$，裁剪宽度 $\epsilon=0.2$。要最大化的目标为 $J=\min[\rho\widehat A,\operatorname{clip}(\rho,0.8,1.2)\widehat A]$。求两项、J 和用于最小化的 actor loss，并解释更新方向。

<details>
<summary>展开参考答案（先独立作答）</summary>

**参考答案：**两项为 $1.4(-2)=-2.8$ 与 $1.2(-2)=-2.4$，取最小得到 $J=-2.8$，actor loss 为 2.8。负优势动作的概率反而上升，属于变差方向，目标仍惩罚；不能把所有区间外样本都当作无梯度。

</details>

**面试追问：**clip 会强制实际概率比留在区间吗？

<details>
<summary>展开追问回答</summary>

**追问回答：**它裁剪目标里的激励，参数仍由许多样本共同更新，实际概率比可能越界。还需监控实际 KL、裁剪比例与质量；PPO 不保证每轮回报增加。

</details>

### 题 2：平均 KL 小能保证每个状态的变化都小吗？

**题目：**KL 散度衡量两个动作分布的差异。假设按旧策略状态分布统计，两类状态权重为 0.999 与 0.001，对应真实 KL 为 0 与 10。求平均 KL，并说明一个阈值为 0.02 的平均检查能否保证第二类状态变化小。

<details>
<summary>展开参考答案（先独立作答）</summary>

**参考答案：**加权平均为 $0.999\times0+0.001\times10=0.01$，通过 0.02 阈值，但第二类状态 KL 仍为 10。平均控制不等于逐状态最坏情况控制；少见且重要的状态需要分桶或独立检查。

</details>

**面试追问：**TRPO 与 PPO 的更新控制有什么区别？

<details>
<summary>展开追问回答</summary>

**追问回答：**TRPO 使用带 KL 约束的局部优化思路，并在条件下讨论改进界；常见 PPO 使用裁剪代理目标和一阶优化，工程上更简单。平均 KL 的采样估计与额外提前停止仍有各自边界。

</details>

### 题 3：为什么同一批 rollout 不能无限重复训练？

**题目：**一批数据由冻结的旧策略采样，保存 old log-prob、奖励与优势。训练者把 PPO minibatch 轮数从 4 改成 100，期间不重新采样。请说明可能提高训练目标却降低真实回报的原因，应该监控什么，以及 old log-prob 是否应每轮刷新。

<details>
<summary>展开参考答案（先独立作答）</summary>

**参考答案：**策略逐渐偏离采样策略，旧状态分布、优势估计和数据覆盖对当前策略可能越来越不合适，还可能过拟合有限样本。应监控相对采样策略的 KL、裁剪比例、梯度和独立回报，按预设规则结束本批更新，再重新采样。分母必须保留采样时的 old log-prob，不能每轮改成当前策略。

</details>

**面试追问：**old policy 和 reference policy 有什么区别？

<details>
<summary>展开追问回答</summary>

**追问回答：**Old 表示这批数据的生成分布，用于概率比；reference 是偏移约束的参照，通常跨批固定。即使初始参数相同，职责也不同。

</details>

### 题 4：推导自然梯度并检查真实 KL

**题目：**求解一阶surrogate加二阶KL约束，写出上述方向；用g=.5、F=.25、delta=.005求步长、新概率与真实KL。提交解析和float64核对。

<details>
<summary>展开参考答案（先独立作答）</summary>

**参考答案：**步长.2，新p≈.549834，真实KL≈.004991689；推导来自 $g=\lambda F\Delta\theta$ 再用约束定尺度。F奇异、有限样本和非局部步长需阻尼或线搜索，不能直接继承精确解保证。

</details>

**面试追问：**Euclidean 梯度和自然梯度总同方向吗？

<details>
<summary>展开追问回答</summary>

**追问回答：**标量例子方向相同，多维F会按分布曲率旋转/缩放，通常不同。

</details>

### 题 5：构造 PPO 平台但 KL 很大的分布

**题目：**使用第7.6节两动作例子，比较新好动作概率.6与.9的clip目标和KL，证明clip不是硬信赖域。

<details>
<summary>展开参考答案（先独立作答）</summary>

**参考答案：**两者目标均.2；KL分别约.020411与.510826。目标存在平台，其他样本、共享参数或优化器状态仍可推动策略远移，应独立监控KL和质量。

</details>

**面试追问：**KL early stopping 能恢复理论单调改进吗？

<details>
<summary>展开追问回答</summary>

**追问回答：**它是采样与工程控制，还存在估计和状态分布近似，不自动恢复理论全部条件。

</details>

### 题 6：手写 PPO loss 与版本不变量

**题目：**实现输入new_logp、old_logp、advantage、valid、epsilon的核心loss，旧量detach；覆盖后训练第5章四分支。再进行多轮更新，证明old缓存不变并记录新采样策略版本。

<details>
<summary>展开参考答案（先独立作答）</summary>

**参考答案：**ratio=exp(new-old.detach)，取minimum后按明确有效token或样本规则归约，advantage停止梯度。无更新时ratio≈1；old整个批次固定，更新后再采样刷新。日志同时含KL、clip定义、回报与预算，防止公式正确而版本错配。

</details>

**面试追问：**把mask乘在exp之后一定安全吗？

<details>
<summary>展开追问回答</summary>

**追问回答：**无效位置若先产生inf/NaN，乘零仍可能NaN；应在计算前处理无效差值并确保有效输入有限，检查边界。

</details>

## 原始资料

按 [TRPO，ICML 2015](https://proceedings.mlr.press/v37/schulman15.html) → [GAE](https://arxiv.org/abs/1506.02438) → [PPO](https://arxiv.org/abs/1707.06347)的顺序阅读。先用本章两个正负优势例子验证实现，再讨论优化器、并行环境与训练吞吐。
