# 第 6 章　Actor–Critic 与 GAE：如何估计“比预期好多少”

策略梯度需要一个好用的优势估计。完整回报容易受远处随机事件影响，一步 TD 又高度依赖价值预测。Actor–Critic 把行动选择与价值评价分工：actor 是策略 $\pi_\theta$，critic 是价值函数 $V_\phi$，参数分别为 $\theta$ 与 $\phi$。GAE 把不同长度的 TD 信息混合，让我们调节对采样结果与价值预测的依赖。

在 `start` 选择投入后，虽在最后一步拿到 2 分，整条路线的折扣回报却是 1.6；还要判断它比从 `start` 通常能得到的分数好多少。**优势**就是“这次动作的回报相对当前状态平均水平的差”。第 5 章用回报减去基线；本章让 critic 估计基线，再用广义优势估计（GAE）混合不同长度的时序差分（TD）误差。先看一步，再看多步。

在当前小环境中，若策略以一半概率退出、一半概率投入，并按最优方式完成后续步骤，`start` 的平均价值是 $(1+1.6)/2=1.3$。投入动作的价值 1.6 比平均高 0.3；这个 0.3 就是其优势。真实任务中无法直接列出所有行动价值，critic 才要从数据估计这个“通常水平”。它是估计器，不是另外一个替模型执行动作的环境。

## 6.1 先理解一步 Actor–Critic

定义 $b_t=1-\mathbf1\{\text{第 }t\text{ 步真正终止}\}$，$\mathbf1$ 是指示函数。一步 TD 残差为

$$
\delta_t=R_{t+1}+\gamma b_tV_\phi(S_{t+1})-V_\phi(S_t).
$$

若 $V_\phi=V^\pi$，数据来自同一策略，给定 $(S_t,A_t)$ 后对下一步取期望，就得到 $\mathbb E[\delta_t\mid S_t,A_t]=A^\pi(S_t,A_t)$。因此它能用于策略更新。如果 critic 不准确，下一状态价值误差会进入优势估计，这就是自举偏差的来源。

critic 可以拟合停止梯度后的目标 $R_{t+1}+\gamma b_tV_\phi(S_{t+1})$，actor 用 $-\log\pi_\theta(A_t\mid S_t)\operatorname{stopgrad}(\delta_t)$ 更新。在同一个网络共享骨干时，两种损失仍有不同语义，权重不当可能让价值拟合干扰策略学习。

这里写的是单步损失项，汇总时还需匹配目标的采样或加权约定。若像第 5 章那样优化从初始状态出发的折扣回报，并逐回合枚举时间步，应保留外层 $\gamma^t$ 权重；若按归一化折扣占用分布采样，该权重已包含在采样分布中。不能一边均匀汇总所有时间步，一边无条件声称得到完全相同的折扣目标梯度。

## 6.2 从多步残差到 GAE

先忽略轨迹边界。把连续 $n$ 步的 TD 残差加起来，内部价值项望远镜抵消：

$$
\sum_{l=0}^{n-1}\gamma^l\delta_{t+l}
=\sum_{l=0}^{n-1}\gamma^lR_{t+l+1}+\gamma^nV_\phi(S_{t+n})-V_\phi(S_t).
$$

所以短窗口依赖更多 critic，长窗口依赖更多实际奖励。广义优势估计（Generalized Advantage Estimation，GAE）再引入 $\lambda\in[0,1]$，对残差几何加权：

$$
\widehat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}.
$$

$\lambda=0$ 只保留一步残差；完整终止回合取 $\lambda=1$ 时得到 $G_t-V_\phi(S_t)$。后者即使基线不准确，减去状态基线本身也不改变策略梯度期望；但如果轨迹在中途截断并用不准确的尾部价值自举，仍可能引入偏差。若 critic 就是真价值，在相应可积和 on-policy 条件下，不同 $\lambda$ 的优势条件期望都正确；现实中通常用较小 $\lambda$ 交换偏差与方差。[GAE 原论文](https://arxiv.org/abs/1506.02438)系统分析了这类估计。

## 6.3 边界需要两个掩码

价值是否自举与优势递推是否跨到下一条数据，是两个问题。再定义 $c_t=1$ 当且仅当数组中的下一步仍属于同一段轨迹，且这里没有终止、截断或采样批次结束。反向递推为

$$
\widehat A_t=\delta_t+\gamma\lambda c_t\widehat A_{t+1},
\qquad \widehat V_t=\operatorname{stopgrad}(\widehat A_t+V_\phi(S_t)).
$$

$\widehat V_t$ 是 critic 的一个常用回归目标。真正终止时 $b_t=0,c_t=0$；外部截断时 $b_t=1,c_t=0$，保留最后状态价值，但不能把下一回合的优势接过来；非终止的采样批次末尾也自举，而递推在该批结束。若保留下一段连续轨迹，可以进一步扩展窗口，但必须确认轨迹身份一致。

```text
收集一批轨迹，并冻结这批数据的旧 logprob 与价值预测
next_advantage = 0
按时间反向遍历每一条轨迹：
    delta = reward + γ * bootstrap_mask * value_of_actual_next_state - value
    advantage = delta + γ * λ * continuation_mask * next_advantage
    value_target = advantage + value
    next_advantage = advantage
actor 用停止梯度的 advantage；critic 拟合停止梯度的 value_target
```

这里的 `actual_next_state` 是环境重置前的最后状态。[官方时间限制说明](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)解释了截断仍需自举的原因。仅使用一个 `done` 掩码很容易既清错尾部价值，又串接不同回合。

## 6.4 两步手算与一个截断反例

取奖励 $[0,1]$、价值预测 $[0.4,0.5,0]$、$\gamma=0.9$、$\lambda=0.8$，第二步真正终止。于是 $\delta_0=0+0.9\times0.5-0.4=0.05$，$\delta_1=1-0.5=0.5$，优势为 $\widehat A_1=0.5$、$\widehat A_0=0.05+0.72\times0.5=0.41$。对应价值目标为 $[0.81,1]$。若取 $\lambda=1$，第一步优势为 $0.5$，价值目标为实际回报 0.9。

另有一条在采样上限停止的转移，奖励 0，当前价值 1，最后状态价值 2。正确 TD 残差是 $0+0.9\times2-1=0.8$；误当终止则得到 $-1$，连改善还是变差的方向都反了。

## 6.5 实验判断与后训练连接

critic 误差、优势方差和最终回报应共同观察。解释方差（explained variance）常写为 $1-\operatorname{Var}(\widehat V-V_\phi)/\operatorname{Var}(\widehat V)$；目标方差接近零时该指标不稳定。高解释方差不证明奖励定义合理，也不证明 actor 更新安全。

语言模型 PPO 往往对每个回答 token 预测价值，再由末尾奖励与逐 token 奖励构造 GAE。padding 不是状态转移，提示词 token 通常不承担回答策略损失。最大生成长度的处理必须与任务目标一致；如果明确把超长视为任务失败，就应按该任务定义设置终止奖励，不能机械套用任何一种掩码。

## 6.6 练习与答案提示

1. 上述两步例子取 $\lambda=0$，价值目标是什么？提示：$[0.45,1]$。
2. 为什么外部截断处不能让 $c_t=1$ 接到重置后的回合？提示：GAE 求和要求后续残差来自同一条条件轨迹。
3. 若 actor loss 不 detach 优势，会多出什么？提示：乘积求导中的 $\log\pi\nabla\widehat A$，它不是所推导的策略梯度项。

## 原始资料

主读 [Schulman 等，GAE](https://arxiv.org/abs/1506.02438)，对照第 3 章 TD 与第 5 章基线推导。本章给出的边界伪码是显式区分终止、外部截断和数组边界的教学写法，迁移到具体框架前应核对其返回的最后观测语义。
