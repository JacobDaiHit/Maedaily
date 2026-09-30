# 第 7 章　TRPO 与 PPO：如何限制一次策略更新

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

## 7.6 练习与答案提示

1. 当 $\widehat A=-2,r=1.4$ 时 clip 目标是多少？提示：$\min(-2.8,-2.4)=-2.8$，坏方向仍被惩罚。
2. 平均 KL 很小是否保证每个状态的 KL 都很小？提示：一个很少出现的状态可以有很大的 KL。
3. 增加同批数据训练轮数为何可能降低真实性能？提示：状态分布近似越来越不可信，而且优势与 critic 目标仍来自旧数据。

## 原始资料

按 [TRPO，ICML 2015](https://proceedings.mlr.press/v37/schulman15.html) → [GAE](https://arxiv.org/abs/1506.02438) → [PPO](https://arxiv.org/abs/1707.06347)的顺序阅读。先用本章两个正负优势例子验证实现，再讨论优化器、并行环境与训练吞吐。
