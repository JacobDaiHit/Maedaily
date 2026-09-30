# 第 5 章：RLHF 与 PPO 训练系统

> 本章目标：画出一次 rollout 到参数更新的数据流，分清 reference、old 与 current policy。前置：[策略梯度](../../RL/chapters/05_策略梯度与方差缩减.md)、[GAE](../../RL/chapters/06_ActorCritic与GAE.md)、[PPO 理论](../../RL/chapters/07_TRPO与PPO.md)。

前两章使用固定的偏好对；这里让当前模型真的生成新回答，再根据得到的分数更新。一次“给出题目 → 模型采样回答 → 评估回答”的过程叫 **rollout**，中文可理解为一次实际采样。**RLHF** 是利用人类反馈训练模型的一类流程；这里的反馈通常先被整理成奖励模型，再进入强化学习。PPO 是限制一次策略更新幅度的具体方法。先沿数据流认清模型角色，再读损失函数。

例如输入仍是“1+1 等于几？”，旧策略采样出“2”，奖励模型给出分数；训练器再问“当前策略现在给刚才这个回答多少概率”。旧策略负责说明样本**从哪里来**，当前策略是**正在更新谁**，参考策略说明**不希望偏离哪份起点模型**。这三个角色即使偶尔权重相同，也要在日志和公式中分别命名。

## 5.1 从一个 loss 到一个循环

在线 RLHF 的困难常不在某行公式，而在多个模型与数据版本必须一致。策略生成回答；奖励模型评价回答；价值模型估计前缀之后的回报；参考策略提供行为偏移的锚点；优化器在采样数据上更新参数。更新后的策略又改变下一批回答的分布。[InstructGPT 方法](https://arxiv.org/html/2203.02155v1)

逻辑上这些角色不同，物理上可能共享骨干、使用独立 head、分时加载或分布在不同设备。不能仅凭“用了 PPO”就断言一定同时存四个完整模型，也不能忽略这些角色各自的计算和状态。

## 5.2 三种策略的身份证

| 记号 | 含义 | 更新时机 | 用途 |
| --- | --- | --- | --- |
| $\pi_\theta$ | 当前被训练策略 | 每个优化步 | 计算新 log-prob 与梯度 |
| $\pi_{old}$ | 生成本批 rollout 的行为策略快照 | 收集新一批数据时刷新 | 重要性比率的分母 |
| $\pi_{ref}$ | 约定的参考策略，常从 SFT 初始化 | 常固定；若改动须说明算法 | 控制对训练起点的偏移 |

因此 old 与 reference 即使在第一轮权重相同，之后也承担不同任务。old log-prob 可以在采样时缓存，无需一直保留一份可训练 old 模型。若异步采样有多个策略版本，要记录每条样本真实对应的行为概率；否则所谓 ratio 没有正确的分母。

## 5.3 奖励、价值与优势

对生成 token $y_t$，一种常见设计把任务奖励放在最后一个有效 token，并加入逐 token 的参考惩罚：

$$
\tilde r_t=
\mathbf1[t=T]r_\phi(x,y)
-\beta\left[\log\pi_{old}(y_t\mid s_t)-\log\pi_{ref}(y_t\mid s_t)\right].
$$

这是 rollout 上的采样估计；单个 log-ratio 可以为负，只有按相应策略取期望才得到非负 KL。不同实现可能把 KL 放进 loss、采用不同估计器，不能把所有实现写成同一种逐 token reward。

价值模型 $V_\psi(s_t)$ 估计后续回报。设终止标记为 $d_t$，则

$$
\delta_t=\tilde r_t+\gamma(1-d_t)V_\psi(s_{t+1})-V_\psi(s_t),\quad
\hat A_t=\delta_t+\gamma\lambda(1-d_t)\hat A_{t+1}.
$$

真正终止时不 bootstrap。达到长度上限的截断究竟作为终止还是仍需 bootstrap，取决于所定义任务与实现；必须与奖励、EOS 处理一致。文本生成常取 $\gamma=1$，但这是任务设计而非 PPO 定理。

## 5.4 PPO 实际限制什么

令 $\rho_t=\exp(\ell_{\theta,t}-\ell_{old,t})$，PPO 的策略 surrogate 为

$$
J_{clip}=\mathbb E_t\min\left(\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right).
$$

正优势动作概率已提高太多时，clip 分支不再奖励继续提高；负优势动作概率已降低太多时，也不再奖励继续降低。若正优势 $A=2$、$\rho=1.4,\epsilon=0.2$，贡献从 $2.8$ 截为 $2.4$。若 $A=-2$、$\rho=1.4$，取最小值为 $-2.8$，因为这是把坏动作提高太多，不能“免责”。[PPO 原始目标](https://arxiv.org/abs/1707.06347)

clip 限制的是特定 surrogate 的激励，不是把所有概率变化硬限制在区间内，更不是质量提升保证。共享参数、多轮更新与状态分布改变仍会造成大步偏移，所以需要观察实际 KL、clip fraction、梯度范数和外部质量。

## 5.5 数据形状与实现顺序

rollout 中回答 token、mask、old log-prob、reference log-prob、values、advantages 常整理为 `[B,T]`，最终序列奖励为 `[B]`。GAE 应沿每条有效回答反向计算，padding 不参与回报或归约。优势和采样时的目标一般在更新期间固定并停止梯度；不应让策略梯度经过奖励计算或 GAE 回流到采样过程。

一个清晰的循环为：采样提示 → 用行为策略生成 → 按有效长度打分 → 保存 old/ref log-prob 与 values → 计算回报和优势 → 做有限轮 minibatch 更新 → 检查 KL 与数值状态 → 刷新采样策略。价值模型另有回归损失；熵项、价值 clipping、reward whitening 都是需明确记录的实现选择。

训练日志至少包含生成 token 数、有效奖励、任务正确率、长度、KL、clip fraction、value error 和吞吐。奖励涨、正确率不涨时先审查 RM；KL 飙升时检查学习率、重复更新次数与 ratio；value error 大时检查时间对齐和终止语义。

资源测量应按阶段拆开：rollout 的逐 token 解码与 KV cache、参考/奖励模型的前向、策略与价值模型的反向，瓶颈各不相同。缩小训练 micro-batch 不一定解决生成缓存峰值；减少更新次数也不一定降低主要由采样占据的耗时。先对一个小批次记录每阶段耗时和峰值显存，再决定分时加载、缩短回答或减少候选。对本机 8 GB 资源，完整大模型 PPO 管线只作为系统设计阅读目标，真实训练从可测量的小模型或小策略开始。

## 5.6 练习与提示

1. 算 $A=-2,\rho=0.6,\epsilon=0.2$ 的 clip 目标。提示：得到 $\min(-1.2,-1.6)=-1.6$。
2. 若用 reference 替代 old 作分母，第一次更新后为何不再是上述 PPO？提示：参考不是本批采样分布。
3. 只有终点奖励的短回答，手算 $\gamma=\lambda=1$ 的 GAE。提示：望远镜求和得到回报减当前价值。

## 原始资料与下一章

- [PPO，2017](https://arxiv.org/abs/1707.06347)：surrogate 与多轮 minibatch 更新。
- [InstructGPT，2022](https://arxiv.org/abs/2203.02155)：LLM 场景的反馈流程及评测。
- 下一章：[GRPO 与 RLVR](06_GRPO与RLVR.md)。
