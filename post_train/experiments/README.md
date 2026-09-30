# 后训练数值实验：先检查公式，再连接模型

第一次动手先做[四个短程序实验](../短程序实验.md)，每次只解释一个数。下面的 45 项核算用于第二遍检查标签、梯度、归一化和边界情况。

本目录的 [posttraining_math_lab.py](posttraining_math_lab.py) 只使用 Python 标准库，在 CPU 上完成确定性的数值检查。它不会下载模型、安装依赖或调用 GPU，也不执行模型生成的代码。**这是数学与数据处理实验，不是 SFT、DPO 或 GRPO 的真实语言模型训练结果。**

## 如何运行

在仓库根目录执行：

```powershell
python .\post_train\experiments\posttraining_math_lab.py
```

若 `python` 不在当前 PATH，本机已核验的解释器可这样调用：

```powershell
& 'C:\Python313\python.exe' .\post_train\experiments\posttraining_math_lab.py
```

运行失败会抛出明确错误；成功时输出 JSON。无需随机种子，因为输入全部固定。Python 3.13.5 于 2026-09-26 实际运行验证；脚本使用的类型语法要求 Python 3.10 或更新版本。

本次实际结果：45 项检查通过。关键输出如下，完整输出还包括数值梯度、KL 最优分布和组内优势。

```json
{
  "checks_passed": 45,
  "completion_logprob": -2.302585093,
  "bt_loss": 0.313261688,
  "dpo_loss": 0.554355244,
  "pass_at_2": 0.7
}
```

## 每一项检查回答什么问题

| 实验 | 检查内容 | 对应章节 |
| --- | --- | --- |
| 标签移位与 mask | `[L,V]` logits 第 t 行预测第 t+1 个 token；提示和 padding 不参与回答 log-prob | [第 1 章](../chapters/01_语言模型到后训练.md) |
| Bradley–Terry | 奖励差为 1 的偏好概率与损失；同提示奖励平移不变 | [第 3 章](../chapters/03_偏好建模与奖励模型.md) |
| DPO | 四个序列 log-prob、初始损失 log(2)、反转偏好、有限差分核对梯度 | [第 4 章](../chapters/04_DPO推导与实现.md) |
| KL 最优策略 | 解析分布与若干候选的目标比较、KL gap 恒等式 | [第 4 章](../chapters/04_DPO推导与实现.md) |
| PPO clip | 正负优势分别检查正确的 `min` 分支 | [第 5 章](../chapters/05_RLHF与PPO训练系统.md) |
| GRPO | 混合组、全对组、全错组；总体标准差与数值保护 | [第 6 章](../chapters/06_GRPO与RLVR.md) |
| 最小验证器 | 单一整数格式、错答案、多答案、冗余文本与超长输出 | [第 6 章](../chapters/06_GRPO与RLVR.md) |
| pass@k | 组合数估计以及全对/全错边界 | [第 7 章](../chapters/07_推理评测_蒸馏与实验设计.md) |

验证器只认可形如 `Answer: 42` 的约定格式；拒绝 `Answer: 042` 是本例的格式选择，不是数学上认为 042 与 42 不等价。训练真实任务时应先写清接受哪些表达，并检查误拒与误收。

## 如何把它变成学习成果

先不运行，手算 `[-4,-6,-5,-5.5]` 四个 log-prob 在 beta=0.2 时的 DPO loss；再解释为何改变提示位置 logits 不改变回答损失。修改一处 mask 或交换 chosen/rejected，看哪个检查失败。最后对照公式，写出“形状、分母、被屏蔽项、停止梯度位置”四件事。

下一步才是接入真实 tokenizer 与小模型：用一个批次比较本脚本的序列求和约定与框架计算，再建立独立评测。标准库脚本没有反向传播、优化器、rollout 或语言能力评测，因此通过这些检查不算完成 [GOAL 中项目 B/C](../../GOAL.md)。真实 GPU 训练需要另行测量显存、采样时间和软件兼容性。
