# Maedaily：用中文学会训练、后训练与工具交互

这个项目帮助你从“能看懂一个例子”走到“能训练、能解释、能排错”。第一遍读中文精讲，照着手算并运行小实验；第二遍进入系统章节和论文省流；第三步再完成真实模型项目。核心讲解和练习答案都在仓库里，不要求先读英文原书。

主线是：**训练基础 → 语言模型 → 监督微调（SFT）/ 直接偏好优化（DPO）→ 强化学习 / 可验证奖励强化学习（RLVR）→ Agent 与多步交互训练**。SFT 学示范答案，DPO 学回答之间的偏好，RLVR 用可执行的规则给生成结果评分。Agent 指能依据工具反馈继续选择动作的系统。循环神经网络（RNN）、激活函数、预训练与系统知识按需要接入。教学内容更新：2026-09-29。

## 第一次来，今天就这样开始

先打开 [第一课：从预测到学习](入门/第一课_从预测到学习.md)。它用输入 2、目标 5 的小例子，解释模型、参数、损失、梯度和学习率，并逐步算出一次更新。

然后打开 PowerShell，逐行执行：

```powershell
Set-Location 'D:/github_book/Maedaily'
python scripts/run_lesson.py --check
python scripts/run_lesson.py first
```

第一行换成自己的项目路径。`--check` 检查已有 Python 和 PyTorch；`first` 只用 Python 标准库，即使还没准备好 PyTorch 也能先做。本机入口能自动找到已验证的 PyTorch 环境，不需要先安装新包。遇到找不到命令、`>>>`、缺少 torch 等情况，按 [环境准备与常见问题](入门/环境准备与常见问题.md) 处理。

第一课正常输出的关键数字是：

```text
更新前：预测=2，损失=4.5
手算梯度：权重梯度=-6，偏置梯度=-3
更新后：权重=1.6，偏置=0.3
更新后：预测=3.5，损失=1.125
```

今天的完成标准是：能解释这四行，并独立算出学习率改成 0.05 后的结果。算不出就回讲义对应步骤，先把这一个例子吃透。

## 第一遍：按这条中文路线学

| 顺序 | 中文精讲 | 你会实际学到什么 | 配套操作 |
| --- | --- | --- | --- |
| 1 | [第一课：从预测到学习](入门/第一课_从预测到学习.md) | 一次参数更新、概率、负对数、折扣回报 | `first` |
| 2 | [张量与训练框架](pytorch&mindspore/从零上手.md) | 形状、广播、自动微分、训练/验证、保存恢复 | `linear` |
| 3 | [激活与序列模型补课](入门/激活与序列模型补课.md) | 非线性、三步 RNN、门控、为什么需要 attention | 手算与文内短代码 |
| 4 | [Transformer 从零上手](transformer/从零上手.md) | token 到向量、三位置注意力、mask、标签移位、小 LM | `lm` |
| 5 | [后训练从零上手](post_train/从零上手.md) | 同一道题的 SFT / 偏好 / RLVR 数据、DPO 与组优势 | `posttrain` |
| 6 | [强化学习从零上手](RL/从零上手.md) | 状态、策略、回报、Bellman、MC/TD/Q、策略梯度与 PPO | `rl` |
| 7 | [工具交互第一课](入门/工具交互第一课.md) | 请求/观察/重试、检索、轨迹 mask、奖励与调用预算 | `agent`、`trajectory` |

最后一列是课程代号，例如运行 `python scripts/run_lesson.py linear`。如果完整实验还显得长，先做三个独立的短程序路线：[训练框架三步](pytorch&mindspore/短程序实验.md)、[RL 三步](RL/短程序实验.md)、[后训练四步](post_train/短程序实验.md)。每次只预测并核对一个结果，之后再运行完整课程。后训练第一次先学 SFT、偏好和具体数值；遇到在线 RL 的概念，再去 RL 精讲补回报与策略梯度，然后返回 GRPO。DPO 的完整推导可以第二遍再读。

[入门目录](入门/README.md) 给出了每天的学习闭环和 [两周起步安排](入门/README.md#两周起步安排)。每个“学习日”可以拆成多天，以能完成练习为准。单次建议 45–90 分钟：读一个例子 → 合上文档手算 → 运行核对 → 改一个条件 → 写五句话复盘。用 [学习记录模板](入门/学习记录模板.md) 保留过程。

## 实验怎样运行，结果怎样看

全部课程通过一个入口运行：

```powershell
python scripts/run_lesson.py --list
python scripts/run_lesson.py linear
python scripts/run_lesson.py lm
python scripts/run_lesson.py rl
python scripts/run_lesson.py posttrain
python scripts/run_lesson.py agent
python scripts/run_lesson.py trajectory
```

只运行正在学的那一课即可。它们不需要 API 密钥或模型下载；`linear`、`lm` 使用 PyTorch 和 CPU，其余只用 Python 标准库。本机指定 PyTorch 环境的写法是：

```powershell
python scripts/run_lesson.py lm --python 'D:/anaconda/envs/pytorch_env/python.exe'
```

每次自动创建一个独立的 `study_runs/时间_课程名/`，终端会打印位置。先看其中的 **`结果说明.txt`**，再看 `console.txt` 完整日志。`run.json` 记录命令与是否通过；训练类课程还有 `summary.json` 和 `learning_curve.csv`。个人运行目录已加入 Git 忽略，各实验目录中原有的教学参照结果另行保留。

指定输出位置可以用 `--output-dir 'study_runs/第一次线性训练'`；目录已有内容时会停止，避免把上次结果覆盖掉。

| 课程 | 默认配置的重要结果 | 怎样理解 |
| --- | --- | --- |
| `first` | 损失 4.5 → 1.125 | 手写解析梯度完成一次更新 |
| `linear` | 学到权重约 `[2,-3]`、偏置 0.5；恢复下一步误差 0 | 真实 PyTorch 训练；遗漏动量会改变下一步更新 |
| `lm` | lr=0.001 的验证损失约 1.053525；lr=0.003 约 1.187625 | 同一初始化和预算下，训练更低不一定验证更好 |
| `rl` | 起点价值 1.6，选择先投入再完成 | 表格 Q-learning 与独立手算一致 |
| `posttrain` | 45 项数值检查；DPO loss 约 0.554355 | 检查概率、掩码、偏好、组优势和验证器 |
| `agent` | 41 项检查；正常计算和暂时故障恢复都得到 4 | 检查规则控制器的工具流程 |
| `trajectory` | 正确动作 mask 的损失 0.25 | 工具观察不应混入策略动作损失 |

不同机器的耗时和浮点末位可能不同，先看脚本检查是否通过和结果方向是否一致。微型 LM 是真实的数字语法模型训练；后训练数学、工具状态机与轨迹核算还不包含真实 LLM 的 SFT / DPO / RLVR 训练，这些继续作为年度项目完成。

## 教材里的重要内容，已经整理到哪里

| 参考材料 | 已整理的关键内容 | 中文学习入口 |
| --- | --- | --- |
| 《动手学深度学习》官方中文版 | 数据形状、反向传播、线性回归、非线性、RNN、attention | 上面的框架、序列补课与 Transformer 精讲 |
| 《Deep Learning》 | 概率、损失、梯度、优化与泛化的区别 | 第一课与框架精讲的原创算例 |
| 《Reinforcement Learning: Theory and Algorithms》 | 第 1 章策略、价值、Bellman、价值迭代 | RL 精讲，附实际读过的 PDF 页码 |
| Lambert《Reinforcement Learning from Human Feedback》 | SFT、奖励模型、策略梯度、GRPO、DPO | 后训练精讲，附章节、页码、完整手算与答案 |
| 《Speech and Language Processing》第三版草稿 | 检索与生成的分工、证据怎样进入回答 | 工具交互第一课与 Agent 章节 |

这些内容用中文重新讲解，配本项目设计的例子和代码。原书负责来源，精讲负责让你在项目内学明白。更完整的对应关系、官方中文教材链接和第二遍选读位置见 [教材重点中文索引与书籍地图](references/参考书籍与课程地图.md)。

## 第二遍：八个目录各自怎么用

| 目录 | 第二遍用途 | 建议阶段 |
| --- | --- | --- |
| [激活函数](激活函数/README.md) | 从例子理解非线性，再查导数、饱和、门控、残差与归一化 | 第 1 月按缺口补 |
| [pytorch&mindspore](pytorch&mindspore/README.md) | 4 章把计算契约、梯度、训练和排错讲完整 | 第 1 月；先 PyTorch，MindSpore 对照选读 |
| [RNN](RNN/README.md) | 三步递推导读、原笔记及梯度、门控与 attention 起点 | 第 1 月作序列基础 |
| [transformer](transformer/README.md) | 5 章覆盖结构、LM、缓存、数据与系统 | 第 1 月前两章，第 7–8 月数据/系统 |
| [RL](RL/README.md) | 8 章从 MDP 到 PPO、探索与离线 RL | 第 4 月主读，前期按需要补 |
| [post_train](post_train/README.md) | 7 章从 SFT 到 DPO、RLHF、RLVR 与评测 | 第 2–6 月主线 |
| [agent_design](agent_design/README.md) | 3 章讲工具、记忆、检索、规划与可靠性 | 为交互训练准备系统基础 |
| [agentic_rl](agentic_rl/README.md) | 3 章讲多步奖励、信用分配、训练系统与实验设计 | 第 9–10 月可选专题 |

正文中的公式和条件是第二遍重点。每章至少能解释一个公式、定位一段代码、说明一个失败条件，然后再扩展规模。[发展脉络](references/学习路线与发展脉络.md) 解释方法为何出现；[年度任务与验收](references/年度学习任务与验收.md) 把它们连到项目 A/B/C。术语、论文与版本记录可从[阅读附录](references/README.md)按问题查阅。本地详细目标保存在 [GOAL](GOAL.md)。

## 什么时候开始读论文

能读懂对应中文精讲、完成手算和小实验后，先看 [20 篇论文的省流索引](references/论文索引.md)。每篇都有中文说明：发表身份与阅读价值、解决什么问题、具体实验、已展示应用、限制，以及对当前项目的用途。

首次只选与当前问题有关的一篇，例如 attention 对应 Transformer，偏好目标对应 DPO，组内奖励对应 DeepSeekMath。公式不懂回本地章节，实验结论不清楚查省流中的基线与预算；英文 PDF 作为原始证据保存，无需为了跟进新论文先通读全部原文。

经典与近期论文是精选，检索截止 2026-09-26，最新收录稿为 2026-09-23，不自动更新。常用词可查 [中文术语表](references/术语与符号.md)，新论文的判断方法见 [证据标准](references/阅读方法与证据标准.md)。

## 学到什么程度，算真正完成一阶段

- **读懂：** 能用自己的话解释定义，独立复算例子。
- **会做：** 能找到关键代码，改一个变量前先预测，修改后解释结果。
- **会查错：** 能定位标签、形状、mask、梯度或评测中的一类错误。
- **能做项目：** 在真实数据和模型上建立基线、训练、评测与消融，保留失败记录。

每阶段的具体交付见年度验收；周投入暂按 10–15 小时安排，尚未确认。运行脚本成功是第一步，真实模型训练与独立实验能力需要接着完成。

## 资料和维护入口

目前有 32 个系统教学章节、上述新手中文精讲、20 篇论文省流及 PDF、2 本公开参考书稿。出处与版本见 [资料清单](references/readings.json)，校验记录见 [下载记录](references/downloads.json)，实跑情况见 [版本与验证记录](references/版本与验证记录.md)。

只有需要检查或补齐原始文件时，才运行下面的资料命令；它不是学习第一步：

```powershell
python scripts/download_readings.py --verify-only
python scripts/download_readings.py
```

已匹配的 PDF 不会重复下载。正文为原创中文教学整理，原始资料保留各自许可与权利。新增论文按 [省流模板](references/论文省流模板.md) 记录出处、版本和证据，避免只增加文件数量而没有学习内容。
