# Maedaily：用中文学会训练、后训练与工具交互

这个项目帮助你从“能看懂一个例子”走到“能训练、能解释、能排错”。先读中文精讲、独立手算并运行小实验，再进入系统章节和论文，最后完成有数据、训练、评测与失败记录的项目。主线是 **训练基础 → 语言模型 → SFT / DPO → RLVR → Agent 与多步交互**；循环神经网络、激活、数据与训练系统按需要补入。教学内容更新：2026-10-02；参考实训收尾：北京时间 2026-10-03。

## 今天开始

打开[第一课：从预测到学习](入门/第一课_从预测到学习.md)，跟着输入 2、目标 5 的例子算一次参数更新。在自己克隆的仓库根目录执行：

```powershell
python scripts/run_lesson.py --check
python scripts/run_lesson.py first
```

`first` 只需 Python 标准库。默认学习率 0.1 的结果为：损失 `4.5 → 1.125`，权重/偏置 `1/0 → 1.6/0.3`。今天的完成标准是解释这次更新，再独立预测学习率 0.05 的结果，用 `python scripts/run_lesson.py first -- --learning-rate 0.05` 核对。运行位置与环境错误见[环境准备与常见问题](入门/环境准备与常见问题.md)。

## 按当前阶段选择

| 现在的目标 | 从这里进入 | 完成后做什么 |
| --- | --- | --- |
| 刚开始，需要先看懂和运行 | [第一遍中文路线](#第一遍按这条中文路线学)与[两周起步安排](入门/README.md#两周起步安排) | 读例子、合上手算、运行核对、改一个条件、写复盘 |
| 已会小实验，要补公式和排错 | [系统章节](#第二遍八个目录各自怎么用)与[完整题目索引](references/面试复习与题目索引.md) | 按章的前置、必读/选读、命令和完成标准推进，先作答再展开答案 |
| 要做真实训练和项目交付 | [年度任务与项目定义](references/年度学习任务与验收.md#project-a)与[模型后训练实验](post_train/experiments/模型后训练实验.md) | 从小模型实际更新进入自己的数据/预训练模型，保留基线、评测与失败证据 |

## 第一遍：按这条中文路线学

| 顺序 | 中文精讲 | 你会实际学到什么 | 配套操作 |
| --- | --- | --- | --- |
| 1 | [第一课：从预测到学习](入门/第一课_从预测到学习.md) | 一次参数更新、概率、负对数、折扣回报 | `first` |
| 2 | [张量与训练框架](pytorch&mindspore/从零上手.md) | 形状、广播、自动微分、训练/验证、保存恢复 | `linear` |
| 3 | [激活与序列模型补课](入门/激活与序列模型补课.md) | 非线性、三步 RNN、门控、为什么需要 attention | 手算与文内短代码 |
| 4 | [Transformer 从零上手](transformer/从零上手.md) | token 到向量、三位置注意力、mask、标签移位、小 LM；[Embedding 原理](transformer/chapters/01_Embedding与表示学习.md)按需展开 | `lm` |
| 5 | [后训练从零上手](post_train/从零上手.md) | 同一道题的 SFT / 偏好 / RLVR 数据、DPO 与组优势 | `posttrain` |
| 6 | [强化学习从零上手](RL/从零上手.md) | 状态、策略、回报、Bellman、MC/TD/Q、策略梯度与 PPO | `rl` |
| 7 | [工具交互第一课](入门/工具交互第一课.md) | 请求/观察/重试、检索、轨迹 mask、奖励与调用预算 | `agent`、`trajectory` |

最后一列是课程代号，例如运行 `python scripts/run_lesson.py linear`。如果完整实验还显得长，先做三个独立的短程序路线：[训练框架三步](pytorch&mindspore/短程序实验.md)、[RL 三步](RL/短程序实验.md)、[后训练四步](post_train/短程序实验.md)。每次只预测并核对一个结果，之后再运行完整课程。后训练第一次先学 SFT、偏好和具体数值；遇到在线 RL 的概念，再去 RL 精讲补回报与策略梯度，然后返回 GRPO。DPO 的完整推导可以第二遍再读。

[入门目录](入门/README.md) 给出了每天的学习闭环和 [两周起步安排](入门/README.md#两周起步安排)。每个“学习日”可以拆成多天，以能完成练习为准。单次建议 45–90 分钟：读一个例子 → 合上文档手算 → 运行核对 → 改一个条件 → 写五句话复盘。用 [学习记录模板](入门/学习记录模板.md) 保留过程。

## 运行当前课程

用 `python scripts/run_lesson.py --list` 查看课程，只运行正在学习的一课。统一入口会打印独立的 `study_runs/` 位置；先读 `结果说明.txt`，再看日志和 `artifacts/` 中的实验文件。参数变更用 `--` 传入，例如 `python scripts/run_lesson.py lm -- --seed 11`。完整目录约定、解释器选择和历史默认结果见[运行与结果说明](入门/环境准备与常见问题.md#统一入口变参实验和结果目录)。

学习顺序可以从 `lm` 进入 `lm-resume` 做完整续训，再运行 `data-asset` 核对数据来源、题族切分与重建。`posttrain` 核算后训练公式；`sft-model`、`dpo-model`、`rlvr-model` 运行小型字符语言模型的真实训练，具体输入、更新与验收见[模型后训练实验](post_train/experiments/模型后训练实验.md)。

要继续提高模型质量，进入[模型质量基线](post_train/experiments/模型质量基线.md)，先用开发集选模型并检查算术与 copy 门槛，再打开冻结评测。

## 项目 A/B/C 的参考实训与公开证据

参考实训把项目验收条目连到实际代码和保留的证据；仓库实现、一次实验的质量与学习者的独立能力分别验收。现有绿色检查不能替代自己的数据、研究、消融或答辩。

| 项目 | 参考入口 | 当前交付状态 |
| --- | --- | --- |
| A：训练与完整恢复 | [A 参考实训说明](transformer/experiments/README.md#项目-a-参考实训)；`python scripts/run_lesson.py project-a` | 2026-10-02 固定40步机制验收通过：六个独立进程、20步精确恢复、遗漏状态负控、固定两学习率曲线和CPU资源/token预算 |
| B：SFT / DPO | [A/B/C 参考实训](post_train/experiments/A_B_C参考实训.md)、[真实训练验收](post_train/experiments/真实模型训练与验收.md) | 2026-10-03 参考验收通过：固定 Qwen、三个 seed、实际 SFT/DPO、B1 消融和公共独立复算齐全 |
| C：RLVR | [真实采样与更新参考](post_train/experiments/A_B_C参考实训.md#项目-c真实分布独立-pg-与-kl)、[年度项目 C](references/年度学习任务与验收.md#project-c) | 2026-10-03 参考验收通过：独立 PG、同分布 replay、C2 两项消融、九分支评测与成本完整保留 |

A 的[紧凑公开证据](references/evidence/project_a_20261002/README.md)随仓库保留，克隆后无需访问历史 `study_runs` 即可复核：

```powershell
python references/evidence/project_a_20261002/verify.py
python references/evidence/project_a_20261002/verify.py --source-root .
```

第一条用标准库校验公开文件并独立复算数据/逐步CSV，第二条再比对本次训练源码版本。完整checkpoint由参考实训重新生成。A当前结果限定合成数字语法、单种子、CPU float64；自然文本和通用语言模型能力另需数据与评测，学习者独立复写、排错和讲解仍按[年度验收](references/年度学习任务与验收.md#project-a)完成。

B/C 先按[参考实训](post_train/experiments/A_B_C参考实训.md)准备固定 revision 的 `Qwen2.5-0.5B-Instruct`，核对[九文件清单](post_train/experiments/reference_model.json)，再运行本地离线矩阵。开发选择、三个实际种子和九个分支由公开[配置](post_train/experiments/reference_config.json)及[选择记录](post_train/experiments/reference_selection.json)固定。首轮 C 的零策略信号和早期解码隐藏特殊 token 的问题保留为失败记录；V2 虽然程序 summary 为 passed，但独立校验拒绝了源码身份不一致，只留作诊断。V3 修复并增加环境、资源、逐 token 对齐证据，真实 summary 与公开包双解释器独立复算通过。最终指标只覆盖合成算术和数字规范化，各分支的未提升、无效与负结果全部保留。项目状态按[版本记录](references/版本与验证记录.md)追加，不将工程检查通过解释成基线质量、人的能力或自然语言 benchmark 通过。

B/C 的[公共证据与复现说明](references/evidence/project_bc_20261002/README.md)包含 5,373 条逐题记录、三种子九分支、阶段/源码身份和真实预算。克隆后只用 Python 标准库复算，不需要模型权重或历史运行目录：

```powershell
python -S references/evidence/project_bc_20261002/scripts/verify_reference_run.py --verify-package references/evidence/project_bc_20261002
```

Python 3.11/3.13 的包内入口及无 ignored 目录副本均实际通过。共同 SFT 基线三种子主任务 `34/34`、数字规范化 `45/45`；后续分支迁移累计仅多一两条正确记录且区间含零，见[全部结果与成本](post_train/experiments/A_B_C参考实训.md#v3-真实结果与公共验收)。人工评分仍为 `not_reviewed`；参考交付通过不替代学习者独立复写和答辩。

## 第二遍：八个目录各自怎么用

| 目录 | 第二遍用途 | 建议阶段 |
| --- | --- | --- |
| [激活函数](激活函数/README.md) | 从例子理解非线性，再查导数、饱和、门控、残差与归一化 | 第 1 月按缺口补 |
| [pytorch&mindspore](pytorch&mindspore/README.md) | 4 章把计算契约、梯度、训练和排错讲完整 | 第 1 月；先 PyTorch，MindSpore 对照选读 |
| [RNN](RNN/README.md) | 三步递推导读、原笔记及梯度、门控与 attention 起点 | 第 1 月作序列基础 |
| [transformer](transformer/README.md) | 6 篇覆盖 Embedding、注意力、LM、缓存、数据与系统 | 第 1 月表示基础、注意力与 LM，第 7–8 月数据/系统 |
| [RL](RL/README.md) | 8 章从 MDP 到 PPO、探索与离线 RL | 第 4 月主读，前期按需要补 |
| [post_train](post_train/README.md) | 7 章从 SFT 到 DPO、RLHF、RLVR 与评测 | 第 2–6 月主线 |
| [agent_design](agent_design/README.md) | 3 章讲工具、记忆、检索、规划与可靠性 | 为交互训练准备系统基础 |
| [agentic_rl](agentic_rl/README.md) | 3 章讲多步奖励、信用分配、训练系统与实验设计 | 第 9–10 月可选专题 |

正文中的公式和条件是第二遍重点。每章至少能解释一个公式、定位一段代码、说明一个失败条件，然后再扩展规模。[发展脉络](references/学习路线与发展脉络.md) 解释方法为何出现；[年度任务与验收](references/年度学习任务与验收.md) 把它们连到项目 A/B/C。术语、论文与版本记录可从[阅读附录](references/README.md)按问题查阅。完整项目范围与交付条件见[年度项目与验收](references/年度学习任务与验收.md#project-a)。

## 教材、论文与项目验收

教材对应关系见[参考书籍与课程地图](references/参考书籍与课程地图.md)。能解释对应章节和小实验后，从[20 篇论文省流索引](references/论文索引.md)挑一篇直接相关的论文；首次只跟一个问题。资料检索截止 2026-09-26，最新收录稿为 2026-09-23，版本不会自动更新。术语和读论文的依据见[阅读附录](references/README.md)。

阶段验收依次要求独立复算、改条件并解释、定位一类错误，以及在自己的数据与模型上完成基线、训练、评测和消融。具体交付以[年度项目 A/B/C](references/年度学习任务与验收.md#project-a)、[数据与评测规范](references/数据资产与评测交付规范.md)和[研究专题验收](references/研究专题与复现验收.md)为准。[全项目审查与能力验收](references/全项目审查与实习能力验收.md)给出课程覆盖、岗位依据和证据范围。

## 资料和维护

仓库包含 33 个系统单元、中文入门精讲、20 篇论文省流及 PDF、2 本公开书稿；Embedding 章另列原始资料链接。出处、校验与历史实跑记录分别见[资料清单](references/readings.json)、[下载记录](references/downloads.json)和[版本与验证记录](references/版本与验证记录.md)。需要核查原始 PDF 时运行 `python scripts/download_readings.py --verify-only`，缺文件再运行 `python scripts/download_readings.py`。新增资料按[省流模板](references/论文省流模板.md)记录出处、版本和证据；原始资料保留各自许可与权利。
