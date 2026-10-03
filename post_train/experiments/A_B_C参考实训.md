# A/B/C 参考实训：从训练闭环到冻结评测

[实验目录](README.md) · [年度项目定义](../../references/年度学习任务与验收.md#project-a) · [真实模型训练与验收](真实模型训练与验收.md) · [数据与评测交付规范](../../references/数据资产与评测交付规范.md)

这份实训把项目 A 的训练与恢复证据，接到项目 B 的 SFT/DPO 和项目 C 的真实采样、奖励、策略更新及独立评测。数字语法、算术与复制归一化都是原创合成教学任务；这里的结果只支持对应数据、模型和预算下的机制解释。

项目 A 已有可在干净克隆中复查的[紧凑公开证据](../../references/evidence/project_a_20261002/README.md)。B/C 提供固定模型、配置、检查与正式矩阵入口。本次参考实训于 2026-10-02 启动，北京时间 2026-10-03 收尾。首轮正式运行中，三个种子的 SFT 开发面板均达到算术 `36/36`、复制 `45/45`、格式全合格；C 的一个变体未证明非零策略梯度，阶段门禁失败，最终面板没有打开。SFT 开发合格不能据此写成整个 B/C 已完成。V2 程序 summary 虽为 passed，但独立校验发现登记源码与当前源码身份不一致，拒绝作为验收结果；原始 source hash 与运行记录保留为诊断。V3 保持 `20 steps / lr=0.0003 / temperature=2.0`，只修复解码与补观测证据，最终面板未用于调参。V3 的真实 runner 退出 0、机制与共同基线质量门禁通过，公共包已完成双解释器独立复算；[公开证据](../../references/evidence/project_bc_20261002/README.md)与下方完整结果表给出实际范围。

## 先完成哪些内容

先能解释[标准库后训练实验](README.md)里的 SFT、DPO、组优势和 clip，再用[模型后训练实验](模型后训练实验.md)核对真实反向传播。字符模型的机制检查和[模型质量基线](模型质量基线.md)分别回答“更新是否正确”和“基线是否具有足够任务质量”。B/C 参考入口使用完整的本地预训练模型，继续采用冻结的 `quality-v2` 数据协议。

本次完成后，应拿实际产物说明：

1. 一次中断后的模型、优化器、采样位置和随机源是否共同恢复；漏掉其中一个状态会怎样改变下一步。
2. SFT 的有效目标位置、DPO 的四个回答概率、RL 的 old/current/reference 概率分别来自哪里。
3. 一个变体是否完成有限、非零的策略梯度更新；奖励不变时为什么不能凭 KL 梯度宣布 RL 已生效。
4. 三个实际训练种子与九个评测分支怎样配对；什么改变只属于一个消融因素。
5. 最终逐题输出是否能独立重算正确率、格式率、区间、长度和成本；失败、回归和不显著变化怎样如实保留。

这些是参考流程的可复核要求。学习者的独立复写、故障注入、口头解释和自己的研究扩展，仍按[年度 A/B/C 交付](../../references/年度学习任务与验收.md#project-b)验收。

## 项目 A：先复查公共训练闭环

在仓库根目录运行：

```powershell
python references/evidence/project_a_20261002/verify.py
python references/evidence/project_a_20261002/verify.py --source-root .
python scripts/run_lesson.py project-a
```

前两条使用标准库核对公开 JSON/CSV、文件 hash、数据与 token 摘要、逐步恢复对照和曲线口径。第二条还校对本次源码版本。第三条需要已有的 PyTorch 环境，真正从零训练并在独立进程恢复，会在新目录生成自己的完整检查点。

公开参考运行采用 CPU float64、seed 7、batch 16，连续训练 40 步，在第 20 步保存后比较恢复的 20 步。学习率 `0.001/0.003` 使用相同初始化、批次和 token 预算。六个子进程实际更新 160 次，使用 23,048 个有效训练 token，完整执行 155.482 秒；这些是原环境的实测，不用于估算另一台机器的速度。完整恢复逐步一致，漏掉优化器或批次采样状态会被对照检出，double shift 和取消因果 mask 的错误注入也被检出。

[公共包说明](../../references/evidence/project_a_20261002/README.md)列出原始 JSON、逐步 CSV 和导出回执。紧凑包没有完整模型检查点，所以标准库校验的证明范围是公开记录重算；实际 tensor 恢复由重新运行实训核验。这个单种子、共享数字语法的例子不能推出普遍最优学习率。

## 项目 B/C：固定本地模型，再离线训练

模型固定为 `Qwen/Qwen2.5-0.5B-Instruct`，revision 为 `7ae557604adf67be50417f59c2c2f167def9a775`。仓库的[模型清单](reference_model.json)保存九份文件的字节数和 SHA256，包含配置、tokenizer、完整权重及许可文件；模型来源与 Apache-2.0 许可随清单保留。不要用同名模型的最新 revision 代替这次登记版本。

使用已有、可加载该模型的 Python 3.10+ 环境。清单记录的验证版本为 PyTorch `2.7.1+cu118` 和 Transformers `5.15.0`；下载还需要 `huggingface_hub`。在自己的环境中复现时应记录实际版本、设备与 dtype，换软件或设备后的逐位一致性要重新检查。

```powershell
python scripts/prepare_reference_model.py
python scripts/prepare_reference_model.py --verify-only
python post_train/experiments/project_bc_reference.py --model-dir study_runs/models/Qwen2.5-0.5B-Instruct --config post_train/experiments/reference_config.json --device cuda --output-dir study_runs/my_project_bc
```

第一条允许联网下载指定 revision 的完整文件，并逐项校验。模型已经准备好时只运行第二条。第三条训练入口使用 `local_files_only=True`、`trust_remote_code=False`，只读取本地文件，不在训练过程中下载模型。没有 CUDA 时可显式改为 `--device cpu`，报告实际耗时；不要把 GPU 的时长当成 CPU 预算。

默认每次建立新的输出目录；显式目录必须是新目录或空目录。检查点用排他创建，原产物不可覆盖。失败后保留目录、`failure.json` 和阶段日志，另开目录运行已重新登记的配置。这个 CLI 没有自动接续整条矩阵的选项；底层精确恢复检查通过，不等于任意中断点都能直接用 CLI 续跑。

[本地 adapter 实现](pretrained_adapter.py)冻结完整基模，仅训练各层 `q_proj/v_proj` 的 LoRA，rank `4`、alpha `8`。基模以 float32、eager attention 加载。不同种子改变 adapter 初始化与实际训练采样，九个分支共享对应种子的 SFT 起点。

V3 用[资源与对齐采集接口](../../scripts/reference_instrumentation.py)在 `00_contract` 写入 `environment.json`、`resource_probe.json` 和 `tokenization_manifest.json`，在 `01_batch` 写入 `token_alignment.jsonl`。环境证据来自实际解释器、库版本、模型参数 dtype、设备及适用 GPU 信息。资源探测使用真实训练记录，计时前后同步 CUDA，执行一次临时 adapter 更新和短生成，再恢复原模型、梯度、模块模式与相关随机状态；临时探测不计作正式 SFT/DPO/RL 更新。

探测记录有效输入/目标/生成 token、forward/backward/optimizer 与生成秒数。CPU peak RSS 是整个进程生命周期的原生内存峰值，不能重置；CUDA allocated 峰值在探测前重置，含已加载模型。峰值计数器本身不能恢复，这两个峰值与整次正式矩阵的测量范围应分别说明。逐位置对齐选择短样本、负数算术和带两个前导零的复制样本，保留特殊 token、未移位 label、prompt/answer/EOS/padding 角色与唯一预测位置。三个边界记录用于对齐和资源探测；独立 CE/序列 log-prob 核验实际另取 `train[:2]` 的两条等长回答，合计有效分母为 4，并保存 `sft_loss_receipt.json`。不能把这次两样本的 loss/梯度核验写成同一组三个边界样本的验证。

V3 已产生的现场环境记录为 Python `3.11.15`、PyTorch `2.7.1+cu118`、Transformers `5.15.0`、huggingface_hub `1.27.0`、safetensors `0.8.0`，Windows / RTX 4060 Laptop GPU，参数为 float32。实际载入 `11.096` 秒；隔离资源探测含 124 输入 token、8 目标 token、2 生成 token 和 1 次临时更新，记录全部调用者状态恢复及 frozen base 不变。

资源探测 CUDA peak allocated 为 `2,337,101,824` bytes（十进制约 2.337 GB），进程 CPU peak RSS 为 `3,602,182,144` bytes（十进制约 3.602 GB）。这两个值来自阶段 00；最终 `summary.cuda_peak_bytes` 是完整正式阶段的另一测量区间，不能混用。这些阶段 00 数值只解释资源探测；整条 V3 的参考验收由全部真实阶段、冻结评测与公共包独立复算共同支持。

## 开发选择与最终评测分开

数据由[固定数据模块](model_quality_data.py)重建。算术题以无序操作数对作为 family，交换顺序和加减版本属于同一题族；复制题的不同数字拼写按规范数值归为同一 family。题族切分避免一个写法在训练、另一个写法在开发或最终同域面板。训练算术标签覆盖 `-9..18`，复制训练包含归一化写法；这解决训练标签缺口，但不保证泛化。

| 面板 | 数量 | 用途 |
| --- | ---: | --- |
| `train` | 340：算术 130、复制 210 | SFT、合成训练偏好、RL 训练奖励 |
| `dev` | 36 | 算术开发质量与配置选择 |
| `dev_copy` | 45 | 复制开发质量与配置选择 |
| `eval_main` | 34 | 未见同域算术题族 |
| `eval_transfer` | 120 | 操作数范围迁移，单独报告 |
| `eval_regression` | 45 | 未见复制归一化题族与能力保留 |

SFT 的 `20 steps / lr=0.0003` 来自 seed 17 的开发校准：预登记两种学习率 `0.0001/0.0003`，各看 `20/60/120` 三个节点，只生成 `dev/dev_copy`。`0.0003` 的 20 步已达到开发满分，后续节点没有增加开发正确题数，因此冻结较短预算。这个选择没有使用最终面板，也不是“20 步通常足够”的建议。正式三种子再训练时，仍逐种子执行相同开发门禁。

开发算术至少 `29/36`、复制至少 `41/45`，两面板格式率必须都是 100%。任何一个 SFT 种子不合格，结果为 `development_unqualified`，`final_metrics=null`，退出码 4；不训练后续 DPO/RL，不运行最终面板。全部合格后先写 `sft_lock.json`，保存三份起点权重 hash 与开发成绩，再建立比较支路。

最终同域算术要求至少 80%，复制至少 90%，两面板格式全合格且各不少于 20 题；对应这份数据即 `eval_main≥28/34`、`eval_regression≥41/45`。迁移面板用于报告能力边界，不参与基线合格判定。最终门禁只检查共同的 `no_update` SFT 基线，不能把表现差的基线换成某个更好的实验分支。打开最终面板后，不用它回头改配置、换 seed 或重选检查点。

## 三个种子、九个分支与单因素消融

[正式配置](reference_config.json)固定 seeds `17/23/41`，batch 8；SFT 与继续 SFT 每批各采样 4 条算术、4 条复制。正式输出的 `preregistration.json` 同时冻结模型文件、实验源码、数据、配置与分支。下面列出已冻结的共同预算；首轮温度为 1.4，第二轮和 V3 为 2.0。V3 另修复生成特殊 token 的解码，并补环境、资源和逐位置对齐证据；其余训练与评测超参数不变，实际运行以自己的 `config.resolved.json` 为准。

| 分支 | 起点与变化 | 登记预算 |
| --- | --- | --- |
| `base` | 完整预训练模型，LoRA 增量为零 | 无任务训练 |
| `no_update` | 共同的 SFT 起点 | SFT 20 步，lr `0.0003`；之后不更新 |
| `dpo` | 从同 seed SFT 分支进行 DPO | 6 步，batch 8，lr `0.0001`，beta `0.2` |
| `dpo_beta` | B1 消融：仅改变 DPO beta | beta `0.1`，其余与 `dpo` 相同 |
| `continue_sft` | 从同 seed SFT 继续监督训练 | 额外 3 步，lr `0.0003` |
| `rlvr` | 从同 seed SFT 进行真实采样与更新 | 3 轮，每轮 4 题组，每组 4 个回答；lr `0.00005` |
| `rlvr_group_size` | C2 消融一：仅改变组内回答数 | group size `4→8` |
| `rlvr_no_format` | C2 消融二：仅去掉格式奖励 | format reward `0.1→0` |
| `sampling_only` | 同 seed SFT 不训练，多次生成后选择 | 每题 4 个候选，合法整数多数票；并列取最早候选 |

这是 27 组“seed × 评测分支”结果，其中 `no_update` 与 `sampling_only` 使用同一训练检查点，不能算成 27 次独立训练。增强采样选择器只读候选文本、停止原因与格式，不读正确答案或奖励。普通分支最终用 greedy；增强采样用温度 1 的随机生成，实际成本包括全部四个候选。

首轮 RL 温度为 `1.4`，当前 V3 冻结为 `2.0`；clip `0.2`、KL 系数 `0.01`、最长生成 8 个新 token 均未改变。SFT 很强时，同组全对或全错都可能造成零奖励方差。首轮 C 出现零信号后，保留失败产物并封存最终面板，再只用训练题进行采样校准。[选择记录](reference_selection.json)保留四个候选温度，各 8 组、每组 4 个回答，共 128 次真实生成：

| 温度 | 正确 / 32 | 合法格式 / 32 | 有正确性奖励方差的组 / 8 |
| --- | ---: | ---: | ---: |
| 1.4 | 32 | 32 | 0 |
| 2.0 | 12 | 12 | 7 |
| 2.5 | 0 | 0 | 0 |
| 3.0 | 0 | 0 | 0 |

选择预登记候选中最低的有正确性奖励方差的温度 `2.0`；“7 个 active groups”是实际观察结果，不是事先要求达到 7 组的门槛。随后冻结新配置，在新目录重新执行全部三种子九分支。V3 使用同一温度 2.0，在解码修复与资源/对齐证据补齐后再次冻结运行；实际成绩及全部负结果见下方 V3 结果表。校准没有访问最终面板，不改变质量门禁，也不能挑出“恰好有梯度”的 seed 当作正式重复。behavior、old、current、reference 都使用同一温度；提高温度提供奖励差异的同时降低单次生成质量，这项代价也要报告。

group size 8 的支路收集 `3×4×8=96` 个回答，size 4 收集 48 个。相同轮数不代表相同采样 token 或算力。继续 SFT 的 3 步也只匹配更新机会数，监督 token、生成开销和目标函数均不同。DPO 与 RL 的更新数、样本数不同，比较时同时报告真实预算；不能从这种短预算差异推出方法更优。

## 阶段依赖决定哪些结论可以写

[正式 runner](project_bc_reference.py)将每个阶段的输入、输出 hash、起止时间、检查与失败原因写入 `stages.jsonl`。依赖没有通过就不进入下一阶段。

| 阶段 | 进入条件与实际证据 |
| --- | --- |
| `00_contract` | 先保存登记、配置、数据与源码版本；完整本地模型加载，记录真实环境、资源探测与 tokenizer manifest |
| `01_batch` | 保存真实逐位置 token/label/mask 对齐；SFT 标量与参数梯度、DPO 四概率与导数、基模冻结和 verifier 边界通过 |
| `exact_resume` | 连续下一步与恢复下一步的模型、optimizer、采样及 RNG 一致 |
| `02_sft` | 三个 seed 真训练、保存检查点，再只评估两个开发面板 |
| `sft_lock.json` | 全部开发合格，锁定共同起点与 hash |
| `03_dpo` | 每个 seed 的两个 beta 分支各从锁定的 SFT 起点训练 |
| `04_rollout_05_update` | 继续 SFT、三个 RL 变体均完成；每个 RL 变体至少证明一次非零 PG 更新 |
| `checkpoint_reload` | 所有计划分支重载后，权重与回答 log-prob 严格匹配；源码 hash 未变化 |
| `06_eval` | 先保存 `final_opening.json`，再一次性运行三块冻结最终面板，记录全部 seed/分支 |
| `07_reproduce` | 保存逐题记录与训练回执，独立重算报告，再判定机制和最终基线质量 |

程序异常写 `failure.json` 并保留已完成步骤。若所有训练和评测实际完成，但最终共同基线质量门禁失败，结果为 `completed_with_failed_gate`，退出码 4。最终指标只覆盖这套合成算术与数字规范化任务，所有 arms 的未提升、无效和负结果都保留。DPO/RL 没有改善、效果不显著或某项能力回归都可以是完整可报告的负结果；“基线不合格”与“方法效果为负”是不同判断。零优势没有实际 PG 更新时，可以报告机制失败，不能将 KL 引起的参数变化包装为有效 RL 训练。

## 项目 B：监督区间与四个偏好概率

SFT 将固定 chat prompt 与回答拼接，EOS 属于回答，提示和 padding 的目标 mask 为假。最后一个提示位置预测第一个回答 token；loss 只做一次因果 shift，再按全批有效回答 token 平均。实现保留完整 Transformer 上下文，只对需要监督的预测位置投影词表，未计算位置的零是占位，不能解释成 token log-prob。

[独立检查](adapter_checks.py)将 compact 路径与完整原始 forward 比较：手写 gather/mask CE、独立 `cross_entropy` 标量和实际 adapter 参数梯度必须一致。提示位置的直接 logits loss 梯度为零；提示表示仍可通过 attention 参与回答计算。每个更新记录真实样本 ID、有效 token、loss、裁剪前梯度范数、参数变化范数和前后权重 hash，拒绝非有限或不连接模型图的 loss。

DPO 只从训练算术题构造 chosen=正确答案、rejected=答案加一。这是合成可验证偏好，不宣称来自人工偏好。对回答加 EOS 的序列 log-prob 求和，不按回答长度平均。四个量为 `policy_chosen/policy_rejected/reference_chosen/reference_rejected`，目标为：

$$
L_{\mathrm{DPO}}=-\frac1N\sum_i\log\sigma\!\left(\beta\left[(s_{\theta,i}^{+}-s_{\theta,i}^{-})-(s_{\mathrm{ref},i}^{+}-s_{\mathrm{ref},i}^{-})\right]\right).
$$

reference 是同 seed 的 SFT adapter 快照，通过 `functional_call` 与冻结基模求值，不进 optimizer，不保存梯度，也不随 policy 更新。独立检查核对初始 loss 约 `ln(2)`、四概率、解析导数、交换偏好后的符号与实际 adapter 参数梯度。B1 只改变 beta；主比较与消融都保留原始逐步记录，不能挑最有利的一步报告。

## 项目 C：真实分布、独立 PG 与 KL

每轮先保存当前 policy 的不可变 old 快照，再在训练题上随机自回归生成。同轮采样期间不更新模型；reference 固定为最初的 SFT 起点，old 在下一轮刷新。行为概率来自真实采样的完整词表分布 `softmax(logits/τ)`，无 top-k、top-p 截断或额外禁用 token。首轮 `τ=1.4`、当前 V3 `τ=2.0`，都与普通 greedy/温度 1 评测不同。

rollout 保留原始文本、token、停止原因、每步 behavior log-prob、组 ID、奖励、优势、old/reference hash 和独立 generator 状态。learner 用同快照、同温度、同 prefix 和动作重算 old 概率；实际 replay 最大误差不超过 `3e-4`，更新前 ratio 距 1 的最大误差也不超过 `3e-4`。这些检查防止把温度 1 的 teacher-forcing 概率当成温度 1.4 或 2.0 的行为概率。

主奖励为 `正确答案×1 + 合法格式×0.1`。合法格式要求严格规范整数、长度不超过 4 字符且遇到 EOS；空回答、前导零、`+5`、`-0`、空白、多个答案、非数值、超长或 token 预算截断均不能获得格式分。只有文本正确但没有 EOS 的截断也不是正确完成。解码只移除末尾登记的 EOS，其他 PAD、BOS 或聊天控制 token 使用 `skip_special_tokens=False` 保留；`[im_start, "5", EOS]` 不能被静默变成合法答案 `"5"`。这个奖励漏洞在较早版本中被发现并修复，历史版本与源码 hash 保留，V3 重新执行最终验收。去格式奖励变体只保留正确答案项；奖励来自训练标签，不注入生成 forward。

每个题组使用总体标准差，`ε=1e-8`：

$$
A_i=\frac{R_i-\bar R}{\sigma_R+\varepsilon}\quad(\sigma_R>\varepsilon),\qquad A_i=0\quad(\sigma_R\leq\varepsilon).
$$

同一回答的所有有效动作 token 共用该回答的 detached 优势。设本批有 $B$ 个回答，$m_{it}$ 为移位后的动作 mask，$T_i=\sum_t m_{it}$；EOS 若实际生成则计入，提示与 padding 不计入。$\ell_{\theta,it}^{\tau}$、$\ell_{\mathrm{old},it}^{\tau}$、$\ell_{\mathrm{ref},it}^{\tau}$ 都是相同温度下的条件 log-prob。实际最小化目标为：

$$
\rho_{it}=\exp(\ell_{\theta,it}^{\tau}-\ell_{\mathrm{old},it}^{\tau}),\qquad
L_{\mathrm{PG}}=-\frac1B\sum_i\frac1{T_i}\sum_t m_{it}\min\!\left(\rho_{it}A_i,\operatorname{clip}(\rho_{it},1-c,1+c)A_i\right),\quad c=0.2.
$$

$$
d_{it}=\ell_{\mathrm{ref},it}^{\tau}-\ell_{\theta,it}^{\tau},\qquad
k_{it}=e^{d_{it}}-d_{it}-1,\qquad
L_{\mathrm{KL}}=\frac1B\sum_i\frac1{T_i}\sum_t m_{it}k_{it},\qquad
L=L_{\mathrm{PG}}+\lambda L_{\mathrm{KL}},\quad\lambda=0.01.
$$

两个项都先按每个回答的有效 token 数平均，再按回答平均，所以每个回答权重相同；它不同于 SFT 的全批有效 token 平均。每批 rollout 只执行一次 optimizer 更新，更新前 current=old、ratio 约为 1，所以本次短流程没有进入 clip 的饱和区间，也没有验证多轮复用轨迹时的 clip 稳定性；它核验的是已登记目标和实际梯度。old 概率、reference 概率和优势都 detached，只有 current 保留模型图。程序单独对 `L_PG` 求 adapter 梯度，确认有限、非零后才使用总 loss 更新；梯度裁剪后还要证明参数真的变化。全组零方差时记录 `skipped_zero_advantage`，不执行 optimizer step。收集轮数、真实更新数、生成 token 与有效更新 token 分别报告。

`k` 是逐动作非负的 k3 形式；固定 prefix、动作确实从 current 分布采样时，它的期望对应 `KL(current || reference)`。本实训保存的是 old 采样轨迹，每轮仅作一次更新，更新前 current=old。报告中的 `kl_sample_estimate` 是这些采样 prefix/token 上的估计，并未精确遍历词表或完整轨迹，也没有对 KL 项加入整轨迹重要性权重。因而不能把有限样本值解释成完整任务分布的精确 KL；它约束的还是温度 τ 的策略分布。

## 完整恢复与可重算报告

检查点包含 adapter、optimizer、真实 step、模型配置与契约 hash、Python 批次采样器状态、全局 PyTorch RNG、适用的 CUDA RNG 和独立 `torch.Generator` 状态。`exact_resume` 先完成一个更新保存，再比较连续与恢复的下一步：批次 ID、全局随机抽样、真实生成动作、模型、optimizer 与所有相关 RNG 均一致。调用检查后的原策略与随机状态也恢复。仅能重载权重或相同 logits，还不足以说明续训一致。

正式目录中的最小复查顺序是：`preregistration.json` → `environment.json` / `resource_probe.json` / `tokenization_manifest.json` → `token_alignment.jsonl` / `batch_checks.json` / `resume_check.json` → 各 seed 的开发与训练记录 → `sft_lock.json` → `reload_checks.json` → `final_opening.json` → `predictions.jsonl` / `training_receipts.json` → `report.json` / `summary.json`。阶段失败时只提交已经产生的真实文件，并明确未执行阶段；不要创建空的最终成绩来凑齐清单。

最终记录存在后，可另开报告目录重算并准备盲评：

```powershell
python post_train/experiments/reference_reporting.py --records study_runs/my_project_bc/predictions.jsonl --training-receipts study_runs/my_project_bc/training_receipts.json --output-dir study_runs/my_project_bc_report
python post_train/experiments/reference_reporting.py --verify-package study_runs/my_project_bc_report/evidence
```

[报告工具](reference_reporting.py)为每个 seed、分支、面板和能力桶报告正确率、格式率、95% Wilson 区间、停止原因、结果长度与实际输入/输出 token。Wilson 区间采用独立题目假设，相关题族会违反这个假设。相对同 seed 的 `no_update`，另按完整 family 配对 bootstrap 2,000 次，重采样时保留同题族的全部写法和生成记录；改善题与退化题都保留。

三个训练种子的均值、样本标准差、范围和逐 seed 成绩单独列出。family bootstrap 描述已训练模型之间的题族不确定性，三 seed 离散度描述训练重复差异；工具没有把二者合成一个区间，也不能凭三 seed 宣布统计显著、等效或普遍优越。增强采样的四个候选是额外生成，不能算成四次独立训练重复。

训练预算从真实回执读取 optimizer steps、有效更新 token、生成样本/token、实际耗时与权重变化。评测成本累计所有候选的输入/输出 token 与生成秒数，结果长度只描述被选中的回答。累计生成秒数不等于并发吞吐、整次训练墙钟或费用；CUDA 峰值使用实际 allocated bytes，CPU 则保留为空，不能从理论显存反推实测。

报告入口还生成去掉分支身份的盲评记录和空白评分表，身份映射另行封存。空白表代表 `not_reviewed`，不是已经完成的人评、AI 评或自评。证据包包含重算必需的记录、报告脚本、回执与 hash，可独立核对，不依赖作者电脑上的 ignored 路径；报告包本身不包含完整基模及全部训练 checkpoint，也不证明训练已完成。训练复现仍使用固定模型清单、runner、配置和原始产物。

## 从机制实训进入一个有边界的研究复现

本课的原始方法阅读固定为 Shao 等的 [DeepSeekMath，arXiv:2402.03300v1](https://arxiv.org/abs/2402.03300v1)，提交日期为 2024-02-05；读[该版本原文](https://arxiv.org/pdf/2402.03300v1)第 4.1 节的式 (3)、(4) 与 outcome supervision。论文用同题多回答的相对奖励代替额外 value model，逐 token clip 并直接加入 KL 正则；这为课程的组优势、PG/KL 分离与消融提供固定参照。

可以把“组大小是否改变零方差组比例”和“去掉格式奖励后是否仍有正确性信号”预登记成一个机制复现。固定 SFT hash、数据、seed、温度、奖励定义、mask 与归约，先写可证伪的问题和失败判据，再收集实际 reward 分布、active group 数、独立 PG 范数、更新幅度与最终配对变化。若研究的是算力公平，应另登记相同生成 token/墙钟预算的对照；本页按相同轮数改变组大小的消融不能回答这个问题。

本实现是 0.5B 模型上的 q/v LoRA、最多 8 token 的整数输出与 3 轮短预算；没有自然语言推理轨迹、训练 reward model、过程奖励或迭代 reference 刷新。它使用总体标准差与零方差跳过规则，KL 在保存的 old 动作上计算，具体分布和归约由前面的公式定义。即使某分支在教学面板改善，也不能称为复现了原论文的大模型数学效果，更不能用 DPO/RL 或采样预算不等的结果宣布算法优越。研究专题的新增数据、完整基线、预算和结论范围，继续按[研究专题与复现验收](../../references/研究专题与复现验收.md)交付。

## V3 真实结果与公共验收

V3 的实际 runner 已退出 0，`summary.status=passed`、`mechanisms_passed=true`，三个训练种子和 27 份原始评测文件齐全。下表来自这次固定协议的真实计数；[公共包](../../references/evidence/project_bc_20261002/README.md)已重算评分、报告、阶段与身份，在禁用第三方包的 Python 3.11/3.13 及仅 evidence 的临时副本中实际退出 0，参考交付验收通过。

| 分支 | 同域算术，每个 seed / 34 | 数字规范化，每个 seed / 45 | 迁移 seed 17 / 120 | 迁移 seed 23 / 120 | 迁移 seed 41 / 120 | 三 seed 迁移计数合计 / 360 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | 33 | 3 | 107 | 107 | 107 | 321 |
| `no_update`（SFT） | 34 | 45 | 118 | 117 | 120 | 355 |
| `dpo` | 34 | 45 | 119 | 117 | 120 | 356 |
| `dpo_beta` | 34 | 45 | 119 | 117 | 120 | 356 |
| `continue_sft` | 34 | 45 | 118 | 119 | 120 | 357 |
| `rlvr` | 34 | 45 | 119 | 117 | 120 | 356 |
| `rlvr_group_size` | 34 | 45 | 119 | 117 | 120 | 356 |
| `rlvr_no_format` | 34 | 45 | 119 | 117 | 120 | 356 |
| `sampling_only` | 34 | 45 | 119 | 117 | 120 | 356 |

三个共同 SFT 起点最终主面板均 `34/34`、数字规范化均 `45/45`，格式全合格，达到登记质量门禁；其余七个 SFT 后分支在这两面板也全对。预训练 `base` 的数字规范化是 `3/45`，不能写成 0；其主/迁移/规范化格式计数分别为 `34/34`、`119/120`、`44/45`。SFT 后所有分支三面板的格式均合格。

SFT 迁移合计 `355/360`（98.61%），DPO、三个 RL 变体与增强采样为 `356/360`（98.89%），继续 SFT 为 `357/360`（99.17%）。这只是三个已训练模型的计数汇总，同一题在多个 seed 重复评测，不能视为 360 道独立题。相对 SFT，前一组累计仅多 1 条正确记录、继续 SFT 多 2 条；主/回归面板已饱和，本预算下 B1/C2 没有观察到正确率差异。全部未提升或无效结果保留，不把一题差作为效果标题。

同 seed 题族配对 bootstrap 中，seed 17 的 DPO/RL 迁移差为 `1/120`，95% 区间为 `[0, 0.025]`；seed 23 的继续 SFT 差为 `2/120`，区间为 `[0, 0.041667]`，其余相应 seed 差为 0。区间均包含 0，且它们条件于这几份训练权重，三 seed 的变异另报。这里没有建立统计显著性、方法等效或方法优越性。

正式训练合计 132 次 optimizer 更新、6,333 个有效更新 token；三个 RL 变体×三个 seed 共生成 576 个回答。精确恢复检查的 3 次物理更新和资源探测的 1 次临时更新另外记录，恢复后不计入正式训练预算。九份固定模型文件在训练后再次核验 SHA/大小通过。所有三个 RL 变体每个 seed 都实际完成 3 次 optimizer 更新，独立 PG 有限非零。主 RL 三 seed 共生成 144 个回答、866 个动作 token；组大小消融共 288 个回答、1,686 个动作 token，同样的 9 次更新消耗了更多采样。增强采样每题 4 个候选；仅迁移面板每 seed 的输入成本就是 `17,760` token，而 greedy 分支为 `4,440`。增强采样的累计生成秒数约 63.9–66.1 秒/seed，SFT greedy 约 2.4–2.8 秒/seed；这些是相应路径的实际累计生成时间，并非并发吞吐或整次墙钟，缓存和批量生成方式也不同。

正式 summary 的 CUDA peak allocated 为 `5,348,077,056` bytes（十进制约 5.348 GB），属于完整正式阶段；阶段 00 的 `2,337,101,824` bytes 探测峰值是不同区间。最终收益、区间与成本只解释合成算术和数字规范化，不能从本次短预算推广到自然语言推理或通用模型性能。

公共包为 `27,677,059` bytes，包含 5,373 条逐题预测、三个 seed×九个 arms、15 个阶段、11 份冻结训练源码与配置，独立评分重算和 72 组面板/题族配对区间。52 份未打包权重的文件 hash 保留在 manifest：它们是原运行的历史回执，离线校验不能独立核实未包含的权重字节。复算使用原始记录；真正重新训练还需准备固定模型，并生成自己的完整检查点。

在仓库根目录执行包内标准库入口：

```powershell
python -S references/evidence/project_bc_20261002/scripts/verify_reference_run.py --verify-package references/evidence/project_bc_20261002
```

[复现说明](../../references/evidence/project_bc_20261002/REPRODUCE.md)与[manifest](../../references/evidence/project_bc_20261002/manifest.json)不依赖 ignored 目录。两解释器的无历史运行目录副本已实际通过。Python 3.13 曾对 68 个成本浮点末位产生差异；校验现在只对指定浮点字段使用 absolute/relative `1e-12` 容差，SHA、整数、bool、ID、评分和配置继续严格一致，原始训练源码与原始 report 未改。17 项跨版本定向测试通过，未因此重新训练或选择参数。

原始盲评空表仍不是已评分证据；另有[AI 辅助匿名抽样审阅](../../references/evidence/project_bc_20261002/run/ai_assisted_review.json)与[交叉核对](../../references/evidence/project_bc_20261002/run/ai_review_crosscheck.json)：真实 27 条，26 条正确、27 条格式合法，与独立脚本 0 分歧。它是有限抽样的 AI 辅助核对，人工审阅仍明确 `not_reviewed`，不替代学习者答辩或人工标注质量。

公开源码与证据包的 SHA 绑定原字节。[换行规则](../../.gitattributes)让普通 Python 使用 LF；本次冻结的 `adapter_checks.py` 原 CRLF 字节单独保留，证据目录中的 CSV/JSON 也不受 Git 换行转换。原始 SHA 校验与 checkout 后的独立复算一起通过，才表示包可以跨克隆复查。标准库 [CI 配置](../../.github/workflows/repository-checks.yml)包含 A/B/C 公共校验步骤；本次只报告本地实际验证，未声称远端 CI 已运行。

最新完整工程回归使用 `-Werror::ResourceWarning`，146 项实际通过、364.850 秒，退出 0 且无 warnings；15 类登记课程的默认配置真实运行通过、227.357 秒，入口/运行时/runner 源码前后不变；原有 34 份历史资料/结果 hash 不变。这些本地验证、公共证据复算与参考模型质量分别记录，不能互相替代。

## 提交时检查

- 保留全部计划 seed 和分支的真实记录，分别写计划数量、完成数量、失败阶段与未执行阶段；不能挑 seed。
- 给出固定模型 revision、九文件验证结果、数据/源码/配置 hash，先锁定开发选择，再打开最终面板。
- 展示一个真实 batch、DPO 四概率、一次可回放 rollout、独立 PG 证明和完整恢复对照。
- 同时报告主任务、迁移、复制回归、格式、长度、实际更新/采样/评测成本，以及 B1/C2 单因素差异。
- 对负结果、零信号、基线不合格与尚未评分的盲评分别下结论；公共文件可以在干净克隆中复查，学习者能独立说明证据的适用范围。
