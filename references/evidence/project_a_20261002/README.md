# 项目 A：2026-10-02 紧凑公开证据

这是 [project_a_reference.py](../../../transformer/experiments/project_a_reference.py) 的一次真实运行导出，克隆后可以读取 JSON/CSV 并使用标准库复核。模型从零训练合成数字语法，没有预训练通用语言能力。学习者独立复写、排错和答辩仍按[年度项目 A](../../年度学习任务与验收.md#project-a)验收。

从仓库根目录执行，不需要 PyTorch：

```powershell
python references/evidence/project_a_20261002/verify.py
python references/evidence/project_a_20261002/verify.py --source-root .
```

第一条命令检查公开文件 SHA/大小，独立重建数字语法的数据摘要与 token 数，重算逐步 batch/loss/状态相等标识、错误对照和曲线口径。第二条再对照本次四份训练源码 SHA；将来修改源码后，该严格版本比较可能失败，此时应按新的源码重跑并发布新版本证据。

实际配置是 seed=7、CPU float64、单线程、batch=16、连续 40 步，对照保存于第 20 步后在新进程恢复。两种固定学习率 0.001/0.003 使用相同初始化、批次和 token 预算，只改变 `optimizer.lr`。六个子进程共执行 160 次真实更新、23,048 个有效训练 token；完整执行用时 155.482 秒，包括解释器启动与审计。

| 学习率 | 初始训练 loss | 最终训练 loss | 初始验证 loss | 最终验证 loss |
| --- | ---: | ---: | ---: | ---: |
| 0.001 | 2.563573 | 1.641162 | 2.570013 | 1.795121 |
| 0.003 | 2.563573 | 0.837108 | 2.570013 | 1.183015 |

20 个恢复步骤的 batch、loss、模型状态、optimizer 和 RNG 都与连续训练严格一致，最终完整 tensor/容器也由原运行直接比较。缺优化器的首批和首 loss 一样，但首次更新后的参数不同；缺采样状态的首批与参数不同。double shift 与取消因果 mask 的错误注入均被检出。

公开文件包括 [evidence.json](evidence.json)、[逐步对照 CSV](step_comparison.csv)、[学习曲线 CSV](learning_curves.csv)、[原始导出回执](export_receipt.json) 和 [公开包 manifest](package_manifest.json)。这四份原始导出文件保持原字节；manifest 另记录它们及本说明、校验程序的 SHA。原始运行曾校验 37 份产物，包含真实 checkpoint，完整 checkpoint 没有放入这个紧凑包。标准库校验只能复核公开证据，不能替代实际恢复 tensor；重新执行实训会在自己的新目录生成全部 checkpoint。

`export_receipt.source_run`、`environment.python_executable`、PID、平台和路径是原始运行环境元数据，不是克隆后的依赖路径。公开校验不会访问 ignored `study_runs`，也不会调用记录中的原解释器。复现时选择自己的已有 PyTorch 环境：

```powershell
python scripts/run_lesson.py project-a
python transformer/experiments/project_a_reference.py --steps 40 --split-step 20 --output-dir study_runs/my_project_a
```

输出目录必须是新目录或空目录。统一入口为此实训提供 600 秒上限；单个 worker 默认上限为 300 秒。换设备或软件版本不保证逐位一致，当前证明限定于固定 CPU/软件。模型没有 dropout、scheduler、AMP 或梯度累积，checkpoint 保存于 optimizer-step 边界。验证是共享语法、共享部分前缀的长度插值，单种子结果不能推出普遍最优学习率。
