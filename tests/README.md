# 仓库验证

在仓库根目录使用已有 Python 3.10 或更新版本运行：

```powershell
python -X utf8 scripts/check_repository.py --clean-copy --tests --smoke stdlib
```

默认检查 Git 跟踪文件和未被忽略的新文件；不扫描 `study_runs/`、个人 `GOAL.md` 或其他被忽略的内容。检查 Python 编译、严格 JSON、Markdown 本地文件及锚点、已下载 PDF 的大小与 SHA-256。严格 JSON 会拒绝重复键、NaN、Infinity 和超出有限浮点范围的指数数值。

`--clean-copy` 把当前公共文件复制到临时目录后重新检查，能发现依赖本机私人文件的链接。待新增文件纳入 Git 后，发布检查使用：

```powershell
python -X utf8 scripts/check_repository.py --tracked-only --archive-ref HEAD
```

`--archive-ref HEAD` 检查已提交版本的真实 Git archive，工作区的未提交改动不会进入该检查。若当前工作区已修复而 HEAD 仍有旧问题，这个门禁会继续失败，直到修复提交到待发布版本。

单独运行回归检查：

```powershell
python -X utf8 -m unittest discover -s tests -p "test_*.py"
```

测试会故意制造失效文件、错误锚点、存在但未公开的文件、重复 JSON 键、损坏的 PDF 和已有实验结果，确认检查器与实验入口能够拒绝这些状态。数学检查独立核对回答 token 的位移与 mask、DPO 四个 log-prob、导数方向、PPO 的负优势裁剪、外部截断后的 bootstrap、Bellman 解、SGNS 梯度与对比学习温度。输出目录测试包含多个线程竞争同一空目录的情况；命令测试核对参数实际生效、源码 SHA-256、解释器版本记录及中文结果交付失败的退出码。

只装标准库的环境仍能跑公共代码、数学和安全回归；模型训练测试会按其环境需求处理。要验证包括 PyTorch 在内的全部默认课程，用已有带 PyTorch 的解释器运行：

```powershell
python -X utf8 scripts/check_repository.py --clean-copy --tests --smoke all --json study_runs/repository_check.json
```

如果验证入口的 Python 没有 PyTorch，可以只给课程指定已有的解释器；`--tests` 仍使用启动验证入口的解释器：

```powershell
python -X utf8 scripts/check_repository.py --smoke all --torch-python "已有环境的 python.exe 路径"
```

冒烟运行把结果放入临时目录，逐课检查退出码、`run.json` 和中文结果说明，并比较运行前后所有公共源文件的 SHA-256。不会覆盖仓库中的固定教学参考结果。完整报告可用 `--json study_runs/repository_check.json` 保存。

GitHub Actions 将标准库验证与 PyTorch CPU 验证分开运行，标准库检查覆盖 Linux/Python 3.10 和 Windows/Python 3.13，CPU 任务固定使用本地已验证的 PyTorch 2.7.1 与 NumPy 2.4.4，并运行所有已登记课程。模型状态哈希与资源观测使用 Tensor.numpy()，因此完整测试环境还需要 NumPy；CPU PyTorch wheel 的依赖不会自动安装它。CI 分别从 PyPI 安装 NumPy、从 PyTorch 官方 CPU 索引安装 PyTorch，并在测试前检查 Tensor.numpy() 是否可用。工作流只有仓库读取权限，报告保存为运行产物。云端状态需以 GitHub Actions 实际运行结果为准；本地通过不能代替云端验证。

## A/B/C 参考实训的验证范围

A 的回归会真实启动多个独立 PyTorch CPU 进程，比较连续/恢复训练，并验证遗漏 optimizer、sampler 与错误标签位移的反例。B/C 的训练编排、LoRA、概率与梯度、状态恢复及资源观测测试使用离线 tiny adapter；它们不下载模型，不把测试替身的指标当作真实 Qwen 的质量证据。真实模型结果另由仓库内的固定参考证据复查。

公开包的标准库入口如下，不需要历史个人运行目录或模型权重：

```powershell
python -X utf8 references/evidence/project_a_20261002/verify.py --source-root .
python -X utf8 scripts/verify_reference_run.py --verify-package references/evidence/project_bc_20261002
```

这两条检查原始文件哈希、数据身份及统计复算；参数梯度和完整训练过程需要按实训说明重新生成检查点。B/C 保留三个真实种子、九个支路、全部三面板预测与采样成本，并用独立答案求解器交叉核验。匿名 AI 辅助审阅与人工复核分别记录，空白人工表继续为 `not_reviewed`。详细范围见 [A/B/C 参考实训](../post_train/experiments/A_B_C参考实训.md)。

Python 源码和公开证据的 Git 换行策略分别固定；已绑定的 CRLF 源文件与证据包保持原始字节，避免不同平台检出使 SHA-256 改变。B/C 的派生成本、时间与 ratio 浮点统计使用明确的绝对和相对 1e-12 容差，其他字段、原始文件和源码哈希仍严格比较。新增 CI 会在标准库任务中执行上述两条公开包核验；远端运行状态仍需以实际 Actions 记录为准。


## 2026-10-03 首次远端 CI 的修正

[首次运行](https://github.com/JacobDaiHit/Maedaily/actions/runs/37089129235) 的 Linux 标准库任务通过；Windows 标准库任务有 4 个路径断言失败，CPU 任务有 14 个缺 NumPy 错误，另有注意力课程的结果交付因依赖警告混入日志而失败。Windows runner 的临时目录使用 `RUNNER~1` 短路径，输出管理会将其解析为对应长路径；测试应比较解析后的同一目录，而不是路径字符串的不同拼写。对测试中期望的目录调用 Path.resolve() 后比较，保留覆盖拒绝、目录竞争和运行收据检查。

这次修正只涉及 CI 的依赖和测试的路径比较，不改动已冻结训练源码或公开证据。远端是否通过仍以修正提交的实际 Actions 结果为准。
