"""中文实验入口：检查已有环境、运行指定课、为每次运行保留独立日志。"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from lesson_runtime import reserve_output_dir

ROOT = Path(__file__).resolve().parents[1]
LESSONS = {
    "first": ("第一课：手算一次学习", "scripts/first_steps.py", False, True),
    "linear": ("框架：梯度、线性训练与恢复", "pytorch&mindspore/experiments/linear_train_lab.py", True, True),
    "lm": ("语言模型：数字语法与学习率对照", "transformer/experiments/tiny_causal_lm.py", True, True),
    "lm-resume": ("语言模型：完整续训与遗漏状态对照", "transformer/experiments/lm_resume_lab.py", True, True),
    "project-a": ("项目 A 参考实训：独立进程恢复与单因素对照", "transformer/experiments/project_a_reference.py", True, True),
    "data-asset": ("数据资产：审计、题族切分与可重建记录", "scripts/data_asset_lab.py", False, True),
    "embedding": ("表示学习：查表梯度、SGNS 与对比学习", "transformer/experiments/embedding_math_lab.py", False, False),
    "attention": ("注意力：VJP、RoPE、缓存与 MLA 等价", "transformer/experiments/attention_math_lab.py", True, False),
    "rl": ("强化学习：表格决策与 Q-learning", "RL/experiments/tabular_mdp.py", False, True),
    "posttrain": ("后训练：概率、DPO 与组优势核算", "post_train/experiments/posttraining_math_lab.py", False, False),
    "agent": ("Agent：工具调用与错误恢复", "agent_design/experiments/tool_state_machine.py", False, False),
    "trajectory": ("Agentic RL：轨迹、mask 与回报核算", "agentic_rl/experiments/trajectory_accounting.py", False, False),
    "sft-model": ("后训练模型：真实监督与参数更新", "post_train/experiments/model_training_lab.py", True, True),
    "dpo-model": ("后训练模型：冻结参考的偏好更新", "post_train/experiments/model_training_lab.py", True, True),
    "rlvr-model": ("后训练模型：采样、奖励与策略更新", "post_train/experiments/model_training_lab.py", True, True),
}
READ_NEXT = {
    "first": "入门/第一课_从预测到学习.md",
    "linear": "pytorch&mindspore/从零上手.md",
    "lm": "transformer/从零上手.md",
    "lm-resume": "transformer/experiments/模型续训实验.md",
    "project-a": "transformer/experiments/README.md",
    "data-asset": "post_train/experiments/数据资产实验.md",
    "embedding": "transformer/chapters/01_Embedding与表示学习.md",
    "attention": "transformer/chapters/03_位置编码_KVCache与注意力效率.md",
    "rl": "RL/从零上手.md",
    "posttrain": "post_train/从零上手.md",
    "agent": "入门/工具交互第一课.md",
    "trajectory": "入门/工具交互第一课.md",
    "sft-model": "post_train/experiments/模型后训练实验.md",
    "dpo-model": "post_train/experiments/模型后训练实验.md",
    "rlvr-model": "post_train/experiments/模型后训练实验.md",
}
MODEL_MODES = {"sft-model": "sft", "dpo-model": "dpo", "rlvr-model": "rlvr"}


def probe(executable: str, need_torch: bool) -> tuple[bool, str]:
    code = "import sys; sys.exit('Python 需要 3.10 或更新版本') if sys.version_info < (3,10) else None; print(sys.version.split()[0])"
    if need_torch:
        code += "; import torch; print('PyTorch ' + torch.__version__)"
    try:
        completed = subprocess.run([executable, "-X", "utf8", "-c", code], capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=30, cwd=ROOT)
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)
    if completed.returncode:
        lines = (completed.stderr or completed.stdout).strip().splitlines()
        return False, lines[-1] if lines else f"退出码 {completed.returncode}"
    return True, completed.stdout.strip().replace("\n", " / ")


def choose_python(explicit: str | None, need_torch: bool) -> tuple[str | None, list[str]]:
    candidates = [explicit] if explicit else [sys.executable]
    # 仅尝试本项目已验证的已有环境；不扫描磁盘、不安装或修改环境。
    if not explicit:
        for known in (Path("C:/Python313/python.exe"), Path("D:/anaconda/envs/pytorch_env/python.exe")):
            if known.is_file():
                candidates.append(str(known))
    reports, seen = [], set()
    for candidate in candidates:
        executable = str(Path(shutil.which(candidate) or candidate).expanduser().resolve())
        key = os.path.normcase(os.path.abspath(executable))
        if key in seen:
            continue
        seen.add(key)
        usable, description = probe(executable, need_torch)
        reports.append(f"{executable}：{'可用' if usable else '不可用'}；{description}")
        if usable:
            return executable, reports
    return None, reports


def chinese_summary(lesson: str, output: Path) -> str:
    """把原脚本机器可读字段解释为中文；原始日志仍完整保留。"""
    filename = "summary.json" if lesson in {"linear", "lm", "rl", "lm-resume", "data-asset", "project-a"} or lesson in MODEL_MODES else "console.txt"
    if lesson == "first":
        data = json.loads((output / "artifacts" / "first_steps.json").read_text(encoding="utf-8"))
        return (f"实际学习率={data['learning_rate']:g}；预测 {data['before']['prediction']:g} → "
                f"{data['after']['prediction']:g}；损失 {data['before']['loss']:g} → "
                f"{data['after']['loss']:g}。\n"
                "已核对解析梯度、概率与折扣回报；先预测改变学习率后的结果。")
    if lesson == "embedding":
        return ("已核对重复 ID 的查表梯度累加、SGNS 手算与有限差分、同步参数更新、"
                "PMI 条件、masked pooling、InfoNCE 温度梯度和检索排序。\n"
                "这是小规模数学核算；完整中间结果见 console.txt，尚未训练句向量模型。")
    data_output = output / "artifacts" if LESSONS[lesson][3] else output
    data = json.loads((data_output / filename).read_text(encoding="utf-8"))
    if lesson == "project-a":
        checks = data["mechanism_acceptance"]
        if (data.get("schema_version") != "maedaily-project-a-reference-v1" or
                data.get("status") != "mechanism_passed" or checks.get("status") != "passed" or
                not checks.get("every_resumed_step_exact") or not checks.get("prefix_replay_exact") or
                not checks.get("final_state_exact") or not all(checks["final_state_exact"].values())):
            raise ValueError("项目 A 参考实训机制尚未通过")
        lines = [f"独立训练子进程 {checks['fresh_process_count']} 个；恢复后每步 batch、loss、参数、optimizer 和 RNG 严格一致。",
                 f"遗漏优化器检出={checks['missing_optimizer_detected']}；遗漏采样器检出={checks['missing_sampler_detected']}。"]
        for name in ("base_lr", "alternate_lr"):
            run = data["quality"][name]
            lines.append(f"固定学习率 {run['learning_rate']:g}：训练损失 {run['initial_train']['loss']:.6f} → "
                         f"{run['final_train']['loss']:.6f}；验证损失 {run['initial_validation']['loss']:.6f} → "
                         f"{run['final_validation']['loss']:.6f}。")
        lines.append("mechanism_passed 是工程机制验收；质量结果仅限合成数字语法、单种子 CPU float64，不能推断通用语言模型或学习者能力。")
        lines.append("逐步证据、双学习率曲线、资源和原始检查点保留在 artifacts/；紧凑公开证据见 references/evidence/project_a_20261002/。")
        return "\n".join(lines)
    if lesson in MODEL_MODES:
        if data["status"] != "passed":
            raise ValueError(f"模型实验阶段尚未通过：{data['status']}")
        mode = data["mode"]
        selected = "sft" if mode == "sft" else mode
        panels = data["metrics"][selected]
        lines = [f"实际模式={mode.upper()}；SFT 参数更新 {data['budgets']['sft']['optimizer_steps']} 步。"]
        if mode == "dpo":
            lines.append(f"DPO 参数更新 {data['budgets']['dpo']['optimizer_steps']} 步；参考模型保持冻结。")
        if mode == "rlvr":
            budget = data["budgets"]["rlvr"]
            lines.append(f"实际采样 {budget['generated_samples']} 条；有效策略更新 {budget['optimizer_steps']} 步；"
                         f"零优势跳过 {budget['zero_advantage_rounds']} 轮。")
        labels = {"eval_main": "主任务", "eval_transfer": "迁移", "eval_regression": "回归"}
        lines.extend(f"{labels[key]}：正确 {panel['correct']}/{panel['total']}；"
                     f"有效格式 {panel['format_valid']}/{panel['total']}。" for key, panel in panels.items())
        lines.append("逐位置标签、梯度、重载与固定评测见 artifacts/；RLVR 的行为概率和更新见 rollouts.json。")
        lines.append("范围：本地小型字符文本模型、一次随机种子；年度项目的消融、独立重复和人工复核继续按工坊验收。")
        return "\n".join(lines)
    if lesson == "lm-resume":
        complete = data["comparisons"]["complete_resume"]
        return (f"完整续训首批数据一致={complete['first_batch_equal']}；"
                f"下一步参数误差={complete['next_step_parameter_max_abs_difference']:.3g}；"
                f"最终参数误差={complete['final_parameter_max_abs_difference']:.3g}。\n"
                f"漏优化器状态已检出={data['checks']['missing_optimizer_detected']}；"
                f"漏批次生成器状态已检出={data['checks']['missing_batch_generator_detected']}。\n"
                "验证范围：固定 CPU float64 的数字语法模型完整恢复；检查点和逐步对照 CSV 已保留。")
    if lesson == "data-asset":
        return (f"输入 {data['input_records']} 条，保留 {data['kept_records']} 条，"
                f"过滤 {data['filtered_records']} 条；划分数量={data['splits']}。\n"
                f"同输入重建一致={data['checks']['rebuild_identical']}；"
                f"题族与精确重复组不跨划分={data['checks']['group_disjoint']}。\n"
                "过滤原因与来源保留率见审计账本和 manifest；字符数不是 token 数，语义近重复需另行检查。")
    if lesson == "attention":
        return (f"注意力解析 VJP 最大误差={data['attention_vjp_max_error']:.3g}；"
                f"RoPE 增量缓存与完整因果计算误差={data['rope_cached_attention_max_error']:.3g}。\n"
                f"错误矩形 mask 的输出偏差={data['wrong_rectangular_mask_error']:.6f}；"
                f"在线 softmax 误差={data['online_softmax_max_error']:.3g}；"
                f"MLA 投影吸收误差={data['mla_projection_absorption_max_error']:.3g}。\n"
                "验证范围：CPU float64 核心函数；GPU 内核和完整模型训练是后续项目。")
    if lesson == "linear":
        return (f"训练均方误差：{data['training']['initial_train_mse']:.6f} → {data['training']['final_train_mse']:.3g}；"
                f"学到的权重与偏置：{data['training']['learned_weight_and_bias']}。\n"
                f"恢复后下一步参数误差：{data['recovery']['next_step_max_parameter_error']:.3g}；"
                f"漏掉优化器状态的错误对照偏差：{data['recovery']['without_optimizer_next_step_error']:.6f}。")
    if lesson == "lm":
        return "\n".join(f"学习率 {run['learning_rate']}：训练损失 {run['initial_train']['loss']:.6f} → "
                         f"{run['final_train']['loss']:.6f}；验证损失 {run['initial_validation']['loss']:.6f} → "
                         f"{run['final_validation']['loss']:.6f}。" for run in data["runs"])
    if lesson == "rl":
        return (f"学到的起点价值：{data['q_learning']['greedy_start_value']:.6f}；"
                f"最大动作价值误差：{data['q_learning']['max_q_error']:.3g}。\n"
                f"本次折扣={data['config']['gamma']:g}；解析最优起点价值={data['analytic_values']['start']:g}；"
                f"学到的策略={data['q_learning']['policy']}。")
    if lesson == "posttrain":
        return (f"数值检查：{data['checks_passed']} 项通过；DPO 损失={data['dpo_loss']:.9f}；"
                f"组内优势={data['grpo_advantages']}；两次采样成功率估计={data['pass_at_2']:.2f}。")
    if lesson == "agent":
        return f"工具状态机检查：{data['checks_passed']} 项通过。默认计算结果为 4；具体异常案例见原始日志。"
    return (f"只对策略动作计算的损失={data['mask']['correct_loss']:.2f}；"
            f"误计工具文本后的损失={data['mask']['incorrect_loss']:.6f}。\n"
            f"折扣 0.9 的逐步回报={data['returns']['gamma_0.9']}；"
            f"真终止目标={data['td_target']['terminated']:.2f}，工程截断目标={data['td_target']['truncated']:.2f}。")



def environment_info(executable: str, needs_torch: bool) -> dict:
    code = ("import json,platform,sys; "
            "info={'python':sys.version,'python_executable':sys.executable,"
            "'platform':platform.platform()}; ")
    if needs_torch:
        code += "import torch; info['pytorch']=torch.__version__; "
    code += "print(json.dumps(info))"
    completed = subprocess.run([executable, "-X", "utf8", "-c", code], cwd=ROOT,
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=30)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "无法取得实际环境版本")
    return json.loads(completed.stdout)


def source_versions() -> dict:
    """Identify the actual working sources, including new uncommitted modules."""
    hashes = {}
    for directory in ("scripts", "pytorch&mindspore", "transformer", "RL", "RNN",
                      "post_train", "agent_design", "agentic_rl"):
        for path in (ROOT / directory).rglob("*.py"):
            if "__pycache__" not in path.parts:
                hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {"python_source_sha256": dict(sorted(hashes.items()))}
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8",
            stderr=subprocess.DEVNULL, timeout=5).strip()
        result["working_tree_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, encoding="utf-8",
            stderr=subprocess.DEVNULL, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        result.update({"git_commit": None, "working_tree_dirty": None})
    return result


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raw = list(sys.argv[1:] if argv is None else argv)
    separator = raw.index("--") if "--" in raw else len(raw)
    runner_args = raw[:separator]
    experiment_args = raw[separator + 1:] if separator < len(raw) else []
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="实验参数写在 -- 后，例如：first -- --learning-rate 0.05",
        allow_abbrev=False)
    parser.add_argument("lesson", nargs="?", choices=LESSONS, help="课程代号，初次运行选 first")
    parser.add_argument("--list", action="store_true", help="列出课程，不运行实验")
    parser.add_argument("--check", action="store_true", help="只检查 Python / PyTorch，不训练")
    parser.add_argument("--python", dest="python_path", help="指定已有 Python；不可用时不换环境")
    parser.add_argument("--output-dir", type=Path, help="新目录或空目录；默认 study_runs 下独立保存")
    args = parser.parse_args(runner_args)
    if (args.list or args.check) and experiment_args:
        parser.error("--list / --check 不接收实验参数")
    if args.list:
        for name, (title, _, torch_needed, _) in LESSONS.items():
            print(f"{name:12} {title} [{'PyTorch / CPU' if torch_needed else '只需 Python 标准库'}]")
        return 0
    if args.check:
        print(f"项目目录：{ROOT}")
        plain, reports = choose_python(args.python_path, False)
        print("\n".join(reports))
        print("标准库课程：" + ("可以开始" if plain else "当前解释器不可用"))
        torch_python, reports = choose_python(args.python_path, True)
        print("\n".join(reports))
        print("PyTorch 课程：" + ("可以开始" if torch_python else "请指定已有环境，或先学标准库课程"))
        return 0 if plain and torch_python else 1
    if not args.lesson:
        parser.error("请选择课程，例如：python scripts/run_lesson.py first；查看课程用 --list")
    title, relative_script, needs_torch, supports_output = LESSONS[args.lesson]
    for option in experiment_args:
        # Child parsers also disable abbreviations, so --output-d cannot bypass this check.
        key = option.split("=", 1)[0]
        if key == "--output-dir":
            parser.error("输出目录由统一入口管理；请把 --output-dir 写在 -- 前")
        if args.lesson in MODEL_MODES and key == "--mode":
            parser.error("模型阶段由课程代号选择；请选 sft-model / dpo-model / rlvr-model")
    if not (ROOT / relative_script).is_file():
        print(f"课程程序不存在：{ROOT / relative_script}")
        return 2
    executable, reports = choose_python(args.python_path, needs_torch)
    print("\n".join(reports), flush=True)
    if executable is None:
        print("没有找到本课可用的解释器。查看 入门/环境准备与常见问题.md。", flush=True)
        return 2
    command = [executable, "-X", "utf8", str(ROOT / relative_script)]
    if args.lesson == "rl":
        command.extend(["--seed", "7", "--episodes", "20000", "--require-q-error", "0.000001"])
    if args.lesson in MODEL_MODES:
        command.extend(["--mode", MODEL_MODES[args.lesson]])
    env = os.environ.copy()
    env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    if any(option in {"--help", "-h"} for option in experiment_args):
        return subprocess.run(command + experiment_args, cwd=ROOT, env=env).returncode
    try:
        actual_environment = environment_info(executable, needs_torch)
        versions = source_versions()
        output = reserve_output_dir(args.output_dir, args.lesson)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"实验未启动：{error}", flush=True)
        return 2
    child_output = output / "artifacts"
    if supports_output:
        command.extend(["--output-dir", str(child_output)])
    command.extend(experiment_args)
    started = datetime.now().astimezone().isoformat()
    print(f"\n开始：{title}\n解释器：{executable}\n结果目录：{output}\n", flush=True)
    log_path = output / "console.txt"
    code = 2
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            try:
                if args.lesson == "project-a":
                    # Fresh Python/PyTorch workers need more startup time than the short labs.
                    try:
                        console, _ = process.communicate(timeout=600)
                        code = process.returncode
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try:
                            console, _ = process.communicate(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            console, _ = process.communicate()
                        code = 124
                        console = (console or "") + "\n项目 A 超过 600 秒；本次未通过，已有证据保留。\n"
                    print(console or "", end="", flush=True)
                    log.write(console or "")
                else:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                    code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                code = 130
            finally:
                if process.stdout is not None:
                    process.stdout.close()
    except OSError as error:
        log_path.write_text(str(error), encoding="utf-8")
        print(f"启动失败：{error}")
    summary_error = None
    summary_status = "not_generated"
    result_code = code
    if code == 0:
        try:
            summary = chinese_summary(args.lesson, output)
            (output / "结果说明.txt").write_text(summary + "\n", encoding="utf-8")
            summary_status = "passed"
            print(f"\n中文结果说明：\n{summary}")
        except (OSError, ValueError, KeyError, TypeError) as error:
            summary_status = "failed"
            summary_error = str(error)
            result_code = 1
            print(f"结果交付未完成（{error}）。请查看原始日志；本次不记为成功。")
    record = {
        "schema_version": 2, "lesson": args.lesson, "title": title, "command": command,
        "experiment_args": experiment_args, "cwd": str(ROOT), "started_at": started,
        "finished_at": datetime.now().astimezone().isoformat(),
        "process_exit_code": code, "exit_code": result_code,
        "program_success": code == 0, "success": result_code == 0,
        "summary_status": summary_status, "summary_error": summary_error,
        "console_log": str(log_path), "environment": actual_environment,
        "source_versions": versions,
    }
    if args.lesson == "project-a":
        record["process_timeout_seconds"] = 600
    if supports_output:
        record["artifact_dir"] = str(child_output)
    (output / "run.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if result_code:
        print(f"\n本次未通过（退出码 {result_code}）。错误日志：{log_path}")
    else:
        print(f"\n本课程序检查与结果交付通过。中文对照讲解：{ROOT / READ_NEXT[args.lesson]}")
        print("下一步：解释一个输出数字，再自己改一个条件，保留两次结果作比较。")
    return result_code


if __name__ == "__main__":
    raise SystemExit(main())
