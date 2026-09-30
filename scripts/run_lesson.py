"""中文实验入口：检查已有环境、运行指定课、为每次运行保留独立日志。"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LESSONS = {
    "first": ("第一课：手算一次学习", "scripts/first_steps.py", False, True),
    "linear": ("框架：梯度、线性训练与恢复", "pytorch&mindspore/experiments/linear_train_lab.py", True, True),
    "lm": ("语言模型：数字语法与学习率对照", "transformer/experiments/tiny_causal_lm.py", True, True),
    "rl": ("强化学习：表格决策与 Q-learning", "RL/experiments/tabular_mdp.py", False, True),
    "posttrain": ("后训练：概率、DPO 与组优势核算", "post_train/experiments/posttraining_math_lab.py", False, False),
    "agent": ("Agent：工具调用与错误恢复", "agent_design/experiments/tool_state_machine.py", False, False),
    "trajectory": ("Agentic RL：轨迹、mask 与回报核算", "agentic_rl/experiments/trajectory_accounting.py", False, False),
}
READ_NEXT = {
    "first": "入门/第一课_从预测到学习.md",
    "linear": "pytorch&mindspore/从零上手.md",
    "lm": "transformer/从零上手.md",
    "rl": "RL/从零上手.md",
    "posttrain": "post_train/从零上手.md",
    "agent": "入门/工具交互第一课.md",
    "trajectory": "入门/工具交互第一课.md",
}


def probe(executable: str, need_torch: bool) -> tuple[bool, str]:
    code = "import sys; assert sys.version_info >= (3,10), 'Python 需要 3.10 或更新版本'; print(sys.version.split()[0])"
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
    known = Path("D:/anaconda/envs/pytorch_env/python.exe")
    if not explicit and need_torch and known.is_file():
        candidates.append(str(known))
    reports, seen = [], set()
    for candidate in candidates:
        executable = shutil.which(candidate) or candidate
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
    filename = "summary.json" if lesson in {"linear", "lm", "rl"} else "console.txt"
    if lesson == "first":
        return "已完成预测、梯度更新、概率与折扣回报的第一次数值练习。"
    data = json.loads((output / filename).read_text(encoding="utf-8"))
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
                "默认任务的解析最优方案：先投入，再完成；回报为 1.6。")
    if lesson == "posttrain":
        return (f"数值检查：{data['checks_passed']} 项通过；DPO 损失={data['dpo_loss']:.9f}；"
                f"组内优势={data['grpo_advantages']}；两次采样成功率估计={data['pass_at_2']:.2f}。")
    if lesson == "agent":
        return f"工具状态机检查：{data['checks_passed']} 项通过。默认计算结果为 4；具体异常案例见原始日志。"
    return (f"只对策略动作计算的损失={data['mask']['correct_loss']:.2f}；"
            f"误计工具文本后的损失={data['mask']['incorrect_loss']:.6f}。\n"
            f"折扣 0.9 的逐步回报={data['returns']['gamma_0.9']}；"
            f"真终止目标={data['td_target']['terminated']:.2f}，工程截断目标={data['td_target']['truncated']:.2f}。")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lesson", nargs="?", choices=LESSONS, help="课程代号，初次运行选 first")
    parser.add_argument("--list", action="store_true", help="列出课程，不运行实验")
    parser.add_argument("--check", action="store_true", help="只检查 Python / PyTorch，不训练")
    parser.add_argument("--python", dest="python_path", help="明确指定已有 Python 解释器；不可用时不偷偷换环境")
    parser.add_argument("--output-dir", type=Path, help="新目录或空目录；默认 study_runs 下自动创建独立目录")
    args = parser.parse_args()
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
        print("PyTorch 课程：" + ("可以开始" if torch_python else "请用 --python 指定已有环境，或先学标准库课程"))
        # --check 检查完整环境；缺少 torch 会返回非零，标准库课仍可单独运行。
        return 0 if plain and torch_python else 1
    if not args.lesson:
        parser.error("请选择课程，例如：python scripts/run_lesson.py first；查看课程用 --list")
    title, relative_script, needs_torch, supports_output = LESSONS[args.lesson]
    executable, reports = choose_python(args.python_path, needs_torch)
    print("\n".join(reports), flush=True)
    if executable is None:
        print("没有找到本课可用的解释器。查看 入门/环境准备与常见问题.md，或先运行 first。", flush=True)
        return 2
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = args.output_dir.expanduser().resolve() if args.output_dir else ROOT / "study_runs" / f"{stamp}_{args.lesson}"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        print(f"结果目录不是空目录，已停止以保留已有记录：{output}")
        print("请换一个 --output-dir，或省略该参数自动创建新目录。")
        return 2
    output.mkdir(parents=True, exist_ok=True)
    command = [executable, "-X", "utf8", str(ROOT / relative_script)]
    if args.lesson == "rl":
        command.extend(["--seed", "7", "--episodes", "20000", "--require-q-error", "0.000001"])
    if supports_output:
        command.extend(["--output-dir", str(output)])
    env = os.environ.copy()
    env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    started = datetime.now().astimezone().isoformat()
    print(f"\n开始：{title}\n解释器：{executable}\n结果目录：{output}\n", flush=True)
    log_path = output / "console.txt"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            try:
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
    except OSError as error:
        log_path.write_text(str(error), encoding="utf-8")
        print(f"启动失败：{error}")
        code = 2
    record = {"lesson": args.lesson, "title": title, "command": command, "cwd": str(ROOT),
              "started_at": started, "finished_at": datetime.now().astimezone().isoformat(),
              "exit_code": code, "success": code == 0, "console_log": str(log_path)}
    (output / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if code:
        print(f"\n本次未通过（退出码 {code}）。错误日志：{log_path}")
    else:
        try:
            summary = chinese_summary(args.lesson, output)
            (output / "结果说明.txt").write_text(summary + "\n", encoding="utf-8")
            print(f"\n中文结果说明：\n{summary}")
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"中文字段解析未完成（{error}），请按本课说明查看完整日志。")
        print(f"\n本课程序检查通过。中文对照讲解：{ROOT / READ_NEXT[args.lesson]}")
        print("下一步：解释一个输出数字，再自己改一个条件；不要把程序通过直接记为掌握。")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
