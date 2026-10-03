"""Public command regressions: forwarded arguments and immutable old output."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_lesson.py"


class RunnerCommands(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="maedaily_runner_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_script(self, script, arguments):
        return subprocess.run([sys.executable, "-X", "utf8", str(script), *map(str, arguments)], cwd=ROOT,
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)

    def test_forwarded_learning_rate_changes_result_and_is_recorded(self):
        output = self.root / "changed_rate"
        completed = self.run_script(RUNNER, ["first", "--python", sys.executable, "--output-dir", output, "--", "--learning-rate", "0.05"])
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        record = json.loads((output / "run.json").read_text(encoding="utf-8"))
        artifact_dir = output / "artifacts"
        self.assertEqual(Path(record["artifact_dir"]), artifact_dir)
        result = json.loads((artifact_dir / "first_steps.json").read_text(encoding="utf-8"))
        self.assertEqual(result["learning_rate"], 0.05)
        self.assertAlmostEqual(result["after"]["loss"], 2.53125)
        position = record["command"].index("--learning-rate")
        self.assertEqual(record["command"][position + 1], "0.05")
        self.assertTrue(record["success"])
        self.assertEqual(record["experiment_args"], ["--learning-rate", "0.05"])
        self.assertEqual(record["summary_status"], "passed")
        self.assertEqual(Path(record["environment"]["python_executable"]).resolve(), Path(sys.executable).resolve())
        self.assertIn("python", record["environment"])
        source = "scripts/first_steps.py"
        self.assertEqual(record["source_versions"]["python_source_sha256"][source], hashlib.sha256((ROOT / source).read_bytes()).hexdigest())
        self.assertTrue((output / "console.txt").is_file())
        self.assertTrue((output / "结果说明.txt").is_file())
        self.assertIn("0.05", (output / "结果说明.txt").read_text(encoding="utf-8"))

    def test_runner_nonempty_directory_is_never_modified(self):
        output = self.root / "old"
        output.mkdir()
        sentinel = output / "summary.json"
        sentinel.write_bytes(b'{"old":true}\n')
        before = hashlib.sha256(sentinel.read_bytes()).hexdigest()
        completed = self.run_script(RUNNER, ["first", "--output-dir", output, "--python", sys.executable])
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(hashlib.sha256(sentinel.read_bytes()).hexdigest(), before)
        self.assertEqual(list(output.iterdir()), [sentinel])

    def test_unknown_child_argument_is_reported_as_failure(self):
        output = self.root / "unknown"
        completed = self.run_script(RUNNER, ["first", "--python", sys.executable, "--output-dir", output, "--", "--no-such-option"])
        self.assertNotEqual(completed.returncode, 0)
        record = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertFalse(record["success"])
        self.assertNotEqual(record["exit_code"], 0)

    def test_child_arguments_cannot_override_managed_output(self):
        output, escape = self.root / "safe", self.root / "escape"
        completed = self.run_script(RUNNER, ["first", "--python", sys.executable, "--output-dir", output, "--", "--output-dir", escape])
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(escape.exists())

    def test_output_abbreviation_cannot_escape_child_guard(self):
        output, escape = self.root / "safe", self.root / "escape"
        completed = self.run_script(RUNNER, ["first", "--python", sys.executable, "--output-dir", output, "--", "--out", escape])
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(escape.exists())

    def test_summary_delivery_failure_does_not_report_success(self):
        spec = importlib.util.spec_from_file_location("runner_summary_failure", RUNNER)
        runner = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            spec.loader.exec_module(runner)
        finally:
            sys.path.pop(0)
        output = self.root / "bad_summary"
        with patch.object(runner, "chinese_summary", side_effect=ValueError("invalid result schema")), contextlib.redirect_stdout(io.StringIO()):
            code = runner.main(["first", "--python", sys.executable, "--output-dir", str(output)])
        self.assertEqual(code, 1)
        record = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["process_exit_code"], 0)
        self.assertTrue(record["program_success"])
        self.assertFalse(record["success"])
        self.assertEqual(record["summary_status"], "failed")
        self.assertIn("invalid result schema", record["summary_error"])

    def test_standalone_output_scripts_reject_existing_artifacts(self):
        scripts = [("scripts/first_steps.py", False), ("scripts/data_asset_lab.py", False),
                   ("RL/experiments/tabular_mdp.py", False),
                   ("pytorch&mindspore/experiments/linear_train_lab.py", True),
                   ("transformer/experiments/tiny_causal_lm.py", True),
                   ("transformer/experiments/lm_resume_lab.py", True),
                   ("post_train/experiments/model_training_lab.py", True)]
        torch_available = importlib.util.find_spec("torch") is not None
        for index, (relative, needs_torch) in enumerate(scripts):
            if needs_torch and not torch_available:
                continue
            with self.subTest(script=relative):
                output = self.root / f"standalone_{index}"
                output.mkdir()
                sentinel = output / "summary.json"
                sentinel.write_bytes(b'{"existing_teaching_reference":true}\n')
                before = sentinel.read_bytes()
                completed = self.run_script(ROOT / relative, ["--output-dir", output])
                self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                self.assertEqual(sentinel.read_bytes(), before)
                self.assertEqual(list(output.iterdir()), [sentinel])

    @staticmethod
    def load_runner():
        spec = importlib.util.spec_from_file_location("project_a_runner_tests", RUNNER)
        runner = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            spec.loader.exec_module(runner)
        finally:
            sys.path.pop(0)
        return runner

    @staticmethod
    def project_a_summary():
        package = ROOT / "references" / "evidence" / "project_a_20261002"
        data = json.loads((package / "evidence.json").read_text(encoding="utf-8"))
        data["status"] = "mechanism_passed"
        return data

    def test_project_a_registration_and_summary_use_actual_schema_and_metrics(self):
        runner = self.load_runner()
        self.assertEqual(runner.LESSONS["project-a"][1:],
                         ("transformer/experiments/project_a_reference.py", True, True))
        output = self.root / "project_a_summary"
        (output / "artifacts").mkdir(parents=True)
        data = self.project_a_summary()
        data["quality"]["base_lr"]["final_validation"]["loss"] = 0.456789
        (output / "artifacts" / "summary.json").write_text(json.dumps(data), encoding="utf-8")
        summary = runner.chinese_summary("project-a", output)
        self.assertIn("0.456789", summary)
        self.assertIn("6 个", summary)
        self.assertIn("mechanism_passed", summary)
        self.assertIn("合成数字语法", summary)

    def test_project_a_incomplete_mechanism_cannot_generate_success_summary(self):
        runner = self.load_runner()
        output = self.root / "unqualified_a"
        (output / "artifacts").mkdir(parents=True)
        for kind in ("status", "steps", "optimizer"):
            data = self.project_a_summary()
            if kind == "status":
                data["status"] = "failed"
            elif kind == "steps":
                data["mechanism_acceptance"]["every_resumed_step_exact"] = False
            else:
                data["mechanism_acceptance"]["final_state_exact"]["optimizer_state"] = False
            (output / "artifacts" / "summary.json").write_text(json.dumps(data), encoding="utf-8")
            with self.subTest(case=kind), self.assertRaisesRegex(ValueError, "机制尚未通过"):
                runner.chinese_summary("project-a", output)

    def test_project_a_forwarding_uses_managed_output_and_long_timeout(self):
        runner = self.load_runner()
        output = self.root / "project_a_forwarding"
        data = self.project_a_summary()
        process = unittest.mock.Mock()
        process.returncode = 0
        process.communicate.return_value = ("real mechanism completion text\n", None)

        def launch(command, **kwargs):
            artifact = Path(command[command.index("--output-dir") + 1])
            artifact.mkdir()
            (artifact / "summary.json").write_text(json.dumps(data), encoding="utf-8")
            return process

        with patch.object(runner, "choose_python", return_value=(sys.executable, [])), \
                patch.object(runner, "environment_info", return_value={"python_executable": sys.executable}), \
                patch.object(runner, "source_versions", return_value={}), \
                patch.object(runner.subprocess, "Popen", side_effect=launch), \
                contextlib.redirect_stdout(io.StringIO()):
            code = runner.main(["project-a", "--output-dir", str(output), "--", "--steps", "4", "--split-step", "2"])
        self.assertEqual(code, 0)
        process.communicate.assert_called_once_with(timeout=600)
        record = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertTrue(record["success"])
        self.assertEqual(record["process_timeout_seconds"], 600)
        self.assertEqual(record["experiment_args"], ["--steps", "4", "--split-step", "2"])
        self.assertEqual(Path(record["artifact_dir"]), output / "artifacts")
        self.assertEqual(record["command"][-4:], ["--steps", "4", "--split-step", "2"])
        self.assertNotIn("--mode", record["command"])

    def test_project_a_timeout_is_failed_and_preserves_partial_log(self):
        runner = self.load_runner()
        output = self.root / "project_a_timeout"
        process = unittest.mock.Mock()
        process.communicate.side_effect = [subprocess.TimeoutExpired(["project-a"], 600), ("partial evidence retained\n", None)]
        with patch.object(runner, "choose_python", return_value=(sys.executable, [])), \
                patch.object(runner, "environment_info", return_value={}), \
                patch.object(runner, "source_versions", return_value={}), \
                patch.object(runner.subprocess, "Popen", return_value=process), \
                patch.object(runner, "chinese_summary") as summary, \
                contextlib.redirect_stdout(io.StringIO()):
            code = runner.main(["project-a", "--output-dir", str(output)])
        self.assertEqual(code, 124)
        process.terminate.assert_called_once()
        summary.assert_not_called()
        record = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertFalse(record["success"])
        self.assertFalse(record["program_success"])
        self.assertIn("partial evidence retained", (output / "console.txt").read_text(encoding="utf-8"))
        self.assertFalse((output / "结果说明.txt").exists())

    def test_public_project_a_package_is_stdlib_verifiable_and_tampering_fails(self):
        import shutil
        package = ROOT / "references" / "evidence" / "project_a_20261002"
        copied = self.root / "public_package"
        shutil.copytree(package, copied)
        valid = self.run_script(copied / "verify.py", [])
        self.assertEqual(valid.returncode, 0, valid.stdout + valid.stderr)
        csv_path = copied / "step_comparison.csv"
        csv_path.write_bytes(csv_path.read_bytes() + b"tampered\n")
        invalid = self.run_script(copied / "verify.py", [])
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("SHA256", invalid.stderr)


if __name__ == "__main__":
    unittest.main()
