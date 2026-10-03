"""Repository gates tested with deliberately broken public-source fixtures."""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_repository", ROOT / "scripts" / "check_repository.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


class RepositoryChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="maedaily_checks_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def put(self, name, text):
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def issues(self, names):
        return checks.validate_repository(self.root, names)["issues"]

    def test_existing_ignored_target_is_not_a_public_link(self):
        self.put("README.md", "[personal goal](GOAL.md)\n")
        self.put("GOAL.md", "# Private local goal\n")
        issues = self.issues(["README.md"])
        self.assertEqual([item["code"] for item in issues], ["unpublished-target"])
        self.assertEqual(issues[0]["line"], 1)
        # The clean-copy gate also fails when that private file is absent.
        clean = checks.clean_copy_check(self.root, ["README.md"])
        self.assertFalse(clean["success"])
        self.assertEqual(clean["issues"][0]["code"], "missing-link")

    def test_chinese_duplicate_explicit_anchors_and_encoded_spaces(self):
        self.put("README.md", "[first](notes/a%20(1).md#重复标题)\n[second](notes/a%20(1).md#重复标题-1)\n[stable](notes/a%20(1).md#project-a)\n[inline](notes/a%20(1).md#use-var_name)\n")
        self.put("notes/a (1).md", '# 重复标题\n# 重复标题\n<a id="project-a"></a>\n# Use `var_name`\n')
        self.assertEqual(self.issues(["README.md", "notes/a (1).md"]), [])

    def test_missing_target_and_fragment_report_the_source_line(self):
        self.put("README.md", "# Intro\n[bad file](missing.md)\n[bad section](chapter.md#missing)\n")
        self.put("chapter.md", "# Existing\n")
        issues = self.issues(["README.md", "chapter.md"])
        self.assertEqual([(item["line"], item["code"]) for item in issues], [(2, "missing-link"), (3, "missing-anchor")])

    def test_reference_images_and_parenthesized_filenames(self):
        self.put("README.md", '[chapter][ch]\n![diagram](assets/plot(2).png)\n[angle](<chapter (2).md> "title")\n[ch]: chapter%20(2).md#result\n')
        self.put("chapter (2).md", "Result\n======\n")
        self.put("assets/plot(2).png", "synthetic image bytes")
        self.assertEqual(self.issues(["README.md", "chapter (2).md", "assets/plot(2).png"]), [])

    def test_code_examples_are_not_document_links(self):
        self.put("README.md", "`[sample](absent.md)`\n```markdown\n[example](absent.md)\n# Hidden\n```\n    [indented](absent.md)\n[real](#visible)\n# Visible\n")
        self.assertEqual(self.issues(["README.md"]), [])
        self.assertNotIn("hidden", checks.markdown_anchors((self.root / "README.md").read_text()))

    def test_html_comments_are_not_document_links_or_anchors(self):
        self.put("README.md", "<!-- [hidden](missing.md)\n# Hidden\n-->\n# Public\n[here](#public)\n")
        self.assertEqual(self.issues(["README.md"]), [])
        self.assertNotIn("hidden", checks.markdown_anchors((self.root / "README.md").read_text()))

    def test_invalid_encoding_in_anchor_target_is_reported_without_crashing(self):
        self.put("README.md", "[section](broken.md#missing)\n")
        (self.root / "broken.md").write_bytes(b"\xff\xfeinvalid")
        issues = self.issues(["README.md", "broken.md"])
        self.assertEqual({item["code"] for item in issues}, {"invalid.md", "unreadable-anchor-target"})

    def test_directory_links_need_public_content(self):
        self.put("README.md", "[notes](notes/)\n[private](private/)\n")
        self.put("notes/topic.md", "# Topic\n")
        self.put("private/local.md", "# Local\n")
        issues = self.issues(["README.md", "notes/topic.md"])
        self.assertEqual([item["code"] for item in issues], ["unpublished-target"])

    def test_link_cannot_escape_repository(self):
        self.put("README.md", "[outside](../other.md)\n")
        self.assertEqual(self.issues(["README.md"])[0]["code"], "outside-link")

    def test_machine_local_file_link_is_not_publishable(self):
        self.put("README.md", "[local](file:///C:/Users/local.md)\n[drive](D:/local.md)\n")
        self.assertEqual([item["code"] for item in self.issues(["README.md"])], ["outside-link", "outside-link"])

    def test_strict_json_rejects_duplicates_and_nonfinite_constants(self):
        for source in ['{"value":1,"value":2}', '{"nested":{"value":1,"value":2}}', '{"value":NaN}', '{"value":Infinity}', '{"value":-Infinity}', '{"value":1e999}']:
            with self.subTest(source=source), self.assertRaises(ValueError):
                checks.strict_json(source)
        self.assertEqual(checks.strict_json('{"value":0,"nested":[true,null,1.5]}')["value"], 0)

    def test_compilation_catches_context_errors_in_python(self):
        self.put("broken.py", "continue\n")
        self.put("broken.json", '{"x":NaN}\n')
        issues = self.issues(["broken.py", "broken.json"])
        self.assertEqual({item["code"] for item in issues}, {"invalid.py", "invalid.json"})

    def test_pdf_receipt_detects_same_size_content_tampering(self):
        data = b"%PDF-1.7\noriginal fixture\n"
        asset = self.root / "papers" / "book.pdf"
        asset.parent.mkdir()
        asset.write_bytes(data)
        catalog = {"fixture": {"status": "ok", "path": "papers/book.pdf", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}
        self.put("references/downloads.json", json.dumps(catalog))
        names = ["references/downloads.json", "papers/book.pdf"]
        good = checks.validate_repository(self.root, names)
        self.assertTrue(good["success"])
        self.assertEqual(good["pdf_integrity_checked"], 1)
        asset.write_bytes(data.replace(b"original", b"modified"))
        broken = checks.validate_repository(self.root, names)
        self.assertFalse(broken["success"])
        self.assertEqual(broken["issues"][0]["code"], "download-integrity")

    @unittest.skipUnless(shutil.which("git"), "git required for public-inventory fixture")
    def test_git_inventory_distinguishes_pending_public_files_from_ignored_files(self):
        subprocess.run(["git", "init", "--quiet", str(self.root)], check=True)
        self.put(".gitignore", "GOAL.md\nstudy_runs/\n")
        self.put("README.md", "# Public\n")
        subprocess.run(["git", "-C", str(self.root), "add", ".gitignore", "README.md"], check=True)
        self.put("new.md", "# Pending public source\n")
        self.put("GOAL.md", "# Local only\n")
        self.put("study_runs/receipt.json", "{}\n")
        visible = checks.git_files(self.root)
        self.assertIn("new.md", visible)
        self.assertNotIn("GOAL.md", visible)
        self.assertNotIn("study_runs/receipt.json", visible)
        self.assertNotIn("new.md", checks.git_files(self.root, tracked_only=True))

    @unittest.skipUnless(shutil.which("git"), "git required for archive fixture")
    def test_archive_gate_rejects_a_link_hidden_by_an_untracked_local_file(self):
        subprocess.run(["git", "init", "--quiet", str(self.root)], check=True)
        self.put("README.md", "[local](local.md)\n")
        subprocess.run(["git", "-C", str(self.root), "add", "README.md"], check=True)
        subprocess.run(["git", "-C", str(self.root), "-c", "user.name=Repository Gate", "-c", "user.email=gate@example.invalid", "-c", "commit.gpgsign=false", "-c", f"core.hooksPath={self.root / 'empty_hooks'}", "commit", "--quiet", "-m", "fixture"], check=True)
        self.put("local.md", "# Local\n")
        self.assertTrue(checks.validate_repository(self.root, checks.git_files(self.root))["success"])
        archived = checks.archive_check(self.root, "HEAD")
        self.assertFalse(archived["success"])
        self.assertEqual(archived["issues"][0]["code"], "missing-link")


if __name__ == "__main__":
    unittest.main()
