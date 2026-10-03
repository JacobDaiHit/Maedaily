"""Portable repository checks; no network, installs, or experiment overwrites.

The default inventory includes tracked files and visible, nonignored new files.
Use --tracked-only for a publication gate, --clean-copy to check the pending
worktree without personal files, and --archive-ref HEAD to verify a Git archive.
"""
from __future__ import annotations

import argparse
from collections import Counter
from html import unescape
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unicodedata
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def strict_json(text: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"nonfinite JSON number: {value}")
        return parsed

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant, parse_float=finite_float)


def git_files(root: Path, tracked_only: bool = False) -> list[str]:
    command = ["git", "-C", str(root), "ls-files", "-z", "--cached"]
    if not tracked_only:
        command += ["--others", "--exclude-standard"]
    result = subprocess.run(command, capture_output=True, check=True)
    return sorted(set(item.decode("utf-8") for item in result.stdout.split(b"\0") if item))


def prose_lines(text: str):
    """Remove fenced and inline code while keeping original line numbers."""
    text = re.sub(r"<!--[\s\S]*?(?:-->|$)", lambda match: "\n" * match.group(0).count("\n"), text)
    fence = None
    for number, raw in enumerate(text.splitlines(), 1):
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", raw)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = (token[0], len(token))
            elif token[0] == fence[0] and len(token) >= fence[1]:
                fence = None
            yield number, ""
            continue
        if fence or raw.startswith("    ") or raw.startswith("\t"):
            yield number, ""
            continue
        yield number, re.sub(r"(`+).*?\1", "", raw)


def _balanced_end(line: str, start: int, opening: str, closing: str) -> int | None:
    level, escaped = 0, False
    for index in range(start, len(line)):
        char = line[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
        elif char == opening:
            level += 1
        elif char == closing:
            level -= 1
            if level == 0:
                return index
    return None


def _destination(body: str) -> str:
    body = body.strip()
    if body.startswith("<") and ">" in body:
        value = body[1:body.index(">")]
    else:
        # A quoted optional link title follows whitespace; parentheses inside
        # a URL were already handled by the balanced-link scanner.
        value = re.split(r"\s+", body, maxsplit=1)[0] if body else ""
    return re.sub(r"\\([\\()\[\]<> ])", r"\1", value)


def markdown_links(text: str):
    """Yield (line, target) for inline, image, and reference Markdown links."""
    lines = list(prose_lines(text))
    definitions = {}
    for number, line in lines:
        match = re.match(r"^\s{0,3}\[([^\]]+)\]:\s*(.*)$", line)
        if match:
            definitions[match.group(1).strip().casefold()] = _destination(match.group(2))
    for number, line in lines:
        if re.match(r"^\s{0,3}\[[^\]]+\]:", line):
            continue
        cursor = 0
        while cursor < len(line):
            start = line.find("[", cursor)
            if start < 0:
                break
            if start and line[start - 1] == "\\":
                cursor = start + 1
                continue
            end = _balanced_end(line, start, "[", "]")
            if end is None:
                break
            label = line[start + 1:end]
            cursor = end + 1
            if cursor < len(line) and line[cursor] == "(":
                close = _balanced_end(line, cursor, "(", ")")
                if close is not None:
                    yield number, _destination(line[cursor + 1:close])
                    cursor = close + 1
            elif cursor < len(line) and line[cursor] == "[":
                close = line.find("]", cursor + 1)
                if close >= 0:
                    key = (line[cursor + 1:close] or label).strip().casefold()
                    if key in definitions:
                        yield number, definitions[key]
                    cursor = close + 1
            elif label.strip().casefold() in definitions:
                yield number, definitions[label.strip().casefold()]


def heading_slug(heading: str) -> str:
    heading = re.sub(r"<[^>]+>", "", heading)
    heading = re.sub(r"!?\[([^\]]+)\]\([^)]*\)", r"\1", heading)
    heading = unescape(heading).lower().strip()
    heading = re.sub(r"[*`~]", "", heading)
    # GitHub retains Chinese text, underscores and hyphens, but removes most
    # other punctuation and symbols before replacing spaces with hyphens.
    heading = "".join(c for c in heading if c in "-_" or not unicodedata.category(c).startswith(("P", "S")))
    return heading.replace(" ", "-")


def markdown_anchors(text: str) -> set[str]:
    anchors, repeated = set(), Counter()
    lines = list(prose_lines(text))
    raw_lines = re.sub(r"<!--[\s\S]*?(?:-->|$)", lambda match: "\n" * match.group(0).count("\n"), text).splitlines()
    previous = ""
    for number, line in lines:
        for match in re.finditer(r"<(?:a|[a-z][\w-]*)\b[^>]*\b(?:id|name)\s*=\s*['\"]([^'\"]+)['\"]", line, re.I):
            anchors.add(unescape(match.group(1)))
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        heading = None
        if match:
            # Inline code contributes its text to a heading's anchor.
            raw_match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", raw_lines[number - 1])
            heading = raw_match.group(1)
        elif re.match(r"^\s{0,3}(?:=+|-+)\s*$", line) and previous.strip():
            heading = previous.strip()
        if heading is not None:
            base = heading_slug(heading)
            suffix = repeated[base]
            repeated[base] += 1
            anchors.add(f"{base}-{suffix}" if suffix else base)
        previous = line
    return anchors


def validate_repository(root: Path, file_names: list[str]) -> dict:
    root = root.resolve()
    inventory = set(file_names)
    issues, counts = [], Counter()
    texts, anchor_cache = {}, {}

    def issue(name, line, code, message):
        issues.append({"path": name, "line": line, "code": code, "message": message})

    for name in sorted(inventory):
        source = root / name
        if not source.is_file():
            issue(name, 0, "missing-source", "inventory file is missing from checkout")
            continue
        if not source.resolve().is_relative_to(root):
            issue(name, 0, "outside-source", "source resolves outside repository")
            continue
        suffix = source.suffix.lower()
        if suffix not in {".py", ".json", ".md"}:
            continue
        try:
            text = source.read_text(encoding="utf-8-sig")
            counts[suffix] += 1
            if suffix == ".py":
                compile(text, name, "exec", dont_inherit=True)
            elif suffix == ".json":
                strict_json(text)
            else:
                texts[name] = text
        except (OSError, UnicodeError, SyntaxError, ValueError) as error:
            issue(name, getattr(error, "lineno", 0) or 0, "invalid" + suffix, str(error))

    for name, text in texts.items():
        for line, raw_target in markdown_links(text):
            counts["links"] += 1
            try:
                parts = urlsplit(unescape(raw_target))
            except ValueError as error:
                issue(name, line, "invalid-link", str(error))
                continue
            if parts.scheme.lower() == "file" or (len(parts.scheme) == 1 and parts.scheme.isalpha()):
                issue(name, line, "outside-link", f"machine-local path cannot be published: {raw_target}")
                continue
            if parts.scheme or parts.netloc:
                continue
            path_text = unquote(parts.path)
            if not path_text and not parts.fragment:
                issue(name, line, "empty-link", "link has no target")
                continue
            target = ((root / path_text.lstrip("/")) if path_text.startswith("/")
                      else (root / name).parent / path_text) if path_text else root / name
            target = target.resolve()
            if not target.is_relative_to(root):
                issue(name, line, "outside-link", f"target escapes repository: {raw_target}")
                continue
            relative = target.relative_to(root).as_posix()
            visible = relative in inventory or (target.is_dir() and (bool(inventory) if relative == "." else any(item.startswith(relative.rstrip("/") + "/") for item in inventory)))
            if not visible:
                code = "unpublished-target" if target.exists() else "missing-link"
                issue(name, line, code, f"target unavailable in public source inventory: {raw_target}")
                continue
            if not target.exists():
                issue(name, line, "missing-link", f"target missing from checkout: {raw_target}")
            elif parts.fragment and target.suffix.lower() == ".md":
                if relative not in anchor_cache:
                    if relative not in texts:
                        issue(name, line, "unreadable-anchor-target", f"anchor target failed Markdown decoding: {raw_target}")
                        continue
                    anchor_cache[relative] = markdown_anchors(texts[relative])
                anchor = unquote(parts.fragment)
                if anchor not in anchor_cache[relative]:
                    issue(name, line, "missing-anchor", f"anchor not present: {raw_target}")
    # A committed PDF catalog is an integrity contract, including offline use.
    catalog_name = "references/downloads.json"
    pdf_checked = 0
    if catalog_name in inventory:
        try:
            catalog = strict_json((root / catalog_name).read_text(encoding="utf-8-sig"))
            if not isinstance(catalog, dict):
                raise ValueError("download catalog must be a JSON object")
            recorded = set()
            for identifier, entry in catalog.items():
                if not isinstance(entry, dict) or entry.get("status") != "ok":
                    issue(catalog_name, 0, "invalid-download", f"unverified download: {identifier}")
                    continue
                name = entry.get("path")
                if not isinstance(name, str) or name not in inventory:
                    issue(catalog_name, 0, "unpublished-download", f"download not in public source inventory: {identifier}")
                    continue
                asset = root / name
                if not asset.is_file() or not asset.resolve().is_relative_to(root):
                    issue(catalog_name, 0, "missing-download", f"download unavailable: {name}")
                    continue
                recorded.add(name)
                data = asset.read_bytes()
                if len(data) != entry.get("bytes") or hashlib.sha256(data).hexdigest() != entry.get("sha256"):
                    issue(catalog_name, 0, "download-integrity", f"size or SHA-256 mismatch: {name}")
                elif not data.startswith(b"%PDF-"):
                    issue(catalog_name, 0, "invalid-pdf", f"download lacks PDF signature: {name}")
                else:
                    pdf_checked += 1
            for name in sorted(inventory):
                if name.lower().endswith(".pdf") and name not in recorded:
                    issue(catalog_name, 0, "uncatalogued-pdf", f"PDF has no integrity receipt: {name}")
        except (OSError, UnicodeError, ValueError) as error:
            issue(catalog_name, 0, "invalid-download-catalog", str(error))
    return {"root": str(root), "files": len(inventory), "python": counts[".py"], "json": counts[".json"],
            "markdown": counts[".md"], "links": counts["links"], "pdf_integrity_checked": pdf_checked, "issues": issues, "success": not issues}


def archive_check(root: Path, ref: str) -> dict:
    completed = subprocess.run(["git", "-C", str(root), "archive", "--format=tar", ref], capture_output=True, check=True)
    with tempfile.TemporaryDirectory(prefix="maedaily_archive_") as temporary:
        destination = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:") as archive:
            names = []
            for member in archive.getmembers():
                target = (destination / member.name).resolve()
                if not target.is_relative_to(destination.resolve()) or member.issym() or member.islnk():
                    raise ValueError(f"unsafe archive entry: {member.name}")
                if member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError(f"missing archive content: {member.name}")
                    with stream, target.open("wb") as output:
                        shutil.copyfileobj(stream, output)
                    names.append(member.name)
        result = validate_repository(destination, names)
        result["root"] = f"git archive {ref}"
        return result


def clean_copy_check(root: Path, names: list[str]) -> dict:
    with tempfile.TemporaryDirectory(prefix="maedaily_clean_") as temporary:
        destination = Path(temporary)
        for name in names:
            source = root / name
            if source.is_file() and source.resolve().is_relative_to(root.resolve()):
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        result = validate_repository(destination, names)
        result["root"] = "clean copy of public source inventory"
        return result


def file_hashes(root: Path, names: list[str]) -> dict[str, str]:
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in names if (root / name).is_file()}


def smoke_check(root: Path, names: list[str], mode: str, torch_python: str | None) -> dict:
    spec = importlib.util.spec_from_file_location("maedaily_runner_checks", root / "scripts" / "run_lesson.py")
    if spec is None or spec.loader is None:
        raise ValueError("cannot import course inventory")
    runner = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root / "scripts"))
    try:
        spec.loader.exec_module(runner)
    finally:
        sys.path.pop(0)
    before = file_hashes(root, names)
    runs = []
    with tempfile.TemporaryDirectory(prefix="maedaily_smoke_") as temporary:
        for lesson, (_, _, need_torch, _) in runner.LESSONS.items():
            if need_torch and mode == "stdlib":
                continue
            executable = (torch_python or sys.executable) if need_torch else sys.executable
            output = Path(temporary) / lesson
            command = [sys.executable, "-X", "utf8", str(root / "scripts" / "run_lesson.py"), lesson,
                       "--python", executable, "--output-dir", str(output)]
            try:
                completed = subprocess.run(command, cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600 if lesson == "project-a" else 240)
                success = completed.returncode == 0 and (output / "run.json").is_file() and (output / "结果说明.txt").is_file()
                runs.append({"lesson": lesson, "exit_code": completed.returncode, "success": success,
                             "error": "" if success else (completed.stdout + completed.stderr)[-4000:]})
            except (OSError, subprocess.TimeoutExpired) as error:
                runs.append({"lesson": lesson, "success": False, "error": str(error)})
    changed = [name for name in before if not (root / name).is_file() or file_hashes(root, [name]).get(name) != before[name]]
    return {"mode": mode, "runs": runs, "changed_source_files": changed, "success": all(run["success"] for run in runs) and not changed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--tracked-only", action="store_true")
    parser.add_argument("--clean-copy", action="store_true")
    parser.add_argument("--archive-ref", metavar="REF")
    parser.add_argument("--tests", action="store_true", help="run unittest discovery in the current interpreter")
    parser.add_argument("--smoke", choices=("stdlib", "all"))
    parser.add_argument("--torch-python", help="existing interpreter for torch courses when --smoke all is selected")
    parser.add_argument("--json", dest="json_path", type=Path, help="write the complete report to this file")
    args = parser.parse_args()
    root = args.root.resolve()
    report = {}
    try:
        names = git_files(root, args.tracked_only)
        report["repository"] = validate_repository(root, names)
        if args.clean_copy:
            report["clean_copy"] = clean_copy_check(root, names)
        if args.archive_ref:
            report["archive"] = archive_check(root, args.archive_ref)
        if args.tests:
            completed = subprocess.run([sys.executable, "-X", "utf8", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"], cwd=root)
            report["tests"] = {"success": completed.returncode == 0, "exit_code": completed.returncode}
        if args.smoke:
            report["smoke"] = smoke_check(root, names, args.smoke, args.torch_python)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        report["setup"] = {"success": False, "error": str(error)}
    success = all(section["success"] for section in report.values())
    report["success"] = success
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, section in report.items():
        if isinstance(section, dict):
            print(f"{name}: {'PASS' if section['success'] else 'FAIL'}")
            for issue in section.get("issues", []):
                print(f"  {issue['path']}:{issue['line']} [{issue['code']}] {issue['message']}")
            if section.get("error"):
                print("  " + section["error"])
            if name == "repository":
                print(f"  {section['files']} source files; {section['python']} Python; {section['json']} JSON; {section['markdown']} Markdown; {section['links']} links")
            for run in section.get("runs", []):
                print(f"  {run['lesson']}: {'PASS' if run['success'] else 'FAIL'}")
                if not run['success']:
                    print("  " + run['error'])
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
