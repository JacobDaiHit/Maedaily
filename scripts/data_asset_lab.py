"""Build a small reproducible teaching data asset with an auditable grouped split.

Standard library only. This is a data-pipeline exercise, not model training,
semantic near-duplicate detection, or a substitute for a real tokenizer.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

from lesson_runtime import reserve_output_dir

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("id", "family_id", "source", "license", "prompt", "completion")


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalized(text: str) -> str:
    # Preserve internal whitespace and case; code and symbolic tasks can depend on them.
    return unicodedata.normalize("NFC", text).strip()


def demo_records() -> list[dict[str, str]]:
    rows = []
    for number in range(40):
        rows.append({"id": f"v{number:03d}", "family_id": f"addition-{number}",
                     "source": "original_teaching", "license": "teaching_placeholder",
                     "prompt": f"Compute {number} + 1.", "completion": str(number + 1),
                     "expected_answer": str(number + 1)})
    for number in range(4):
        rows.append({**rows[number], "id": f"paraphrase-{number}", "prompt": f"What is one more than {number}?"})
    rows.append({**rows[1], "id": "z-duplicate", "source": "second_teaching_source",
                 "family_id": "duplicate-under-another-family", "prompt": "  Compute 1 + 1.  "})
    rows.append({**rows[7], "id": "bad-label", "completion": "9"})
    rows.append({**rows[0], "id": "bad-empty", "completion": ""})
    rows.append({**rows[0], "id": "bad-long", "family_id": "long-example", "prompt": "x" * 1000})
    for answer in ("4", "5"):
        rows.append({**rows[0], "id": f"conflict-{answer}", "family_id": f"conflict-family-{answer}",
                     "prompt": "Compute 2 + 2.", "completion": answer, "expected_answer": answer})
    return rows


def build(rows: list[dict], *, seed: int, max_chars: int) -> tuple[dict[str, bytes], dict]:
    if not rows:
        raise ValueError("输入不能是空数据集")
    ids = []
    for line, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"第 {line} 条必须是 JSON object")
        if any(not isinstance(row.get(field), str) for field in REQUIRED):
            raise ValueError(f"第 {line} 条缺少字符串字段：{REQUIRED}")
        if any(not row[field].strip() for field in ("id", "family_id", "source", "license")):
            raise ValueError(f"第 {line} 条的身份、题族、来源与许可记录不能为空")
        if "expected_answer" in row and not isinstance(row["expected_answer"], str):
            raise ValueError(f"第 {line} 条 expected_answer 必须是字符串")
        ids.append(row["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("样本 id 重复；先建立唯一原始身份，不能静默覆盖")

    parent = {sample_id: sample_id for sample_id in ids}

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left, right):
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    family_first, prompt_first = {}, {}
    gold_by_prompt = defaultdict(set)
    for row in rows:
        prompt = normalized(row["prompt"])
        for value, first in ((row["family_id"], family_first), (prompt, prompt_first)):
            # Empty prompts will be filtered; do not join unrelated empty examples.
            if not value: continue
            if value in first: union(row["id"], first[value])
            else: first[value] = row["id"]
        if "expected_answer" in row:
            gold_by_prompt[prompt].add(normalized(row["expected_answer"]))
    conflicting = {prompt for prompt, gold in gold_by_prompt.items() if len(gold) > 1}
    members = defaultdict(list)
    for sample_id in ids: members[find(sample_id)].append(sample_id)
    groups = {root: digest(json_bytes(sorted(component)))[:20] for root, component in members.items()}

    kept, audit, seen_pairs = [], [], {}
    source_input = Counter(row["source"] for row in rows)
    for input_row, row in sorted(enumerate(rows, 1), key=lambda pair: pair[1]["id"]):
        prompt, completion = normalized(row["prompt"]), normalized(row["completion"])
        expected = normalized(row["expected_answer"]) if "expected_answer" in row else None
        group = groups[find(row["id"])]
        record = {"id": row["id"], "input_row": input_row, "raw_record_sha256": digest(json_bytes(row)),
                  "source": row["source"], "group_id": group}
        reason = None
        if not prompt or not completion: reason = "empty_text"
        elif len(prompt) + len(completion) > max_chars: reason = "too_long"
        elif prompt in conflicting: reason = "conflicting_gold"
        elif expected is not None and completion != expected: reason = "gold_mismatch"
        pair = (prompt, completion)
        if reason is None and pair in seen_pairs:
            reason = "exact_duplicate"
            record["canonical_id"] = seen_pairs[pair]
        if reason is not None:
            audit.append({**record, "decision": "drop", "reason": reason})
            continue
        seen_pairs[pair] = row["id"]
        bucket = int(digest(f"{seed}:{group}".encode("utf-8"))[:16], 16) % 100
        split = "train" if bucket < 70 else "validation" if bucket < 85 else "test"
        clean = {field: row[field] for field in REQUIRED}
        clean.update({"prompt": prompt, "completion": completion, "group_id": group, "split": split,
                      "character_count": len(prompt) + len(completion), "token_count": None,
                      "tokenizer_id": None, "raw_record_sha256": record["raw_record_sha256"]})
        if expected is not None: clean["expected_answer"] = expected
        kept.append(clean)
        audit.append({**record, "decision": "keep", "reason": "accepted", "split": split})
    if not kept:
        raise ValueError("没有通过过滤的样本；检查质量规则及输入，不能构建空训练资产")
    group_splits = defaultdict(set)
    for row in kept: group_splits[row["group_id"]].add(row["split"])
    if any(len(splits) > 1 for splits in group_splits.values()):
        raise AssertionError("题族/精确重复连通组跨 split")
    data_bytes = b"".join(json_bytes(row) for row in kept)
    audit_bytes = b"".join(json_bytes(row) for row in audit)
    counts = Counter(row["reason"] for row in audit if row["decision"] == "drop")
    source_kept = Counter(row["source"] for row in kept)
    split_counts = Counter(row["split"] for row in kept)
    configuration = {"pipeline_version": "data-asset-lab-v1", "seed": seed, "max_characters": max_chars,
                     "normalization": "Unicode NFC and strip outer whitespace; preserve case/internal whitespace",
                     "split": "group hash buckets 70/15/15; proportions approximate on small data",
                     "grouping": "explicit family_id OR identical normalized prompt; transitive union",
                     "duplicate_key": ["normalized prompt", "normalized completion"],
                     "tokenizer": None}
    manifest = {"configuration": configuration, "canonical_input_sha256": digest(b"".join(json_bytes(row) for row in rows)),
                "pipeline_script_sha256": digest(Path(__file__).read_bytes()),
                "output_sha256": {"data.jsonl": digest(data_bytes), "audit.jsonl": digest(audit_bytes)},
                "counts": {"input": len(rows), "kept": len(kept), "dropped": len(rows)-len(kept),
                           "reasons": dict(sorted(counts.items())), "splits": dict(sorted(split_counts.items()))},
                "sources": {source: {"input": count, "kept": source_kept[source],
                                     "retention": source_kept[source]/count} for source, count in sorted(source_input.items())},
                "scope": "teaching provenance audit; given gold labels only; explicit families and exact duplicates; no semantic near-duplicate detector, model or tokenizer"}
    files = {"raw.jsonl": b"".join(json_bytes(row) for row in rows), "data.jsonl": data_bytes,
             "audit.jsonl": audit_bytes, "manifest.json": json_bytes(manifest)}
    summary = {"experiment": "teaching_data_asset", "input_records": len(rows), "kept_records": len(kept),
               "filtered_records": len(rows)-len(kept), "filter_reasons": dict(sorted(counts.items())),
               "splits": dict(sorted(split_counts.items())), "group_count": len(group_splits),
               "checks": {"group_disjoint": True, "unique_ids": True, "ledger_complete": len(audit)==len(rows)},
               "outputs": sorted(files), "scope": manifest["scope"]}
    return files, summary


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--input", type=Path, help="JSONL 输入；省略时使用含故意错误的 50 条原创教学记录")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-chars", type=int, default=512)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.max_chars < 1: parser.error("--max-chars 必须是正整数")
    output = args.output_dir.expanduser().resolve() if args.output_dir else ROOT / "study_runs" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_data_asset")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"输出目录非空，已保留现有文件：{output}")
    try:
        if args.input:
            lines = args.input.expanduser().resolve().read_text(encoding="utf-8-sig").splitlines()
            def unique_fields(pairs):
                result = {}
                for key, value in pairs:
                    if key in result: raise ValueError(f"JSON 字段重复：{key}")
                    result[key] = value
                return result
            rows = [json.loads(line, object_pairs_hook=unique_fields,
                               parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
                    for line in lines if line.strip()]
        else: rows = demo_records()
        files, summary = build(rows, seed=args.seed, max_chars=args.max_chars)
        repeated, _ = build(rows, seed=args.seed, max_chars=args.max_chars)
        if files != repeated: raise AssertionError("相同输入和配置不能逐字重建资产")
        summary["checks"].update({"rebuild_identical": True, "status": "passed"})
        if not args.input and args.max_chars == 512:
            expected = {"empty_text":1, "too_long":1, "gold_mismatch":1, "conflicting_gold":2, "exact_duplicate":1}
            if summary["filter_reasons"] != expected: raise AssertionError("植入的错误未按预期被识别")
        output = reserve_output_dir(output, "data-asset")
        for name, content in files.items(): (output/name).write_bytes(content)
        (output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    except (ValueError, OSError) as error:
        parser.exit(2, f"数据资产未构建：{error}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
