"""Shared, standard-library output protection for standalone course experiments."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
CLAIM_FILE = ".maedaily-output.json"


def reserve_output_dir(requested: Path | None, lesson: str, *, root: Path | None = None) -> Path:
    """Claim a new/empty directory once; never reuse another run's artifacts.

    The exclusive marker prevents two processes from accepting the same empty
    directory. It is a permanent receipt, not a live-process lock: failed runs
    remain reserved so their evidence cannot be overwritten.
    """
    if requested is None:
        base = (root or ROOT) / "study_runs"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        requested = base / f"{stamp}_{lesson}_{uuid.uuid4().hex[:8]}"
    output = requested.expanduser().resolve()
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise ValueError(f"输出目录必须是新目录或空目录，已保留现有内容：{output}")
    else:
        output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema_version": 1, "lesson": lesson,
               "reserved_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        with (output / CLAIM_FILE).open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except FileExistsError as error:
        raise ValueError(f"输出目录已被另一运行占用，已保留现有内容：{output}") from error
    return output
