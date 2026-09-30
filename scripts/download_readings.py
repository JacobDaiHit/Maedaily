"""Download the public originals listed in references/readings.json (stdlib only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "references" / "readings.json"
RECEIPTS = ROOT / "references" / "downloads.json"


def check_pdf(path: Path) -> tuple[int, str]:
    data = path.read_bytes()
    if len(data) < 1024 or not data.startswith(b"%PDF-") or b"%%EOF" not in data[-4096:]:
        raise ValueError("Not a complete-looking PDF; refusing HTML or truncated response")
    return len(data), hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", action="append", dest="ids", help="Only these manifest IDs; repeatable")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    items = json.loads(MANIFEST.read_text(encoding="utf-8"))
    known = {item["id"] for item in items}
    if args.ids and set(args.ids) - known:
        parser.error("Unknown IDs: " + ", ".join(sorted(set(args.ids) - known)))
    receipts = json.loads(RECEIPTS.read_text(encoding="utf-8")) if RECEIPTS.exists() else {}
    failures = 0
    for item in items:
        ident = item["id"]
        if args.ids and ident not in args.ids:
            continue
        target = (ROOT / item["local_pdf"]).resolve()
        if not target.is_relative_to(ROOT) or target.suffix != ".pdf":
            raise ValueError("Manifest destination must be a PDF inside this repository")
        now = datetime.now(timezone.utc).isoformat()
        try:
            if target.exists():
                size, digest = check_pdf(target)
                old = receipts.get(ident, {})
                if old.get("sha256") and old["sha256"] != digest:
                    raise ValueError("Existing file differs from its recorded SHA-256")
                if old.get("url") and old["url"] != item["pdf_url"]:
                    raise ValueError("Manifest URL changed; preserve the old version before replacing it")
                action = "verified"
            else:
                if args.verify_only:
                    raise FileNotFoundError(str(target))
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".pdf.part")
                request = urllib.request.Request(item["pdf_url"], headers={"User-Agent": "Maedaily-reading-library/1.0"})
                try:
                    with urllib.request.urlopen(request, timeout=45) as response, temporary.open("wb") as output:
                        total = 0
                        while chunk := response.read(256 * 1024):
                            total += len(chunk)
                            if total > 100 * 1024 * 1024:
                                raise ValueError("PDF exceeds the 100 MiB per-file limit")
                            output.write(chunk)
                    size, digest = check_pdf(temporary)
                    temporary.replace(target)
                finally:
                    if temporary.exists():
                        temporary.unlink()
                action = "downloaded"
                time.sleep(1)
            receipts[ident] = {
                **receipts.get(ident, {}),
                "status": "ok", "url": item["pdf_url"], "path": item["local_pdf"],
                "bytes": size, "sha256": digest, "checked_at_utc": now,
                "downloaded_at_utc": receipts.get(ident, {}).get("downloaded_at_utc", now),
            }
            receipts[ident].pop("error", None)
            print(f"{ident}: {action}, {size:,} bytes", flush=True)
        except Exception as error:
            failures += 1
            receipts[ident] = {**receipts.get(ident, {}), "status": "failed", "error": str(error), "checked_at_utc": now}
            print(f"{ident}: FAILED: {error}", flush=True)
        RECEIPTS.parent.mkdir(parents=True, exist_ok=True)
        RECEIPTS.write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
