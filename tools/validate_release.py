"""Validate packaged checksums and paired test sets; no models or APIs used."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate() -> dict:
    checks = json.loads((ROOT / "release_checksums.json").read_text(encoding="utf-8"))
    for row in checks["files"]:
        path = (ROOT / row["path"]).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError("Missing or invalid package path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Checksum mismatch: {row['path']}")
    manifest = json.loads((ROOT / "data/testset_manifest.json").read_text(encoding="utf-8"))
    paired = {}
    for entry in manifest["files"]:
        path = ROOT / entry["file"]
        records = json.loads(path.read_text(encoding="utf-8-sig"))
        if len(records) != 200 or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Test-set checksum or count mismatch")
        for record in records:
            if set(record) != {"instruction", "input", "output"} or not all(isinstance(v, str) and v.strip() for v in record.values()):
                raise ValueError("Invalid or empty test record")
        paired[(entry["original_task_label"], entry["prompting"])] = [(r["input"], r["output"]) for r in records]
    tasks = {task for task, _ in paired}
    if len(tasks) != 6:
        raise ValueError("Expected six tasks")
    for task in tasks:
        if paired[task, "0-shot"] != paired[task, "5-shot"]:
            raise ValueError("Mode variants do not align")
    credential = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}|\bBearer\s+[A-Za-z0-9_.-]{20,}")
    for entry in checks["files"]:
        path = ROOT / entry["path"]
        if path.suffix in {".py", ".json", ".md", ".txt"} and credential.search(path.read_text(encoding="utf-8-sig")):
            raise ValueError(f"Credential-like token found in {entry['path']}")
    return {"checksummed_files": len(checks["files"]), "tasks": 6, "unique_task_instances": 1200, "prompt_variants": 12, "formatted_records": 2400, "external_api_calls": 0}


if __name__ == "__main__":
    print(json.dumps(validate(), ensure_ascii=False, indent=2))
