"""Small, explicit persistence primitives used by notebook and command line."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time

BASE_ID = "Qwen/Qwen3.5-0.8B"
BASE_REV = "2fc06364715b967f1860aea9cf38778875588b17"
PATCH_REV = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
ARMS = {
    "A": {"variant": "original", "thinking": True, "repetition_penalty": 1.0},
    "C": {"variant": "original", "thinking": False, "repetition_penalty": 1.0},
    "C_pen": {"variant": "original", "thinking": True, "repetition_penalty": 1.08},
    "B": {"variant": "patched", "thinking": True, "repetition_penalty": 1.0},
    "D": {"variant": "patched", "thinking": False, "repetition_penalty": 1.0},
    "R1": {"variant": "sign", "thinking": True, "repetition_penalty": 1.0},
    "S1": {"variant": "orthogonal", "thinking": True, "repetition_penalty": 1.0},
}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Incomplete/corrupt JSONL line {number} in {path}; preserve and repair explicitly") from exc
    return rows

def append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())

def completed_records(path):
    indexed = {}
    for row in read_jsonl(path):
        key = row["key"]
        if key in indexed:
            raise ValueError(f"Duplicate prediction key: {key}")
        indexed[key] = row
    return indexed

def request_seed(example_id, language, seed=17):
    # Same per-example seed across arms; independent of batch and resume order.
    return int(digest([example_id, language, seed])[:8], 16) % (2**31 - 1)

def record_key(config_hash, dataset_hash, arm, row, seed):
    return digest([config_hash, dataset_hash, arm, row["id"], row["language"], seed])

def checkpoint(source, destination):
    """Copy a closed JSONL/config file via temp file; verify before replacement."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    if file_hash(source) != file_hash(temporary):
        raise IOError(f"Backup checksum mismatch: {source}")
    temporary.replace(destination)

def snapshot_tracking(run_dir, backup_dir):
    run_dir, backup_dir = Path(run_dir), Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    database = run_dir / "tracking" / "mlflow.db"
    if database.exists():
        target = backup_dir / "mlflow.db"
        with sqlite3.connect(database) as source, sqlite3.connect(target) as dest:
            source.backup(dest)
    artifacts = run_dir / "tracking" / "artifacts"
    if artifacts.exists():
        shutil.copytree(artifacts, backup_dir / "artifacts", dirs_exist_ok=True)

def provenance(root):
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unavailable (archive upload)"
    return {"git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
            "python": sys.version, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
