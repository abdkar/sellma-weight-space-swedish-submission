"""Configuration and subprocess orchestration. All test decisions precede outputs."""
from pathlib import Path
import json
import subprocess
import sys
import time
from .common import ARMS, BASE_REV, PATCH_REV, digest, file_hash, load_json, save_json, read_jsonl, provenance
from .data import selected

def code_hash(root):
    root = Path(root)
    files = sorted((root / "sellma_wsp").glob("*.py")) + sorted((root / "vendor/upstream").glob("*.py"))
    return digest({str(p.relative_to(root)): file_hash(p) for p in files})

def config(root, run_dir, models_dir, phase="smoke", cap=128, pairs=40,
           arms=None, engine="vllm", batch_size=128, dtype="bfloat16", backup_dir=None,
           enforce_eager=False):
    root = Path(root).resolve()
    rows = read_jsonl(root / "data/tasks.jsonl")
    return {"root": str(root), "run_dir": str(Path(run_dir).resolve()),
            "models_dir": str(Path(models_dir).resolve()), "phase": phase, "cap": cap,
            "pairs": pairs, "dev_pairs": 4, "arms": arms or list(ARMS), "engine": engine,
            "batch_size": batch_size, "dtype": dtype, "seed": 17,
            "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "presence_penalty": 1.5, "frequency_penalty": 0.0,
            "max_model_len": cap + 1024, "gpu_memory_utilization": 0.80,
            "enforce_eager": enforce_eager, "kv_cache_dtype": "auto",
            "mamba_cache_dtype": "auto", "mamba_ssm_cache_dtype": "float32",
            "base_revision": BASE_REV, "patch_revision": PATCH_REV,
            "dataset_hash": digest(rows), "code_hash": code_hash(root),
            "backup_dir": str(backup_dir) if backup_dir else None,
            "data_reviewed": False, "scope": "core", "time_limit_minutes": 85,
            "decision": "Not frozen", "provenance": provenance(root)}

def config_hash(cfg):
    # Paths and wall-clock metadata do not change scientific identity.
    ignored = {"root", "run_dir", "models_dir", "backup_dir", "provenance", "decision", "data_reviewed"}
    return digest({k:v for k,v in cfg.items() if k not in ignored})

def phase_dir(cfg):
    return Path(cfg["run_dir"]) / f"{cfg['phase']}-{config_hash(cfg)[:12]}"

def validate_config(cfg):
    if cfg["engine"] not in {"vllm", "transformers"}: raise ValueError("Unsupported engine")
    if cfg["dtype"] not in {"bfloat16", "float16"}: raise ValueError("No quantization in the primary study")
    if cfg["phase"] == "test":
        if cfg["cap"] not in {8192, 16384}: raise ValueError("Primary cap must be 8192 or 16384")
        if cfg["pairs"] not in {16,24,40}: raise ValueError("Use a predeclared 16/24/40 pair scope")
        if cfg["scope"] == "core" and set(cfg["arms"]) != set(ARMS): raise ValueError("Core requires all seven arms")
        if cfg["scope"] == "pilot" and not {"A","B","C","D","C_pen"}.issubset(cfg["arms"]):
            raise ValueError("Reduced pilot retains A/B/C/D/C_pen; record absent random controls")
        if not cfg["data_reviewed"]: raise ValueError("Review input pairs and gold answers before freezing")
    if len(set(cfg["arms"])) != len(cfg["arms"]) or not set(cfg["arms"]).issubset(ARMS): raise ValueError("Invalid arms")
    if cfg["batch_size"] < 1: raise ValueError("Invalid batch size")
    if code_hash(cfg["root"]) != cfg["code_hash"]: raise ValueError("Code changed; create a new configuration")
    if digest(read_jsonl(Path(cfg["root"])/"data/tasks.jsonl")) != cfg["dataset_hash"]: raise ValueError("Dataset changed")

def freeze(cfg, decision, measured_seconds, pilot_config):
    """Requires a same-engine, same-cap pilot with original, patched and random arms."""
    validate_config(cfg)
    if cfg["phase"] != "test": raise ValueError("Freeze only the held-out test")
    for key in ["cap", "engine", "dtype", "batch_size", "code_hash", "dataset_hash"]:
        if cfg[key] != pilot_config[key]: raise ValueError(f"Pilot mismatch: {key}")
    pilot_rows = read_jsonl(phase_dir(pilot_config)/"predictions.jsonl")
    if not {"A","B","R1"}.issubset({r["condition"] for r in pilot_rows}):
        raise ValueError("Time A, B and R1 on development data before freezing")
    expected = selected(read_jsonl(Path(cfg['root'])/'data/tasks.jsonl'), 'pilot', dev_pairs=pilot_config['dev_pairs'])
    required = {(arm,r['id'],r['language']) for arm in pilot_config['arms'] for r in expected}
    observed = {(r['condition'],r['id'],r['language']) for r in pilot_rows}
    if required != observed: raise ValueError('Complete the planned development pilot before using its timing projection')
    if not any(r['endpoint']['completed'] for r in pilot_rows if r['condition'] in {'A','B'}) and cfg['scope'] != 'pilot':
        raise ValueError('Neither A nor B completed a development example; increase cap or label the scope a pilot')
    if measured_seconds <= 0 or measured_seconds > 0.8 * cfg["time_limit_minutes"] * 60:
        raise ValueError("Projected runtime exceeds the 20% reserve; reduce scope or report a pilot")
    cfg = dict(cfg, decision=decision, projected_seconds=measured_seconds,
               pilot_config_hash=config_hash(pilot_config))
    folder = phase_dir(cfg)
    if (folder / "predictions.jsonl").exists(): raise ValueError("Cannot refreeze after test generation")
    save_json(folder / "frozen.json", {"config": cfg, "hash": config_hash(cfg), "frozen_utc": provenance(cfg["root"])["timestamp_utc"]})
    save_json(Path(cfg["run_dir"])/"test_config.json", cfg)
    return cfg

def run(cfg, groups=None):
    validate_config(cfg)
    folder = phase_dir(cfg)
    if cfg["phase"] == "test":
        frozen = load_json(folder/"frozen.json")
        if frozen["hash"] != config_hash(cfg): raise ValueError("Frozen configuration mismatch")
    folder.mkdir(parents=True, exist_ok=True)
    config_file = folder / "config.json"
    save_json(config_file, cfg)
    requested = groups or ["original","patched","sign","orthogonal"]
    start = time.monotonic()
    # Persist a deadline across repeated notebook group cells/resumes in this runtime.
    deadline_path = folder/"deadline.json"
    if not deadline_path.exists():
        save_json(deadline_path, {"unix_deadline": time.time()+cfg["time_limit_minutes"]*60})
    for group in requested:
        arms = [a for a in cfg["arms"] if ARMS[a]["variant"] == group]
        if not arms: continue
        if time.time() >= load_json(deadline_path)["unix_deadline"]:
            print("Time limit reached. Preserve partial results; no new worker launched.")
            break
        command = [sys.executable, "-m", "sellma_wsp.worker", "--config", str(config_file), "--variant", group]
        print(f"Running {group}: {', '.join(arms)}", flush=True)
        log = folder / f"{group}.log"
        with log.open("a", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=cfg["root"], stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in process.stdout:
                stream.write(line); stream.flush()
                # Bound notebook output; full engine logs remain in the artifact.
                if "SELLMA:" in line: print(line.strip(), flush=True)
            status = process.wait()
        if status:
            print(log.read_text(encoding="utf-8")[-7000:])
            raise RuntimeError(f"Worker failed ({status}). Full log: {log}; no engine was silently switched.")
    return {"directory": str(folder), "elapsed_seconds": time.monotonic()-start,
            "completed": len(read_jsonl(folder/"predictions.jsonl"))}

def estimate(cfg, test_batch_size=None, safety=1.5):
    """Conservative projection from measured pilot calls.

    vLLM decodes a whole call in parallel, so a call lasts about as long as its longest
    output. We therefore project per *call*, not per request: each arm needs
    ceil(prompts / batch_size) calls, each costing the slowest observed pilot call, times a
    safety factor for the larger batch. Thinking-off arms are charged the same worst case.
    """
    import math
    folder = phase_dir(cfg)
    rows = read_jsonl(folder/"predictions.jsonl")
    calls = {}
    for row in rows:
        calls.setdefault(row["condition"], set()).add((row.get("call_id"), row["batch_seconds"]))
    if not {"A","B","R1"}.issubset(calls): raise ValueError("Pilot needs A, B, R1")
    worst = max(seconds for v in calls.values() for _, seconds in v)
    load_times = [load_json(p)["load_seconds"] for p in folder.glob("runtime-*.json")]
    loading = max(load_times, default=0)
    batch = test_batch_size or cfg["batch_size"]
    out = []
    for arms in [7,5]:
        for n in [40,24,16]:
            per_arm_calls = math.ceil((2*n+16)/batch)
            seconds = arms*per_arm_calls*worst*safety + (4 if arms==7 else 2)*loading
            out.append({"pairs": n, "conditions": arms, "cap": cfg["cap"], "calls_per_arm": per_arm_calls,
                        "worst_pilot_call_seconds": round(worst,1), "projected_seconds": round(seconds),
                        "projected_minutes": round(seconds/60,1),
                        "within_85min_with_reserve": seconds <= 85*60*0.8})
    return out
