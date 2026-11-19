"""One process per weight variant. Run through experiment.run, not notebook imports."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from .common import (ARMS, load_json, save_json, read_jsonl, append_jsonl, completed_records,
                     request_seed, record_key, checkpoint, snapshot_tracking, digest)
from .data import selected, render_user
from .experiment import config_hash, phase_dir, validate_config
from .scoring import score, repetition, sections, language_label

def hardware():
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("No CUDA GPU: select a GPU runtime in Colab")
    return {"gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
            "gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
            "native_bf16": torch.cuda.get_device_capability(0)[0] >= 8,
            "versions": {p: importlib.metadata.version(p) for p in ["torch","transformers","numpy"]}}

def main(cfg, variant):
    import torch
    from transformers import AutoTokenizer
    validate_config(cfg)
    folder = phase_dir(cfg)
    hw = hardware()
    if cfg["dtype"] == "bfloat16" and not hw["native_bf16"]:
        raise RuntimeError("Native BF16 unavailable. Prefer L4/A100; an FP16 pilot needs a new declared config.")
    if cfg["engine"] == "vllm": hw["versions"]["vllm"] = importlib.metadata.version("vllm")
    signature = digest(hw)
    hardware_file = folder / "hardware.json"
    if hardware_file.exists() and load_json(hardware_file)["signature"] != signature:
        raise RuntimeError("Hardware/software changed on resume; do not mix outputs. Start a declared new run.")
    save_json(hardware_file, {**hw, "signature": signature})
    weights = load_json(Path(cfg["models_dir"])/"weights_manifest.json")
    if cfg['dtype'] == 'float16':
        from .weights import upstream, diagnostics, load_checkpoint_tensors
        S = upstream(cfg['root'])
        targets = list(weights['variants']['patched'])
        base_raw = load_checkpoint_tensors(weights['paths']['original'], targets, S)
        variant_raw = load_checkpoint_tensors(weights['paths'][variant], targets, S)
        patch_raw = load_checkpoint_tensors(weights['paths']['patched'], targets, S)
        import numpy as np
        precision = {}
        for name in targets:
            original = S.bf16_to_f32(base_raw[name][1]).astype(np.float16).astype(np.float64)
            changed = S.bf16_to_f32(variant_raw[name][1]).astype(np.float16).astype(np.float64)
            patched = S.bf16_to_f32(patch_raw[name][1]).astype(np.float16).astype(np.float64)
            if not np.isfinite(changed).all(): raise ValueError('Nonfinite FP16 weights')
            columns = np.flatnonzero(np.any(patched != original, axis=0))
            precision[name] = diagnostics((changed-original)[:,columns], (patched-original)[:,columns])
        save_json(folder/f'postcast-{variant}.json',precision)
    tokenizer = AutoTokenizer.from_pretrained(weights["paths"]["original"], trust_remote_code=False)
    arms = [a for a in cfg["arms"] if ARMS[a]["variant"] == variant]
    rows = selected(read_jsonl(Path(cfg["root"])/"data/tasks.jsonl"), cfg["phase"], cfg["pairs"], cfg["dev_pairs"])
    destination = folder / "predictions.jsonl"
    done = completed_records(destination)
    fingerprint = config_hash(cfg)
    pending = [(a, r) for a in arms for r in rows if record_key(fingerprint, cfg["dataset_hash"], a, r, cfg["seed"]) not in done]
    if not pending:
        print("SELLMA: all requested keys already complete; skipped without loading weights")
        return
    rendered = {}
    for arm, row in pending:
        prompt = tokenizer.apply_chat_template([{"role":"user", "content":render_user(row)}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=ARMS[arm]["thinking"])
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids)+cfg["cap"] > cfg["max_model_len"]: raise ValueError("Prompt + cap exceeds context")
        if ARMS[arm]["thinking"] and "<think>" not in prompt: raise ValueError("Thinking-on template lacks marker")
        if not ARMS[arm]["thinking"] and "</think>" not in prompt: raise ValueError("Thinking-off template did not pre-close reasoning")
        rendered[(arm,row["id"],row["language"])] = (prompt, prompt_ids)
    save_json(folder/f"prompts-{variant}.json", [{"arm":a,"id":i,"language":l,"prompt":p,"token_ids":t}
               for (a,i,l),(p,t) in rendered.items()])
    start = time.monotonic()
    if cfg["engine"] == "vllm":
        from vllm import LLM, SamplingParams
        model = LLM(model=weights["paths"][variant], tokenizer=weights["paths"]["original"],
                    dtype=cfg["dtype"], trust_remote_code=False, seed=cfg["seed"],
                    max_model_len=cfg["max_model_len"], max_num_seqs=cfg["batch_size"],
                    gpu_memory_utilization=cfg["gpu_memory_utilization"],
                    enforce_eager=cfg["enforce_eager"], enable_prefix_caching=False,
                    kv_cache_dtype=cfg["kv_cache_dtype"], mamba_cache_dtype=cfg["mamba_cache_dtype"],
                    mamba_ssm_cache_dtype=cfg["mamba_ssm_cache_dtype"], language_model_only=True,
                    generation_config="vllm")
        try:
            resolved = str(model.llm_engine.vllm_config)
        except Exception as exc:  # attribute layout differs between vLLM releases
            resolved = f"vLLM {importlib.metadata.version('vllm')} (config not introspectable: {type(exc).__name__})"
    else:
        from transformers import Qwen3_5ForConditionalGeneration, LogitsProcessor, LogitsProcessorList
        model = Qwen3_5ForConditionalGeneration.from_pretrained(weights["paths"][variant],
                dtype=getattr(torch,cfg["dtype"]), device_map="cuda", trust_remote_code=False).eval()
        class Presence(LogitsProcessor):
            def __init__(self, prompt_length, penalty): self.prompt_length, self.penalty = prompt_length, penalty
            def __call__(self, ids, scores):
                for j in range(ids.shape[0]):
                    scores[j, torch.unique(ids[j,self.prompt_length:])] -= self.penalty
                return scores
        resolved = "Transformers single-request generation; repetition applies to prompt+output, presence to output only"
    load_seconds = time.monotonic()-start
    runtime = {"load_seconds":load_seconds, "hardware":hw, "resolved_engine":resolved,
               "dtype":cfg["dtype"], "precision_modified":cfg["dtype"] != "bfloat16"}
    save_json(folder/f"runtime-{variant}.json", runtime)
    print(f"SELLMA: {variant} loaded in {load_seconds:.1f}s on {hw['gpu']}", flush=True)
    lock = subprocess.run([sys.executable,"-m","pip","freeze"],capture_output=True,text=True)
    (folder/"environment.lock.txt").write_text(lock.stdout,encoding="utf-8")
    for arm in arms:
        arm_pending = [r for a,r in pending if a==arm]
        batch_size = cfg["batch_size"] if cfg["engine"]=="vllm" else 1
        for offset in range(0,len(arm_pending),batch_size):
            if time.time() >= load_json(folder/"deadline.json")["unix_deadline"]:
                print("SELLMA: deadline reached; preserving partial condition",flush=True)
                return
            batch = arm_pending[offset:offset+batch_size]
            prompts = [rendered[(arm,r["id"],r["language"])][0] for r in batch]
            seeds = [request_seed(r["id"],r["language"],cfg["seed"]) for r in batch]
            start = time.monotonic()
            if cfg["engine"]=="vllm":
                params = [SamplingParams(temperature=cfg["temperature"],top_p=cfg["top_p"],top_k=cfg["top_k"],
                         presence_penalty=cfg["presence_penalty"],frequency_penalty=cfg["frequency_penalty"],
                         repetition_penalty=ARMS[arm]["repetition_penalty"],seed=s,max_tokens=cfg["cap"],
                         skip_special_tokens=False,logprobs=1 if cfg["phase"]=="smoke" else None) for s in seeds]
                outputs = model.generate(prompts,params,use_tqdm=False)
                generated = []
                for out in outputs:
                    completion = out.outputs[0]
                    if completion.logprobs:
                        assert all(math.isfinite(lp[token].logprob) for token,lp in zip(completion.token_ids,completion.logprobs))
                    generated.append((list(completion.token_ids),completion.finish_reason,str(completion.stop_reason)))
            else:
                torch.manual_seed(seeds[0]); torch.cuda.manual_seed_all(seeds[0])
                inputs = tokenizer(prompts[0],return_tensors="pt",add_special_tokens=False).to("cuda")
                with torch.inference_mode():
                    if cfg["phase"]=="smoke":
                        logits = model(**inputs).logits[:,-1,:]
                        assert torch.isfinite(logits).all(), "Nonfinite forward logits"
                    ids = model.generate(**inputs,max_new_tokens=cfg["cap"],do_sample=True,
                            temperature=cfg["temperature"],top_p=cfg["top_p"],top_k=cfg["top_k"],
                            repetition_penalty=ARMS[arm]["repetition_penalty"],
                            logits_processor=LogitsProcessorList([Presence(inputs.input_ids.shape[1],cfg["presence_penalty"])]),
                            forced_eos_token_id=None)
                tokens = ids[0,inputs.input_ids.shape[1]:].tolist()
                eos = model.generation_config.eos_token_id
                eos = [eos] if isinstance(eos,int) else eos
                finished = bool(tokens and tokens[-1] in eos)
                generated = [(tokens,"stop" if finished else "length","eos" if finished else "cap")]
            batch_seconds = time.monotonic()-start
            call_id = digest([arm, offset, [r["id"]+r["language"] for r in batch]])[:12]
            completed = []
            for row,(token_ids,finish,stop),seed in zip(batch,generated,seeds):
                text = tokenizer.decode(token_ids,skip_special_tokens=False)
                thinking = ARMS[arm]["thinking"]
                endpoint = score(text,thinking,row["gold"],row["answer_type"],finish,len(token_ids),cfg["cap"])
                budget_scores = {}
                for budget in [512,1024,2048,4096,8192,16384]:
                    if budget > cfg["cap"]: continue
                    prefix = tokenizer.decode(token_ids[:budget],skip_special_tokens=False)
                    budget_scores[str(budget)] = score(prefix,thinking,row["gold"],row["answer_type"],finish,len(token_ids),budget)
                reasoning,final,_ = sections(text,thinking)
                # Generated reasoning tokens up to the close marker; exclude special tokens.
                close_id = tokenizer.convert_tokens_to_ids("</think>")
                reason_ids = token_ids[:token_ids.index(close_id)] if close_id in token_ids else token_ids if thinking else []
                reason_ids = [t for t in reason_ids if t not in tokenizer.all_special_ids]
                prefix = tokenizer.decode(reason_ids[:250],skip_special_tokens=True)
                views = {"prefix250":prefix,"reasoning":reasoning,"final":final} if row["language"]=="sv" else {}
                completed.append({**row,"key":record_key(fingerprint,cfg["dataset_hash"],arm,row,cfg["seed"]),
                    "condition":arm,"config_hash":fingerprint,"dataset_hash":cfg["dataset_hash"],
                    "request_seed":seed,"decoding_seed":cfg["seed"],"thinking":thinking,
                    "prompt_rendered":rendered[(arm,row["id"],row["language"])][0],
                    "prompt_token_ids":rendered[(arm,row["id"],row["language"])][1],
                    "token_ids":token_ids,"output":text,"tokens":len(token_ids),"finish_reason":finish,
                    "stop_reason":stop,"cap_hit":finish=="length", "batch_seconds":batch_seconds,
                    "batch_count":len(batch),"call_id":call_id,"endpoint":endpoint,"budgets":budget_scores,
                    **repetition(token_ids,tokenizer.all_special_ids), "language_views":views,
                    "language_labels":{k:language_label(v) for k,v in views.items()}})
            append_jsonl(destination,completed)
            if cfg["backup_dir"]:
                backup = Path(cfg["backup_dir"])/folder.name
                checkpoint(destination,backup/"predictions.jsonl")
                checkpoint(folder/"config.json",backup/"config.json")
            print(f"SELLMA: {arm} saved {offset+len(batch)}/{len(arm_pending)} requests; batch {batch_seconds:.1f}s",flush=True)
        log_tracking(cfg,arm)
        if cfg["backup_dir"]: snapshot_tracking(folder,Path(cfg["backup_dir"])/folder.name/"tracking")
    snapshot_tracking(folder,folder/"tracking_snapshot")

def log_tracking(cfg,arm):
    """One MLflow run per condition/phase, reused on resume; JSONL remains authoritative."""
    folder = phase_dir(cfg)
    try:
        import mlflow
        tracking = folder/"tracking"
        tracking.mkdir(exist_ok=True)
        mlflow.set_tracking_uri("sqlite:///"+str((tracking/"mlflow.db").resolve()).replace("\\","/"))
        experiment = mlflow.get_experiment_by_name("sellma-weight-space")
        exp_id = experiment.experiment_id if experiment else mlflow.create_experiment("sellma-weight-space",artifact_location=(tracking/"artifacts").resolve().as_uri())
        index_path = tracking/"run_ids.json"
        index = load_json(index_path) if index_path.exists() else {}
        kwargs = {"run_id":index[arm]} if arm in index else {"experiment_id":exp_id,"run_name":arm}
        with mlflow.start_run(**kwargs) as active:
            index[arm]=active.info.run_id; save_json(index_path,index)
            mlflow.log_params({"condition":arm,"phase":cfg["phase"],"cap":cfg["cap"],"dtype":cfg["dtype"],
                "engine":cfg["engine"],"seed":cfg["seed"],"config_hash":config_hash(cfg),"dataset_hash":cfg["dataset_hash"]})
            rows=[r for r in read_jsonl(folder/"predictions.jsonl") if r["condition"]==arm]
            if rows:
                mlflow.log_metrics({"requests":len(rows),"correct_completed":sum(r["endpoint"]["correct_completed"] for r in rows)/len(rows),
                    "cap_hit":sum(r["cap_hit"] for r in rows)/len(rows),"mean_tokens":sum(r["tokens"] for r in rows)/len(rows)})
            mlflow.log_artifact(str(folder/"config.json"))
        print(f"SELLMA: MLflow recorded {arm}",flush=True)
    except Exception as exc:
        append_jsonl(folder/"tracking_errors.jsonl",[{"condition":arm,"error":str(exc)}])
        print(f"SELLMA: MLflow unavailable ({type(exc).__name__}); JSONL evidence retained",flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True); parser.add_argument("--variant",required=True)
    args=parser.parse_args()
    main(load_json(args.config),args.variant)
