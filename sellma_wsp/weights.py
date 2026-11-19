"""Canonical BF16 reconstruction and two controls; no inference dependencies."""
from pathlib import Path
import hashlib
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import numpy as np
from .common import BASE_ID, BASE_REV, PATCH_REV, load_json, save_json, file_hash

def upstream(root):
    path = Path(root) / "vendor/upstream/st_numpy.py"
    spec = importlib.util.spec_from_file_location("sellma_st_numpy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def random_delta(delta, kind, seed):
    delta = np.asarray(delta, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if kind == "sign":
        return np.abs(delta) * rng.choice([-1.0, 1.0], size=delta.shape)
    _, _, vh = np.linalg.svd(delta, full_matrices=False)
    v = vh[0]
    g = delta @ v
    if g @ g < 1e-20:
        raise ValueError("Zero control direction")
    for _ in range(10):
        u = rng.standard_normal(delta.shape[0])
        u -= g * (g @ u) / (g @ g)
        norm = np.linalg.norm(u)
        if norm > 1e-12:
            return np.linalg.norm(delta) * np.outer(u / norm, v)
    raise ValueError("Could not construct orthogonal control")

def diagnostics(delta, reference):
    norm, reference_norm = np.linalg.norm(delta), np.linalg.norm(reference)
    singular = np.linalg.svd(delta, compute_uv=False)
    return {"norm": float(norm), "norm_ratio": float(norm / reference_norm),
            "changed_entries": int(np.count_nonzero(delta)),
            "cosine_with_patch": float(np.sum(delta * reference) / (norm * reference_norm)) if norm else None,
            "rank1_energy_fraction": float(singular[0]**2 / np.sum(singular**2)) if norm else None}

def tensor_hashes(path, S):
    header, offset = S.read_header(path)
    hashes = {}
    with open(path, "rb") as f:
        for name, info in header.items():
            if name == "__metadata__": continue
            start, end = info["data_offsets"]
            f.seek(offset + start)
            h, remaining = hashlib.sha256(), end-start
            while remaining:
                chunk = f.read(min(1024*1024, remaining))
                if not chunk: raise IOError("Truncated safetensors")
                h.update(chunk); remaining -= len(chunk)
            hashes[name] = h.hexdigest()
    return hashes

def checkpoint_files(folder):
    """Resolve Hugging Face's indexed shards or a standard single-file checkpoint."""
    folder = Path(folder)
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        weight_map = load_json(index)["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid weight_map in {index}")
        names = sorted(set(weight_map.values()))
    else:
        weight_map = None
        names = ["model.safetensors"]
    for name in names:
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError(f"Unexpected checkpoint filename: {name!r}")
        if not (folder / name).is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {folder / name}")
    return [folder / name for name in names], weight_map


def load_checkpoint_tensors(folder, names, S):
    files, weight_map = checkpoint_files(folder)
    wanted = set(names)
    tensors = {}
    for path in files:
        selected = wanted if weight_map is None else {n for n in wanted if weight_map.get(n) == path.name}
        if selected:
            tensors.update(S.load(path, selected))
    if wanted != tensors.keys():
        raise ValueError(f"Missing checkpoint tensors: {sorted(wanted - tensors.keys())}")
    return tensors


def rewrite_shard(source, destination, replacements, S):
    """Copy a shard and overwrite only the target tensor byte ranges."""
    header, offset = S.read_header(source)
    temporary = destination.with_suffix(".safetensors.tmp")
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("r+b") as stream:
            for name, bits in replacements.items():
                start, end = header[name]["data_offsets"]
                if bits.nbytes != end-start:
                    raise ValueError(f"Replacement size mismatch for {name}")
                stream.seek(offset+start)
                stream.write(bits.tobytes())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def reconstruct(root, raw, edit_dir, S):
    # The upstream CLI accepts one safetensors file. It only needs the edit's inputs.
    with tempfile.TemporaryDirectory(dir=edit_dir.parent) as scratch:
        scratch = Path(scratch)
        target_tensors = scratch / "targets.safetensors"
        S.save(target_tensors, raw)
        output = scratch / "reconstructed"
        subprocess.run([sys.executable, str(root / "vendor/upstream/apply_edit.py"),
                        str(target_tensors), str(edit_dir), str(output)], check=True)
        return {name: np.load(output / (name + ".npy"), allow_pickle=False) for name in raw}


def prepare(root, models_dir):
    from huggingface_hub import snapshot_download
    root, models_dir = Path(root).resolve(), Path(models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    S = upstream(root)
    manifest = load_json(root / "configs/patch_manifest.json")
    source = Path(snapshot_download(BASE_ID, revision=BASE_REV,
                  allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.txt", "LICENSE*"]))
    source_files, weight_map = checkpoint_files(source)
    print(f"Checkpoint files: {', '.join(p.name for p in source_files)}", flush=True)
    shard_hashes = {path.name: tensor_hashes(path, S) for path in source_files}
    shard_for_tensor = {name: shard for shard, hashes in shard_hashes.items() for name in hashes}
    if len(shard_for_tensor) != sum(map(len, shard_hashes.values())):
        raise ValueError("Duplicate tensor names in checkpoint shards")
    if weight_map is not None and weight_map != shard_for_tensor:
        raise ValueError("Checkpoint index does not match the tensors in the downloaded shards")
    base_hashes = {name: h for hashes in shard_hashes.values() for name, h in hashes.items()}
    edit_dir = models_dir / "patch"
    edit_dir.mkdir(exist_ok=True)
    shutil.copy2(root / "configs/patch_manifest.json", edit_dir / "manifest.json")
    npz = edit_dir / "edit.npz"
    if not npz.exists():
        url = f"https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/{PATCH_REV}/exp8_math/runs/p50c/edit/edit.npz"
        temporary = npz.with_suffix(".npz.tmp")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(npz)
    targets = [x["name"] for x in manifest["tensors"]]
    raw = load_checkpoint_tensors(source, targets, S)
    reconstructed = reconstruct(root, raw, edit_dir, S)
    replacements = {"patched": {}, "sign": {}, "orthogonal": {}}
    report = {"base_id": BASE_ID, "base_revision": BASE_REV, "patch_revision": PATCH_REV,
              "patch_file_sha256": file_hash(npz), "checkpoint_files": [p.name for p in source_files],
              "variants": {}, "paths": {"original": str(source)}}
    for i, item in enumerate(manifest["tensors"]):
        name = item["name"]
        if raw[name][0] != "BF16" or base_hashes[name] != item["base_sha256"]:
            raise ValueError("Canonical base hash/dtype mismatch")
        base = S.bf16_to_f32(raw[name][1])
        patched = reconstructed[name]
        bits = S.bf16_bits(patched)
        if hashlib.sha256(bits.tobytes()).hexdigest() != item["edited_sha256"]:
            raise ValueError("Edited hash mismatch")
        # The source file is never modified. Zero edit and restoring the saved bits are exact.
        assert np.array_equal(S.bf16_bits(base + np.zeros_like(base)), raw[name][1])
        restored = bits.copy(); restored[:] = raw[name][1]
        assert np.array_equal(restored, raw[name][1])
        delta = patched.astype(np.float64) - base.astype(np.float64)
        columns = np.flatnonzero(np.any(delta != 0, axis=0))
        assert len(columns) == 64 and np.count_nonzero(delta) == item["changed_entries"]
        replacements["patched"][name] = bits
        for kind, seed in [("sign", 101), ("orthogonal", 201)]:
            proposal = random_delta(delta[:, columns], kind, seed + i)
            candidate = base.copy()
            candidate[:, columns] = (base[:, columns].astype(np.float64) + proposal).astype(np.float32)
            replacements[kind][name] = S.bf16_bits(candidate)
    for variant, tensors in replacements.items():
        folder = models_dir / variant
        folder.mkdir(exist_ok=True)
        unexpected = {p.name for p in folder.glob("*.safetensors")} - set(shard_hashes)
        if unexpected:
            raise ValueError(f"Unexpected weight files in {folder}: {sorted(unexpected)}. Use a fresh models directory.")
        if weight_map is None and (folder / "model.safetensors.index.json").exists():
            raise ValueError(f"Unexpected weight index in {folder}. Use a fresh models directory.")
        for file in source.iterdir():
            if file.is_file() and file.suffix != ".safetensors": shutil.copy2(file, folder / file.name)
        hashes = {}
        for path in source_files:
            shard_replacements = {name: bits for name, bits in tensors.items() if shard_for_tensor[name] == path.name}
            destination = folder / path.name
            rewrite_shard(path, destination, shard_replacements, S)
            hashes.update(tensor_hashes(destination, S))
        if any(hashes.get(name) != h for name, h in base_hashes.items() if name not in tensors):
            raise ValueError(f"Unexpected tensor change in {variant}")
        info = {}
        for name in targets:
            base = S.bf16_to_f32(raw[name][1]).astype(np.float64)
            delta = S.bf16_to_f32(tensors[name]).astype(np.float64)-base
            reference = S.bf16_to_f32(replacements["patched"][name]).astype(np.float64)-base
            columns = np.flatnonzero(np.any(reference != 0, axis=0))
            info[name] = {**diagnostics(delta[:, columns], reference[:, columns]), "sha256": hashes[name]}
            if variant == "patched":
                expected = next(x["edited_sha256"] for x in manifest["tensors"] if x["name"] == name)
                assert hashes[name] == expected
        report["variants"][variant] = info
        report["paths"][variant] = str(folder)
        print(f"Verified {variant}: only the two permitted tensors differ")
    report["base_tensor_hashes"] = base_hashes
    report["zero_and_restore_checks"] = "passed on both canonical tensors"
    save_json(models_dir / "weights_manifest.json", report)
    return report
