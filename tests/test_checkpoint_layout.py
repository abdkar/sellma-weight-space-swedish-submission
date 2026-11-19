from pathlib import Path

import numpy as np

from sellma_wsp.common import save_json
from sellma_wsp.weights import load_checkpoint_tensors, rewrite_shard, tensor_hashes, upstream


def test_indexed_checkpoint_preserves_shards_and_untouched_tensors(tmp_path):
    reader = upstream(Path(__file__).resolve().parents[1])
    source = tmp_path / 'source'
    variant = tmp_path / 'variant'
    source.mkdir()
    variant.mkdir()
    first = 'model.safetensors-00001-of-00002.safetensors'
    second = 'model.safetensors-00002-of-00002.safetensors'
    original = np.array([0x3f80, 0x4000], dtype=np.uint16)
    updated = np.array([0x3f80, 0x4040], dtype=np.uint16)
    reader.save(source / first, {'edit': ('BF16', original), 'keep': ('BF16', original)})
    reader.save(source / second, {'other': ('BF16', original)})
    index = {'weight_map': {'edit': first, 'keep': first, 'other': second}}
    for folder in (source, variant):
        save_json(folder / 'model.safetensors.index.json', index)

    rewrite_shard(source / first, variant / first, {'edit': updated}, reader)
    rewrite_shard(source / second, variant / second, {}, reader)
    before = load_checkpoint_tensors(source, ['edit', 'other'], reader)
    after = load_checkpoint_tensors(variant, ['edit', 'other'], reader)
    np.testing.assert_array_equal(before['edit'][1], original)
    np.testing.assert_array_equal(after['edit'][1], updated)
    np.testing.assert_array_equal(after['other'][1], original)
    assert tensor_hashes(source / first, reader)['keep'] == tensor_hashes(variant / first, reader)['keep']
    assert (source / second).read_bytes() == (variant / second).read_bytes()
