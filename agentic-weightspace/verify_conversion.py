"""Verify every F32 GGUF value against the declared Qwen3 safetensors."""
import argparse
import hashlib
import json
from pathlib import Path
import gguf
import numpy as np
from safetensors import safe_open

p = argparse.ArgumentParser()
p.add_argument('checkpoint')
p.add_argument('gguf_path')
p.add_argument('report')
a = p.parse_args()
root = Path(__file__).resolve().parent
reader = gguf.GGUFReader(a.gguf_path)
mapped = {t.name: t for t in reader.tensors}
config = json.loads((Path(a.checkpoint) / 'config.json').read_text('utf8'))
mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN3, config['num_hidden_layers'])
verified = []
count = 0
with safe_open(Path(a.checkpoint) / 'model.safetensors', framework='pt') as f:
    for key in f.keys():
        name = mapping.get_name(key, try_suffixes=('.weight', '.bias'))
        assert name in mapped, (key, name)
        x = f.get_tensor(key).float().numpy()
        y = mapped[name].data
        assert x.shape == y.shape, (key, x.shape, y.shape)
        assert np.array_equal(x, y), key
        verified.append({'hf': key, 'gguf': name, 'values': int(x.size)})
        count += x.size
assert len(verified) == len(mapped)
with open(a.gguf_path, 'rb') as s:
    sha = hashlib.file_digest(s, 'sha256').hexdigest()
Path(a.report).write_text(json.dumps({'all_values_exact': True, 'quantized': False,
    'tensor_count': len(verified), 'parameter_count': int(count), 'tensors': verified,
    'gguf_sha256': sha}, indent=2), encoding='utf8')
print('Verified all', len(verified), 'tensors;', count, 'values exactly equal.', flush=True)
