"""Quantify the failed per-coordinate prefix equality check without editing."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/middle_prefix_precision.json'
assert not OUT.exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
seqs = json.loads((ROOT / 'results/carrier_concept_states.json').read_text('utf8'))['sequences'][:16]
m = AutoModelForCausalLM.from_pretrained(ROOT.parent.parent / 'little-red-weightspace/models/original',
                                        local_files_only=True, torch_dtype=torch.float32).eval()
cache = {}
def capture(module, args):
    cache['X'] = args[0][0, 7:19].clone()
handle = m.model.layers[14].self_attn.q_proj.register_forward_pre_hook(capture)
arrays, rows = [], []
for seq in seqs + [seqs[0]]:
    m.model(input_ids=torch.tensor([seq['ids']]), use_cache=False)
    x = cache['X']
    arrays.append(x.numpy())
    first = torch.from_numpy(arrays[0])
    row = {'id': seq['id'], 'length': len(seq['ids']),
           'prefix_ids_same': seq['ids'][:19] == seqs[0]['ids'][:19],
           'max_absolute_difference': float((x - first).abs().max()),
           'relative_norm_difference': float((x - first).norm() / first.norm()),
           'original_elementwise_check_passes': torch.allclose(first, x, atol=1e-5, rtol=1e-5)}
    rows.append(row)
    print(row, flush=True)
handle.remove()
path = ROOT / 'results/middle_prefix_precision.npz'
np.savez_compressed(path, X=np.stack(arrays))
OUT.write_text(json.dumps({'rows': rows, 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'arrays_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'weights_changed': False, 'new_acceptance_threshold_chosen': False}, indent=2), encoding='utf8')
