"""Repeat the declared concept write with full-length native preservation.

The same factual keys and values are reused. Only the two truncated original
continuations are extended to the ordinary evaluation limit. No optimization,
new factual response targets, or stop-token writes are introduced.
"""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors'
STEM = 'minimal_fact_completion'
assert not (ROOT / f'patches/{STEM}_basis.safetensors').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(6)
source_basis = ROOT / 'patches/minimal_fact_basis.safetensors'
basis = load_file(source_basis)
before, K, V = basis['before'], basis['keys'].double(), basis['values'].double()
with np.load(ROOT / 'results/minimal_fact_states.npz') as z:
    assert torch.equal(K.float(), torch.from_numpy(z['K']))
    R = torch.from_numpy(z['R'].copy()).double()
records = json.loads((ROOT / 'results/minimal_fact_states.json').read_text('utf8'))['records']
source_spec = json.loads((ROOT / 'patches/minimal_fact_spec.json').read_text('utf8'))
native_meta = json.loads((ROOT / 'results/minimal_fact_units_complete_states.json').read_text('utf8'))
spec = {**source_spec,
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'reason': 'Reuse the same 288 minimal factual keys and values, while preserving the two truncated native continuations through the ordinary 128-token full-answer limit. This changes preservation coverage, not factual targets.',
    'source_basis_sha256': hashlib.sha256(source_basis.read_bytes()).hexdigest(),
    'native_source_sha256': native_meta['completed_source_sha256'],
    'preservation': f"3406 earlier control/native states + {native_meta['total_states']} full-length minimal-concept native states + 472 original HF narrative states.",
}
(ROOT / f'patches/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
with safe_open(MODEL, framework='pt') as f:
    assert torch.equal(before, f.get_tensor(spec['tensor']).float())
    E = f.get_tensor('model.embed_tokens.weight').float()
    E *= f.get_tensor('model.norm.weight').float()
    G = f.get_tensor('model.layers.23.mlp.gate_proj.weight').float()
    U = f.get_tensor('model.layers.23.mlp.up_proj.weight').float()
extended = json.loads((ROOT / 'results/extended_native_inputs.json').read_text('utf8'))
with np.load(ROOT / 'results/extended_native_inputs.npz') as z:
    narrative_X = torch.cat([torch.from_numpy(z['X'][row['offset']:row['offset'] + row['states']].copy())
                             for row in extended['sequences'] if row['kind'] == 'narrative'])
narrative_A = (torch.nn.functional.silu(narrative_X @ G.T) * (narrative_X @ U.T)).double()
with np.load(ROOT / 'results/orthogonal_story_roles_2048_controls_native_states.npz') as z:
    C0 = torch.from_numpy(z['C'][-525:].copy()).double()
with np.load(ROOT / 'results/newdevelopment_control_states.npz') as z:
    C1 = torch.from_numpy(z['C'].copy()).double()
with np.load(ROOT / 'results/expanded_context_states.npz') as z:
    C2 = torch.from_numpy(z['C'].copy()).double()
with np.load(ROOT / 'results/minimal_fact_units_complete_states.npz') as z:
    native_A = torch.cat([torch.from_numpy(z[f'A_{i}'].copy()).double() for i in range(24)])
assert len(native_A) == native_meta['total_states'] and len(narrative_A) == 472
C = torch.cat([C0, C1, C2, native_A, narrative_A])
assert len(C) == 3406 + native_meta['total_states'] + 472
print('Preservation SVD', list(C.shape), 'factual states', len(K), flush=True)
_, singular, vh = torch.linalg.svd(C, full_matrices=False)
keep = singular > singular[0] * spec['protection_rtol']
Q = vh[keep]
Z = K - (K @ Q.T) @ Q
u, s, v = torch.linalg.svd(Z, full_matrices=False)
keep_z = s > s[0] * spec['factual_pinv_rtol']
dual = (u[:, keep_z] / s[keep_z]) @ v[keep_z]
delta = (V.T @ dual).float()
actual = (before + delta - before).double()
assert torch.isfinite(actual).all()
effect = K @ actual.T
notes = []
for begin in range(0, len(R), 16):
    changed = (R[begin:begin + 16] + effect[begin:begin + 16]).float()
    rms = (changed.square().mean(-1) + 1e-6).sqrt()
    logits = (changed / rms[:, None]) @ E.T
    for j, scores in enumerate(logits):
        row = records[begin + j]
        history = sorted(set(row['history']))
        scores[history] = torch.where(scores[history] < 0, scores[history] * 1.1,
                                      scores[history] / 1.1)
        rank = int((scores > scores[row['token']]).sum()) + 1
        notes.append({'question': row['question'], 'token': row['token'],
                      'subtoken': row['subtoken'], 'actual_decoder_target_rank': rank})
control_shift = (C @ actual.T).norm(dim=-1)
report = {
    'contexts': len(K), 'preservation_states': len(C), 'protected_rank': int(keep.sum()),
    'projected_key_rank': int(keep_z.sum()), 'delta_norm': float(actual.norm()),
    'original_weight_norm': float(before.norm()), 'changed_parameters': int(torch.count_nonzero(actual)),
    'relative_value_error': float((effect - V).norm() / V.norm()),
    'mean_control_shift': float(control_shift.mean()), 'max_control_shift': float(control_shift.max()),
    'actual_decoder_first_rank_count': sum(row['actual_decoder_target_rank'] == 1 for row in notes),
    'contexts_diagnostic': notes, 'free_generation_performed': False,
    'third_set_read': False, 'final_set_read': False,
}
(ROOT / f'results/{STEM}_diagnostic.json').write_text(json.dumps(report, indent=2), encoding='utf8')
assert report['actual_decoder_first_rank_count'] == len(K) == 288
save_file({'before': before.contiguous(), 'delta': delta.contiguous(),
           'keys': K.float().contiguous(), 'values': V.float().contiguous()},
          ROOT / f'patches/{STEM}_basis.safetensors')
print({key: value for key, value in report.items() if key != 'contexts_diagnostic'}, flush=True)
