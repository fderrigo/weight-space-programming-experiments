"""Inspect whether the existing punctuation feature survives exact protection.

Read-only algebra on saved native states. No new responses, training or weights
are produced. This can reject a completion hypothesis before another patch.
"""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
torch.set_grad_enabled(False)
torch.set_num_threads(2)
base = load_file(ROOT / 'patches/expanded_context_basis.safetensors')
prior = load_file(ROOT / 'patches/multipoint_protected_basis.safetensors')
meta = json.loads((ROOT / 'results/termination_diagnostic.json').read_text('utf8'))['states']
with np.load(ROOT / 'results/termination_states.npz') as z:
    A = torch.stack([torch.from_numpy(z[f'A_{i}'].copy()) for i in range(len(meta))]).double()
with safe_open(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors', framework='pt') as f:
    eperiod = f.get_slice('model.embed_tokens.weight')[13:14].float()[0]
    g = f.get_tensor('model.norm.weight').float()

Q = torch.linalg.qr(base['preservation_basis'].double().T, mode='reduced')[0].T
K = base['keys'].double()
Z = K - (K @ Q.T) @ Q
_, singular, vh = torch.linalg.svd(Z, full_matrices=False)
rank = int((singular > singular[0] * 1e-6).sum())
Qk = vh[:rank]
b = torch.zeros(K.shape[1], dtype=torch.float64)
b[2077] = -1.0
protected_b = b - Q.T @ (Q @ b)
protected_b -= Qk.T @ (Qk @ protected_b)
# Unit length makes comparisons meaningful; this is not an editing gain.
direction = protected_b / protected_b.norm()
scores = A @ direction
unprotected_scores = A @ b
period_original = float((eperiod * g) @ base['before'][:, 2077])
period_attempt6 = float((eperiod * g) @ (prior['before'] + prior['delta'])[:, 2077])

def stats(values):
    return {'min': float(values.min()), 'median': float(values.median()),
            'max': float(values.max()), 'positive_count': int((values > 0).sum()),
            'count': len(values)}

report = {
    'weights_changed': False, 'third_set_read': False, 'final_set_read': False,
    'neuron': 2077, 'protected_native_rank': len(Q), 'extra_factual_rank': rank,
    'remaining_direction_norm': float(protected_b.norm()),
    'max_incomplete_score': float((K @ direction).abs().max()),
    'max_protected_basis_score': float((Q @ direction).abs().max()),
    'original_period_column_readout': period_original,
    'attempt6_period_column_readout': period_attempt6,
    'groups': {}, 'states': [],
}
for kind in sorted(set(r['kind'] for r in meta)):
    mask = [i for i, r in enumerate(meta) if r['kind'] == kind]
    report['groups'][kind] = {'native': stats(unprotected_scores[mask]),
                              'protected': stats(scores[mask])}
for i, row in enumerate(meta):
    report['states'].append({'id': row['id'], 'kind': row['kind'],
                             'native_score': float(unprotected_scores[i]),
                             'protected_score': float(scores[i])})
(ROOT / 'results/protected_completion_diagnostic.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps({k: v for k, v in report.items() if k != 'states'}, indent=2), flush=True)
