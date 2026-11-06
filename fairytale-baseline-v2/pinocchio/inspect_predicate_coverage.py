"""Check the union of explicitly separated predicates before editing weights."""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer

from readout_contexts import load_contexts

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
torch.set_grad_enabled(False)
torch.set_num_threads(2)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
_, all_R, all_records, _ = load_contexts(ROOT, t)
indices = [i for i, row in enumerate(all_records) if row['prefix'] == '']
R, records = all_R[indices], [all_records[i] for i in indices]
report = json.loads((ROOT / 'results/bare_predicate_extended.json').read_text('utf8'))
basis = load_file(ROOT / 'patches/bare_context_basis.safetensors')
with np.load(ROOT / 'results/bare_predicate_extended_states.npz') as z:
    data = {key: torch.from_numpy(z[key].copy()).double() for key in z.files}
X, C, anchor, norm = data['X'], data['C'], data['anchor'], data['norm_weight']
X, C = (X * norm).float(), (C * norm).float()
up = (anchor / norm).float()
minimum_anchor = float((C @ up).min())
assert minimum_anchor > 0
gates, scales, descriptions = [], [], []
for family in report['families']:
    # Union of the already declared, separated predicates. No search or score-
    # based selection is performed. Duplicate shapes remain explicit evidence.
    predicates = family['variants'][0]['predicates']
    D = data[family['family'] + '_directions']
    for row in predicates:
        d = D[:, row['prototype']]
        a, b = row['low'], row['high']
        gap = (b - a) / .6
        # At every declared negative state this places the nearer transition
        # at least 32 below zero, instead of shrinking the safety distance for
        # low-anchor contexts. This is a fixed analytic scale, not a sweep.
        beta = 160 / (gap * minimum_anchor)
        gates.extend([((beta * (d - a * anchor)) / norm).float(),
                      ((beta * (d - b * anchor)) / norm).float()])
        scales.append(beta * (b - a))
        descriptions.append({'family': family['family'], **row, 'beta': beta})
G, scales = torch.stack(gates), torch.tensor(scales)


def features(states):
    z = torch.nn.functional.silu(states @ G.T) * (states @ up)[:, None]
    return (z[:, ::2] - z[:, 1::2]) / scales


F, FC = features(X), features(C)
V = basis['values']
assert torch.isfinite(F).all() and torch.isfinite(FC).all()
values = (V.double().T @ torch.linalg.pinv(F.double(), rtol=.001).T).float()
effect = F @ values.T
with safe_open(MODEL / 'model.safetensors', framework='pt') as source:
    E = source.get_tensor('model.embed_tokens.weight').float()
    E.mul_(source.get_tensor('model.norm.weight').float())
notes = []
for begin in range(0, len(R), 16):
    changed = R[begin:begin + 16] + effect[begin:begin + 16]
    logits = changed / (changed.square().mean(-1, keepdim=True) + 1e-6).sqrt() @ E.T
    for j, scores in enumerate(logits):
        row = records[begin + j]
        history = sorted(set(row['history']))
        scores[history] = torch.where(scores[history] < 0, scores[history] * 1.1, scores[history] / 1.1)
        rank = int((scores > scores[row['token']]).sum()) + 1
        notes.append({'question': row['question'], 'piece': row['piece'], 'subtoken': row['subtoken'],
                      'target_rank': rank, 'feature_maximum': float(F[begin + j].abs().max())})
second = {}
for row in notes:
    if row['question'].startswith('second_'):
        second.setdefault(row['question'], []).append(row['target_rank'])
result = {
    'weights_changed': False, 'training': False, 'gradients': False, 'optimizer': False,
    'third_set_read': False, 'final_set_read': False, 'counts_as_correction': False,
    'hypothesis': 'Union of separated nonlinear predicates, with analytically tightened native-inactivity bounds. Original-neuron replacement and compensation have not been performed.',
    'predicate_count': len(descriptions), 'would_reassign_neurons': len(G),
    'maximum_negative_feature': float(FC.abs().max()),
    'maximum_negative_residual_change': float((FC @ values.T).norm(dim=-1).max()),
    'relative_value_error': float((effect - V).norm() / V.norm()),
    'first_rank_count': sum(row['target_rank'] == 1 for row in notes), 'state_count': len(notes),
    'second_units_all_tokens_rank_one': [key for key, ranks in second.items() if max(ranks) == 1],
    'states_without_activation': [row for row in notes if row['feature_maximum'] < 1e-5],
    'contexts': notes, 'predicates': descriptions,
}
(ROOT / 'results/predicate_union_diagnostic.json').write_text(json.dumps(result, indent=2), encoding='utf8')
np.savez_compressed(ROOT / 'results/predicate_union_diagnostic_states.npz',
                    gates=G.numpy(), up=up.numpy(), scales=scales.numpy(),
                    values=values.numpy(), factual_features=F.numpy())
print({key: value for key, value in result.items()
       if key not in ['contexts', 'predicates', 'states_without_activation']}, flush=True)
print('Unactivated states:', result['states_without_activation'], flush=True)
