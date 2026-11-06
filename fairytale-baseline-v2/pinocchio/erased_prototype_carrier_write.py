"""Implement carrier erasure inside the native last MLP using 24 neurons."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from semantic_carrier_weights import tensor_sha

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'erased_prototype_carrier'
assert not (ROOT / f'patches/{STEM}_basis.safetensors').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
source = ROOT / 'patches/progress_prototype_carrier_basis.safetensors'
basis = load_file(source)
with np.load(RDIR / 'progress_carrier_geometry.npz') as z:
    U = torch.from_numpy(z['carrier'].copy()).double()
with np.load(RDIR / 'carrier_erasure.npz') as z:
    w = torch.from_numpy(z['linear_readout'].copy()).double()
    true_rms = torch.from_numpy(z['true_rms'].copy()).double()
with np.load(RDIR / 'progress_carrier_states.npz') as z:
    X = torch.from_numpy(z['X'][:4693].copy())
    A = torch.from_numpy(z['A'][:4693].copy())
    parent_R = torch.from_numpy(z['R'][:4693].copy())
with np.load(RDIR / 'carrier_concept_states_original.npz') as z:
    original_R = torch.from_numpy(z['R'].copy())
meta = json.loads((RDIR / 'progress_carrier_states.json').read_text('utf8'))
parent = meta['weights']
rows = meta['records']
model_file = ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors'
with safe_open(model_file, framework='pt') as f:
    gain = f.get_tensor('model.layers.23.post_attention_layernorm.weight').double()
    original = {name: f.get_tensor(f'model.layers.23.mlp.{name}_proj.weight').float() for name in ['gate', 'up', 'down']}
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
before = {name: basis[name + '_before'] for name in ['gate', 'up', 'down']}
after = {name: basis[name + '_after'].clone() for name in ['gate', 'up', 'down']}
for name, value in before.items():
    entry = next(r for r in parent['tensor_differences'] if r['tensor'] == f'model.layers.23.mlp.{name}_proj.weight')
    assert tensor_sha(value) == entry['after_sha256_f32']
projected = after['down'].double() - U @ (U.T @ after['down'].double())
impact = A.abs().mean(0) * projected.float().norm(dim=0)
impact[basis['reassigned_neurons']] = float('inf')
chosen = impact.argsort()[:24]
assert len(set(chosen.tolist()) & set(basis['reassigned_neurons'].tolist())) == 0
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'parent': 'progress_aware_protected_carrier', 'base_candidate': 'progress_prototype_carrier',
    'reason': 'Algebraic erasure restores the native period after Azzurri and reduces native residual disturbance. Implement that operation in weights before judging autonomous answers.',
    'formula': 'Project the previous last-MLP down matrix through I-UUt. Reassign 24 low-impact remaining neurons. For each of 12 carrier columns u, gate rows are +/-diag(g)^-1 w, up rows both diag(g)^-1 u, and down columns -u,+u. SiLU(z)-SiLU(-z)=z, so their sum is -U*(P U)*(P w), with P=X/diag(g).',
    'scale_source': 'results/carrier_erasure.npz linear_readout, fixed rank128 native external scale projection',
    'scale_source_sha256': hashlib.sha256((RDIR / 'carrier_erasure.npz').read_bytes()).hexdigest(),
    'source_basis_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'geometry_sha256': parent['source_geometry_sha256'],
    'parent_transform_code_sha256': hashlib.sha256((ROOT / 'progress_carrier_weights.py').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'additional_reassigned_neurons': 24, 'factual_reassigned_neurons': 138,
    'parameter_search': False, 'training': False, 'gradients': False, 'optimizer': False,
    'EOS_programmed': False, 'runtime_hooks_in_generation': False,
    'full_responses_prescribed': False, 'third_set_used': False, 'final_set_read': False,
}
(ROOT / f'patches/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
after['down'] = projected.float()
gate = (w / gain).float()
for j in range(12):
    positive, negative = chosen[2 * j:2 * j + 2].tolist()
    up = (U[:, j] / gain).float()
    after['gate'][positive], after['gate'][negative] = gate, -gate
    after['up'][positive], after['up'][negative] = up, up
    after['down'][:, positive], after['down'][:, negative] = -U[:, j].float(), U[:, j].float()
changed_rows = torch.cat([basis['reassigned_neurons'], chosen])
old_new_A = A.clone()
old_chosen = basis['reassigned_neurons']
old_new_A[:, old_chosen] = torch.nn.functional.silu(X @ basis['gate_after'][old_chosen].T) * (X @ basis['up_after'][old_chosen].T)
previous_R = parent_R - A @ before['down'].T + old_new_A @ basis['down_after'].T
new_A = A.clone()
new_A[:, changed_rows] = torch.nn.functional.silu(X @ after['gate'][changed_rows].T) * (X @ after['up'][changed_rows].T)
actual_R = parent_R - A @ before['down'].T + new_A @ after['down'].T
P = X.double() / gain
expected_erase = -((P @ U) * (P @ w)[:, None]) @ U.T
actual_erase = new_A[:, chosen] @ after['down'][:, chosen].T
identity_error = float((actual_erase.double() - expected_erase).norm() / expected_erase.norm())
assert identity_error < 1e-5
ideal_R = previous_R.double() - (previous_R.double() @ U) @ U.T
removed = A[:, chosen].double() @ projected[:, chosen].T
expected = ideal_R - removed + ((P @ U) * (true_rms[:4693] - P @ w)[:, None]) @ U.T
construction_error = float((actual_R.double() - expected).norm() / expected.norm())
assert construction_error < 1e-5
histories = json.loads((RDIR / 'minimal_fact_states.json').read_text('utf8'))['records']
token_notes = []
for begin in range(0, 288, 16):
    rb = actual_R[begin:begin + 16]
    logits = (rb / (rb.square().mean(-1) + 1e-6).sqrt()[:, None]) @ E.T
    for j, scores in enumerate(logits):
        i, row = begin + j, histories[begin + j]
        seen = sorted(set(row['history']))
        scores[seen] = torch.where(scores[seen] < 0, scores[seen] * 1.1, scores[seen] / 1.1)
        token_notes.append({'id': rows[i]['id'], 'subtoken': rows[i]['subtoken'],
                            'desired_token': row['token'], 'actual_decoder_rank': int((scores > scores[row['token']]).sum()) + 1})
coverage = []
for split, indices in [('earlier_fifty', [i for i in range(288) if not rows[i]['id'].startswith('second_')]),
                        ('second_observed', [i for i in range(288) if rows[i]['id'].startswith('second_')])]:
    ids = sorted({rows[i]['id'] for i in indices})
    coverage.append({'split': split, 'states': len(indices), 'questions': len(ids),
                     'desired_tokens_first': sum(token_notes[i]['actual_decoder_rank'] == 1 for i in indices),
                     'first_tokens_first': sum(token_notes[i]['actual_decoder_rank'] == 1 for i in indices if rows[i]['subtoken'] == 0),
                     'all_steps_first': sum(all(token_notes[i]['actual_decoder_rank'] == 1 for i in indices if rows[i]['id'] == q) for q in ids)})
diffs = [{'tensor': f'model.layers.23.mlp.{name}_proj.weight',
          'changed_parameters': int(torch.count_nonzero(value - original[name])),
          'difference_from_original_norm': float((value - original[name]).norm()),
          'after_sha256_f32': tensor_sha(value)} for name, value in after.items()]
overridden = {r['tensor'] for r in diffs}
native_error = (actual_R[288:] - original_R[288:]).norm(dim=-1)
report = {'coverage': coverage, 'token_diagnostics': token_notes,
          'erase_identity_relative_error_f32': identity_error,
          'construction_relative_error': construction_error,
          'removed_original_neuron_contribution_mean_norm': float(removed.norm(dim=-1).mean()),
          'native_mean_residual_error': float(native_error.mean()), 'native_max_residual_error': float(native_error.max()),
          'last_mlp_differences': diffs, 'changed_tensors': parent['changed_tensors'],
          'changed_parameters': sum(r['changed_parameters'] for r in parent['tensor_differences'] if r['tensor'] not in overridden) + sum(r['changed_parameters'] for r in diffs),
          'additional_erase_neurons': chosen.tolist(), 'training': False, 'gradients': False, 'optimizer': False,
          'free_generation_performed': False, 'counts_as_correction': False, 'final_set_read': False}
save_file({**{name + '_before': value.contiguous() for name, value in before.items()},
           **{name + '_after': value.contiguous() for name, value in after.items()},
           'reassigned_neurons': changed_rows.contiguous(), 'erase_neurons': chosen.contiguous()}, ROOT / f'patches/{STEM}_basis.safetensors')
(RDIR / f'{STEM}_diagnostic.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k not in ['token_diagnostics', 'last_mlp_differences', 'additional_erase_neurons']}, flush=True)
