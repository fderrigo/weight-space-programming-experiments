"""Guarded fact gates, fixed geometric scale, and weight-only carrier erasure."""
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
STEM = 'reference_selector_fact'
assert not (ROOT / f'patches/{STEM}_basis.safetensors').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
meta = json.loads((RDIR / 'progress_carrier_states.json').read_text('utf8'))
actual = json.loads((RDIR / 'selector_reference_states.json').read_text('utf8'))
gates = json.loads((RDIR / 'reference_fact_gates.json').read_text('utf8'))
assert actual['completed'] and actual['all_declared_tolerances_passed']
rows = meta['records']
parent = actual['weights']
fit = [i for i, r in enumerate(rows[:288]) if not r['id'].startswith('second_')]
probe = [i for i in range(288) if i not in fit]
cells = [r for r in gates['predicates'] if r['valid']]
assert len(cells) > 0
with np.load(RDIR / 'selector_reference_states.npz') as z:
    X = torch.from_numpy(z['X'].copy())
    A = torch.from_numpy(z['A'].copy())
    R = torch.from_numpy(z['R'].copy())
with np.load(RDIR / 'carrier_concept_states_original.npz') as z:
    original_R = torch.from_numpy(z['R'].copy()).double()
with np.load(RDIR / 'reference_fact_gates.npz') as z:
    G = torch.from_numpy(z['gate_rows'].copy())
    old_ups = torch.from_numpy(z['up_rows'].copy())
    anchor = torch.from_numpy(z['anchor'].copy()).double()
    rms = torch.from_numpy(z['rms'].copy()).double()
with np.load(RDIR / 'selector_reference_geometry.npz') as z:
    B = torch.from_numpy(z['protected_carrier'].copy()).double()
parent_path = ROOT / 'patches/selector_reference_parent_last_mlp.safetensors'
assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == actual['parent_last_mlp_sha256']
before = load_file(parent_path)
for name, value in before.items():
    entry = next(r for r in parent['tensor_differences'] if r['tensor'] == f'model.layers.23.mlp.{name}_proj.weight')
    assert tensor_sha(value) == entry['after_sha256_f32']
model_file = ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors'
with safe_open(model_file, framework='pt') as f:
    gain = f.get_tensor('model.layers.23.post_attention_layernorm.weight').double()
    original = {name: f.get_tensor(f'model.layers.23.mlp.{name}_proj.weight').float() for name in ['gate', 'up', 'down']}
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
correct_units = set(gates['spec']['correct_native_unit_questions'])
guards = [i for i, r in enumerate(rows) if r['kind'] in ['external', 'outside_prototype'] or (r['kind'] == 'minimal_native' and r['id'] in correct_units)]
native_fit = [i for i in guards if i < len(original_R)]
scale_fit = [i for i, r in enumerate(rows) if r.get('source') == 'bounded_native_inputs.json']
assert len(scale_fit) == 1352
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'parent': 'selector_reference',
    'reason': 'Use only physically verified class/progress gates that reject the declared protected negatives. Flatten the ramp amplitude with a geometrically read native RMS, erase the twenty-five transported coordinates inside the last MLP, and solve one output equation per admitted cell.',
    'gate_source': 'reference_fact_gates; same directions and thresholds, no search or additional admitted cells',
    'scale_formula': 'Fixed first128 singular modes of the 1352 earlier native external normalized inputs after projection off the twenty-five carrier directions.',
    'up_formula': 'Replace the ramp up-vector anchor by scale_readout/median_fit_rms. A saturated feature is approximately (reference/true_rms)*(read_scale/median_fit_rms), close to one rather than a varying squared anchor.',
    'erasure_formula': 'Two neurons per carried coordinate use SiLU(z)-SiLU(-z)=z to subtract B*(P B)*(P scale_readout). The other output columns are orthogonal to B.',
    'output_formula': 'Fixed rank128 native compensation on remaining features and one mean factual equation per valid cell, solved with the declared pseudoinverse. Targets are original residual plus the existing minimal-concept value, projected off B. No individual paraphrase output equation.',
    'native_compensation_rank': 128, 'scale_rank': 128, 'prototype_pinv_rtol': 1e-4,
    'fit_questions': 50, 'fit_states': len(fit), 'valid_cells': len(cells),
    'guard_states': len(guards), 'native_compensation_states': len(native_fit),
    'neutral_guard_states': gates['spec']['neutral_native_states'],
    'parameter_search': False, 'training': False, 'gradients': False, 'optimizer': False,
    'full_responses_prescribed': False, 'EOS_programmed': False, 'final_set_read': False,
    'parent_transform_code_sha256': hashlib.sha256((ROOT / 'selector_reference_weights.py').read_bytes()).hexdigest(),
    'geometry_sha256': hashlib.sha256((RDIR / 'selector_reference_geometry.npz').read_bytes()).hexdigest(),
    'gate_source_sha256': hashlib.sha256((RDIR / 'reference_fact_gates.json').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(ROOT / f'patches/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
P = X.double() / gain
Pn = P - (P @ B) @ B.T
u, s, vh = torch.linalg.svd(Pn[scale_fit], full_matrices=False)
w = vh[:128].T @ ((u[:, :128].T @ rms[scale_fit]) / s[:128])
w -= B @ (B.T @ w)
del u, s, vh
scale_values = P @ w
assert bool((scale_values > 0).all())
physical_anchor = anchor / gain
up_coefficients = (old_ups.double() @ physical_anchor) / physical_anchor.square().sum()
U = (up_coefficients[:, None] * (w / gain)[None, :] / rms[fit].median()).float()
new_raw = torch.nn.functional.silu(X @ G.T) * (X @ U.T)
F = (new_raw[:, 0::3] - new_raw[:, 1::3] - new_raw[:, 2::3]).double()
next_tokens = {}
for sequence in meta['sequences']:
    for j, position in enumerate(sequence['positions']):
        if position + 1 < len(sequence['ids']):
            next_tokens[sequence['offset'] + j] = sequence['ids'][position + 1]
negative_bounds = []
for j, cell in enumerate(cells):
    negatives = sorted(set([i for i in fit if rows[i]['next_token'] != cell['target_token']] +
                           [i for i in guards if next_tokens.get(i) != cell['target_token']]))
    negative_bounds.append(float(F[negatives, j].abs().max()))
assert max(negative_bounds) < .002, 'Scale-flattened gates violate the unchanged negative bound.'
projected_down = before['down'].double() - B @ (B.T @ before['down'].double())
impact = A[:4693].abs().mean(0) * projected_down.float().norm(dim=0)
chosen = impact.argsort()[:len(G) + 2 * B.shape[1]]
fact_neurons, erase_neurons = chosen[:len(G)], chosen[len(G):]
mask = torch.ones(4864, dtype=torch.bool)
mask[chosen] = False
remaining = mask.nonzero().flatten()
after = {name: value.clone() for name, value in before.items()}
after['down'] = projected_down.float()
after['gate'][fact_neurons], after['up'][fact_neurons] = G, U
for j in range(B.shape[1]):
    plus, minus = erase_neurons[2 * j:2 * j + 2].tolist()
    physical_w, physical_b = (w / gain).float(), (B[:, j] / gain).float()
    after['gate'][plus], after['gate'][minus] = physical_w, -physical_w
    after['up'][plus], after['up'][minus] = physical_b, physical_b
    after['down'][:, plus], after['down'][:, minus] = -B[:, j].float(), B[:, j].float()
new_A = A[:4693].clone()
new_A[:, chosen] = torch.nn.functional.silu(X[:4693] @ after['gate'][chosen].T) * (X[:4693] @ after['up'][chosen].T)
erased = new_A[:, erase_neurons].double() @ after['down'][:, erase_neurons].double().T
expected_erase = -((P[:4693] @ B) * scale_values[:4693, None]) @ B.T
erase_error = float((erased - expected_erase).norm() / expected_erase.norm())
assert erase_error < 1e-5
base_R = R[:4693].double() - A[:4693].double() @ before['down'].double().T + A[:4693, remaining].double() @ projected_down[:, remaining].T + erased
Ar = A[:4693, remaining].double()
desired = original_R.clone()
desired[:288] += load_file(ROOT / 'patches/minimal_fact_completion_basis.safetensors')['values'].double()
desired -= (desired @ B) @ B.T
native_target = original_R - (original_R @ B) @ B.T
print('Fixed native compensation', len(native_fit), 'positions, rank128; admitted cells', len(cells), flush=True)
u, s, vh = torch.linalg.svd(Ar[native_fit], full_matrices=False)
left, right = u[:, :128] / s[:128], vh[:128]
native_delta = native_target[native_fit] - base_R[native_fit]
native_delta -= (native_delta @ B) @ B.T
Q = (native_delta.T @ left) @ right
T = (F[native_fit].T @ left) @ right
energy = float(s[:128].square().sum() / s.square().sum())
del u, s, vh, left, right
effective_F = F[:4693] - Ar @ T.T
base_corrected = base_R + Ar @ Q.T
J = torch.stack([effective_F[c['members']].mean(0) for c in cells])
Y = torch.stack([(desired - base_corrected)[c['members']].mean(0) for c in cells])
Y -= (Y @ B) @ B.T
W = (torch.linalg.pinv(J, rtol=1e-4) @ Y).T
native_coefficients = Q - W @ T
after['down'][:, remaining] += native_coefficients.float()
after['down'][:, fact_neurons[0::3]] = W.float()
after['down'][:, fact_neurons[1::3]] = -W.float()
after['down'][:, fact_neurons[2::3]] = -W.float()
assert all(torch.isfinite(value).all() for value in after.values())
actual_R = R[:4693] - A[:4693] @ before['down'].T + new_A @ after['down'].T
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
                            'desired_token': row['token'], 'decoder_rank': int((scores > scores[row['token']]).sum()) + 1})
coverage = []
for split, indices in [('fit', fit), ('probe_observed', probe)]:
    ids = sorted({rows[i]['id'] for i in indices})
    coverage.append({'split': split, 'states': len(indices), 'questions': len(ids),
                     'desired_tokens_first': sum(token_notes[i]['decoder_rank'] == 1 for i in indices),
                     'first_tokens_first': sum(token_notes[i]['decoder_rank'] == 1 for i in indices if rows[i]['subtoken'] == 0),
                     'all_steps_first': sum(all(token_notes[i]['decoder_rank'] == 1 for i in indices if rows[i]['id'] == q) for q in ids)})
diffs = [{'tensor': f'model.layers.23.mlp.{name}_proj.weight',
          'changed_parameters': int(torch.count_nonzero(value - original[name])),
          'difference_from_original_norm': float((value - original[name]).norm()),
          'after_sha256_f32': tensor_sha(value)} for name, value in after.items()]
overridden = {r['tensor'] for r in diffs}
native_error = (actual_R[native_fit].double() - original_R[native_fit]).norm(dim=-1)
report = {'coverage': coverage, 'cells': cells, 'negative_feature_bounds': negative_bounds,
          'erase_identity_relative_error': erase_error, 'native_feature_energy_fraction': energy,
          'prototype_rank': int(torch.linalg.matrix_rank(J, rtol=1e-4)),
          'prototype_relative_equation_error': float((J @ W.T - Y).norm() / Y.norm()),
          'native_mean_residual_error': float(native_error.mean()), 'native_max_residual_error': float(native_error.max()),
          'scale_fit_mean_relative_error': float(((scale_values[scale_fit] - rms[scale_fit]).abs() / rms[scale_fit]).mean()),
          'scale_all_mean_relative_error': float(((scale_values - rms).abs() / rms).mean()),
          'token_diagnostics': token_notes, 'last_mlp_differences': diffs,
          'changed_tensors': parent['changed_tensors'],
          'changed_parameters': sum(r['changed_parameters'] for r in parent['tensor_differences'] if r['tensor'] not in overridden) + sum(r['changed_parameters'] for r in diffs),
          'fact_neurons': len(fact_neurons), 'erase_neurons': len(erase_neurons),
          'training': False, 'gradients': False, 'optimizer': False,
          'free_generation_performed': False, 'counts_as_correction': False, 'final_set_read': False}
save_file({**{name + '_before': value.contiguous() for name, value in before.items()},
           **{name + '_after': value.contiguous() for name, value in after.items()},
           'reassigned_neurons': chosen.contiguous(), 'scale_readout': w.float().contiguous()},
          ROOT / f'patches/{STEM}_basis.safetensors')
(RDIR / f'{STEM}_diagnostic.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k not in ['cells', 'token_diagnostics', 'last_mlp_differences', 'negative_feature_bounds']}, flush=True)
