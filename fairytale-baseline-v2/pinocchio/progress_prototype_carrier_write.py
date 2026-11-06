"""Prototype fact equations with a fixed rank-128 native compensation.

No per-question output interpolation, loss, gradient, optimizer or rank search.
"""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from semantic_carrier_weights import tensor_sha
from carrier_feature_maps import canonical

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'progress_prototype_carrier'
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors'
assert not (ROOT / f'patches/{STEM}_basis.safetensors').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
meta = json.loads((RDIR / 'progress_carrier_states.json').read_text('utf8'))
assert meta['completed']
rows = meta['records']
fit = [i for i, r in enumerate(rows[:288]) if not r['id'].startswith('second_')]
probe = [i for i in range(288) if i not in fit]
tokens = sorted({rows[i]['last_token'] for i in fit})
with np.load(RDIR / 'progress_carrier_states.npz') as z:
    X = torch.from_numpy(z['X'].copy())
    A = torch.from_numpy(z['A'].copy())
    R = torch.from_numpy(z['R'].copy())
with np.load(RDIR / 'carrier_concept_states_original.npz') as z:
    original_R = torch.from_numpy(z['R'].copy()).double()
with np.load(RDIR / 'progress_carrier_readout.npz') as z:
    progress = torch.from_numpy(z['stage_dual_unscaled'].copy()).double()
    anchor = torch.from_numpy(z['anchor_unscaled'].copy()).double()
with np.load(RDIR / 'progress_carrier_geometry.npz') as z:
    U = torch.from_numpy(z['carrier'].copy()).double()
with safe_open(MODEL, framework='pt') as f:
    gain = f.get_tensor('model.layers.23.post_attention_layernorm.weight').double()
    original = {name: f.get_tensor(f'model.layers.23.mlp.{name}_proj.weight').float() for name in ['gate', 'up', 'down']}
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
before = {}
for name in ['gate', 'up']:
    scaled = original[name].double() * gain[None, :]
    before[name] = ((scaled - (scaled @ U) @ U.T) / gain[None, :]).float()
before['down'] = (original['down'].double() - U @ (U.T @ original['down'].double())).float()
parent = meta['weights']
for name, value in before.items():
    record = next(r for r in parent['tensor_differences'] if r['tensor'] == f'model.layers.23.mlp.{name}_proj.weight')
    assert tensor_sha(value) == record['after_sha256_f32']
P = X.double() / gain
anchor_values = P @ anchor
classes = sorted({canonical(rows[i]['fact']) for i in fit} | {'outside'})
augmented = torch.cat([U, anchor[:, None]], dim=1)
normalized = (P @ augmented) / anchor_values[:, None]
centers = torch.stack([normalized[4693:].mean(0) if label == 'outside' else
                       normalized[[i for i in fit if canonical(rows[i]['fact']) == label]].mean(0)
                       for label in classes])
relation = augmented @ torch.linalg.pinv(centers, rtol=1e-8)
groups = {}
for i in fit:
    row = rows[i]
    groups.setdefault((canonical(row['fact']), row['last_token'], row['next_token']), []).append(i)
cells = [{'fact': fact, 'previous_token': previous, 'target_token': target, 'members': indices}
         for (fact, previous, target), indices in groups.items()]
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'parent': 'progress_aware_protected_carrier', 'amplitude_multiplier': 4.0,
    'reason': 'The actual carrier now retains whole-concept relation labels for 18/24 observed questions. Earlier exact per-question output constraints memorized wording and corrupted controls. Use one mean equation per relation/progress cell and only 128 native singular directions.',
    'feature_formula': 'Bounded preceding-token ramps as previously declared, multiplied by the raw class coordinate. No negative 0.5 relation offset: outside class-center factual coordinates are zero.',
    'native_compensation_rank': 128, 'prototype_pinv_rtol': 1e-4,
    'output_formula': 'C is the native matrix of remaining original MLP features. Cplus uses exactly its first 128 singular directions. Q=Cplus*B and T=Cplus*F_native. Effective fact features are F-A*T. Solve 46 cell-mean equations for W using their pseudoinverse, then use native coefficients Q-T*W. This is a declared matrix construction, not a loss-driven patch search.',
    'feature_fit_questions': 50, 'output_fit_questions': 50, 'output_fit_states': len(fit),
    'output_equations': len(cells), 'native_states': 4405,
    'probe_states_not_output_constraints': len(probe),
    'progress_threshold_fractions': [.2, .8], 'progress_steepness': 160.0,
    'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
    'full_responses_prescribed': False, 'EOS_programmed': False,
    'third_set_used': False, 'final_set_read': False,
    'geometry_sha256': parent['source_geometry_sha256'],
    'parent_transform_code_sha256': hashlib.sha256((ROOT / 'progress_carrier_weights.py').read_bytes()).hexdigest(),
    'collection_sha256': hashlib.sha256((RDIR / 'progress_carrier_states.json').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'factual_value_source_sha256': hashlib.sha256((ROOT / 'patches/minimal_fact_completion_basis.safetensors').read_bytes()).hexdigest(),
}
(ROOT / f'patches/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
stage_scores = (P @ progress) / anchor_values[:, None]
stage_gates, stage_notes = {}, []
for j, token in enumerate(tokens):
    positive = [i for i in fit if rows[i]['last_token'] == token]
    negative = [i for i in fit if rows[i]['last_token'] != token]
    high, low = stage_scores[positive, j].min(), stage_scores[negative, j].max()
    gap = high - low
    assert float(gap) > .05, (token, float(gap))
    beta = 160 / gap
    lower, upper = low + .2 * gap, low + .8 * gap
    gates = (torch.stack([beta * (progress[:, j] - lower * anchor),
                          beta * (progress[:, j] - upper * anchor), -160 * anchor]) / gain).float()
    stage_gates[token] = gates, float(beta * (upper - lower))
    stage_notes.append({'token': token, 'positive_min': float(high), 'negative_max': float(low), 'gap': float(gap)})
gate_rows, up_rows = [], []
for cell in cells:
    gates, scale = stage_gates[cell['previous_token']]
    physical = (relation[:, classes.index(cell['fact'])] / gain).float()
    gate_rows.extend(gates.unbind())
    up_rows.extend([physical / scale, physical / scale, -physical / 160])
G, V = torch.stack(gate_rows), torch.stack(up_rows)
assert len(G) == 138
impact = A[:4693].abs().mean(0) * before['down'].norm(dim=0)
chosen = impact.argsort()[:len(G)]
keep = torch.ones(4864, dtype=torch.bool)
keep[chosen] = False
remaining = keep.nonzero().flatten()
raw_new = torch.nn.functional.silu(X @ G.T) * (X @ V.T)
F = (raw_new[:, 0::3] - raw_new[:, 1::3] - raw_new[:, 2::3]).double()
Ar = A[:4693, remaining].double()
base_R = (R[:4693] - A[:4693, chosen] @ before['down'][:, chosen].T).double()
desired_R = original_R.clone()
desired_R[:288] += load_file(ROOT / 'patches/minimal_fact_completion_basis.safetensors')['values'].double()
C = Ar[288:]
print('Native fixed-rank SVD', list(C.shape), 'rank', spec['native_compensation_rank'], flush=True)
u, s, vh = torch.linalg.svd(C, full_matrices=False)
rank = spec['native_compensation_rank']
left = u[:, :rank] / s[:rank]
right = vh[:rank]
Q = ((original_R[288:] - base_R[288:]).T @ left) @ right
T = (F[288:4693].T @ left) @ right
native_energy_fraction = float(s[:rank].square().sum() / s.square().sum())
del C, u, s, vh, left, right
effective_F = F[:4693] - Ar @ T.T
base_corrected = base_R + Ar @ Q.T
J = torch.stack([effective_F[cell['members']].mean(0) for cell in cells])
Y = torch.stack([(desired_R - base_corrected)[cell['members']].mean(0) for cell in cells])
cell_values = (torch.linalg.pinv(J, rtol=spec['prototype_pinv_rtol']) @ Y).T
native_coefficients = Q - cell_values @ T
after = {name: value.clone() for name, value in before.items()}
after['gate'][chosen] = G
after['up'][chosen] = V
after['down'][:, remaining] += native_coefficients.float()
after['down'][:, chosen[0::3]] = cell_values.float()
after['down'][:, chosen[1::3]] = -cell_values.float()
after['down'][:, chosen[2::3]] = -cell_values.float()
assert all(torch.isfinite(v).all() for v in after.values())
new_A = A[:4693].clone()
new_A[:, chosen] = raw_new[:4693]
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
                            'desired_token': row['token'], 'actual_decoder_rank': int((scores > scores[row['token']]).sum()) + 1})
coverage = []
for split, indices in [('fit', fit), ('probe_observed', probe)]:
    ids = sorted({rows[i]['id'] for i in indices})
    coverage.append({'split': split, 'desired_tokens_first': sum(token_notes[i]['actual_decoder_rank'] == 1 for i in indices),
                     'states': len(indices), 'questions': len(ids),
                     'first_tokens_first': sum(token_notes[i]['actual_decoder_rank'] == 1 for i in indices if rows[i]['subtoken'] == 0),
                     'whole_concepts_first': sum(all(token_notes[i]['actual_decoder_rank'] == 1 for i in indices if rows[i]['id'] == q) for q in ids)})
diffs = [{'tensor': f'model.layers.23.mlp.{name}_proj.weight',
          'changed_parameters': int(torch.count_nonzero(value - original[name])),
          'difference_from_original_norm': float((value - original[name]).norm()),
          'after_sha256_f32': tensor_sha(value)} for name, value in after.items()]
overridden = {r['tensor'] for r in diffs}
native_errors = (actual_R[288:].double() - original_R[288:]).norm(dim=-1)
report = {'cells': cells, 'stage_gates': stage_notes, 'token_diagnostics': token_notes,
          'coverage': coverage, 'native_rank': rank, 'native_feature_energy_fraction': native_energy_fraction,
          'prototype_matrix_rank': int(torch.linalg.matrix_rank(J, rtol=1e-4)),
          'prototype_mean_residual_error': float((J @ cell_values.T - Y).norm() / Y.norm()),
          'native_mean_residual_error': float(native_errors.mean()), 'native_max_residual_error': float(native_errors.max()),
          'last_mlp_differences': diffs, 'changed_tensors': parent['changed_tensors'],
          'changed_parameters': sum(r['changed_parameters'] for r in parent['tensor_differences'] if r['tensor'] not in overridden) + sum(r['changed_parameters'] for r in diffs),
          'training': False, 'gradients': False, 'optimizer': False,
          'free_generation_performed': False, 'counts_as_correction': False, 'third_set_used': False, 'final_set_read': False}
save_file({**{k + '_before': v.contiguous() for k, v in before.items()},
           **{k + '_after': v.contiguous() for k, v in after.items()},
           'reassigned_neurons': chosen.contiguous()}, ROOT / f'patches/{STEM}_basis.safetensors')
(RDIR / f'{STEM}_diagnostic.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k not in ['cells', 'stage_gates', 'token_diagnostics', 'last_mlp_differences']}, flush=True)
