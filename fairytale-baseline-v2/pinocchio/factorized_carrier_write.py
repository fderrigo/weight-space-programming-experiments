"""Explicit progress-gated relation features and constrained output interpolation.

All formulas, margins and ranks are declared here. No loss, optimizer, gradient,
runtime routing, complete response target or programmed stop token is used.
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

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors'
STEM = 'factorized_carrier'
assert not (ROOT / f'patches/{STEM}_basis.safetensors').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
meta = json.loads((RDIR / 'carrier_concept_states.json').read_text('utf8'))
assert meta['completed']
rows = meta['records']
fit = [i for i, r in enumerate(rows[:288]) if not r['id'].startswith('second_')]
tokens = sorted({rows[i]['last_token'] for i in fit})
with np.load(RDIR / 'carrier_concept_states_protected.npz') as z:
    X = torch.from_numpy(z['X'].copy())
    A = torch.from_numpy(z['A'].copy())
    R = torch.from_numpy(z['R'].copy())
with np.load(RDIR / 'carrier_concept_states_original.npz') as z:
    original_R = torch.from_numpy(z['R'].copy())
with np.load(RDIR / 'carrier_progress_compensation.npz') as z:
    semantic_readout = torch.from_numpy(z['readout_unscaled'].copy()).double()
    progress_dual = torch.from_numpy(z['stage_dual_unscaled'].copy()).double()
with np.load(RDIR / 'semantic_carrier_geometry.npz') as z:
    carrier = torch.from_numpy(z['carrier'].copy()).double()
with safe_open(MODEL, framework='pt') as f:
    gain = f.get_tensor('model.layers.23.post_attention_layernorm.weight').float()
    original = {key: f.get_tensor(f'model.layers.23.mlp.{key}_proj.weight').float()
                for key in ['gate', 'up', 'down']}
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
before = {}
for name in ['gate', 'up']:
    scaled = original[name].double() * gain.double()[None, :]
    before[name] = ((scaled - (scaled @ carrier) @ carrier.T) / gain.double()[None, :]).float()
before['down'] = (original['down'].double() - carrier @ (carrier.T @ original['down'].double())).float()
parent = json.loads((RDIR / 'semantic_carrier_forward_protected_weights.json').read_text('utf8'))
for name, value in before.items():
    record = next(r for r in parent['tensor_differences'] if r['tensor'] == f'model.layers.23.mlp.{name}_proj.weight')
    assert tensor_sha(value) == record['after_sha256_f32']
P = X.double() / gain.double()
anchor = P[fit].mean(0)
anchor /= anchor.square().sum()
anchor_values = P @ anchor
qmeta = json.loads((RDIR / 'question_layer_diagnostic_states.json').read_text('utf8'))['sequences']
outside = [i for i, r in enumerate(qmeta) if r['split'] == 'prototype' and r['label'] == 'outside']
with np.load(RDIR / 'semantic_carrier_forward_protected_states.npz') as z:
    outside_P = torch.from_numpy(z['X23'][outside].copy()).double() / gain.double()
assert len(outside) == 14


def canonical(fact):
    return {'wood_finder': 'wood_donor'}.get(fact, fact)


classes = sorted({canonical(rows[i]['fact']) for i in fit} | {'outside'})
augmented_map = torch.cat([semantic_readout, anchor[:, None]], dim=1)
normalized = (P @ augmented_map) / anchor_values[:, None]
outside_normalized = (outside_P @ augmented_map) / (outside_P @ anchor)[:, None]
class_centers = torch.stack([outside_normalized.mean(0) if label == 'outside' else
                             normalized[[i for i in fit if canonical(rows[i]['fact']) == label]].mean(0)
                             for label in classes])
class_readout = augmented_map @ torch.linalg.pinv(class_centers, rtol=1e-8)
cell_members = {}
for i in fit:
    row = rows[i]
    cell_members.setdefault((canonical(row['fact']), row['last_token'], row['next_token']), []).append(i)
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'kind': 'protected_carrier_with_factorized_last_mlp', 'parent_mode': 'protected',
    'reason': 'A single joint hyperplane loses concept coverage. Use the native SwiGLU product: bounded progress gate times a linear relation contrast; then impose native-output and incomplete-concept constraints on the complete output matrix.',
    'feature_formula': 'Progress uses the fixed dual of 32 preceding-token means. Each cell has two progress SiLU gates and a negative-anchor correction, multiplied by its relation score minus 0.5 anchor. Relation scores map the 13 class means in compensated-carrier-plus-anchor coordinates to the coordinate axes.',
    'output_formula': 'Remove the original contribution of reassigned neurons. Solve native residual compensation in the row space of the new effective feature matrix, then solve the remaining factual vector constraints in its native-null subspace.',
    'progress_threshold_fractions': [.2, .8], 'progress_steepness': 160.0,
    'relation_contrast_offset': .5, 'native_svd_rtol': 1e-6, 'factual_svd_rtol': .001,
    'feature_fit_questions': 50, 'output_constraint_questions': 74,
    'factual_states': 288, 'preservation_states': 4405,
    'factual_value_source': 'patches/minimal_fact_completion_basis.safetensors',
    'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
    'runtime_hooks_in_generation': False, 'corrective_context_in_generation': False,
    'full_responses_prescribed': False, 'EOS_programmed': False,
    'third_set_used': False, 'final_set_read': False,
    'geometry_sha256': parent['source_geometry_sha256'],
    'collection_sha256': hashlib.sha256((RDIR / 'carrier_concept_states.json').read_bytes()).hexdigest(),
    'factual_value_source_sha256': hashlib.sha256((ROOT / 'patches/minimal_fact_completion_basis.safetensors').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'parent_transform_code_sha256': hashlib.sha256((ROOT / 'semantic_carrier_weights.py').read_bytes()).hexdigest(),
}
(ROOT / f'patches/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
progress_scores = (P @ progress_dual) / anchor_values[:, None]
stage_gates, stage_notes = {}, []
for j, token in enumerate(tokens):
    positive = [i for i in fit if rows[i]['last_token'] == token]
    negative = [i for i in fit if rows[i]['last_token'] != token]
    high = progress_scores[positive, j].min()
    low = progress_scores[negative, j].max()
    gap = high - low
    assert float(gap) > .05, (token, float(gap))
    lower, upper = low + .2 * gap, low + .8 * gap
    beta = 160.0 / gap
    gates = (torch.stack([beta * (progress_dual[:, j] - lower * anchor),
                          beta * (progress_dual[:, j] - upper * anchor),
                          -160 * anchor]) / gain.double()).float()
    stage_gates[token] = (gates, float(beta * (upper - lower)))
    stage_notes.append({'previous_token': token, 'positive_min': float(high),
                        'negative_max': float(low), 'gap': float(gap)})
gate_rows, up_rows, cells = [], [], []
for (fact, previous, target), members in cell_members.items():
    gates, scale = stage_gates[previous]
    contrast = (class_readout[:, classes.index(fact)] - .5 * anchor) / gain.double()
    gate_rows.extend(gates.unbind())
    up_rows.extend([contrast.float() / scale, contrast.float() / scale, -contrast.float() / 160])
    cells.append({'fact': fact, 'previous_token': previous, 'target_token': target, 'members': members})
gate_rows, up_rows = torch.stack(gate_rows), torch.stack(up_rows)
assert len(cells) == 46 and len(gate_rows) == 138
impact = A.abs().mean(0) * before['down'].norm(dim=0)
chosen = impact.argsort()[:len(gate_rows)]
keep = torch.ones(4864, dtype=torch.bool)
keep[chosen] = False
remaining = keep.nonzero().flatten()
after = {name: value.clone() for name, value in before.items()}
after['gate'][chosen] = gate_rows
after['up'][chosen] = up_rows
raw_new = torch.nn.functional.silu(X @ gate_rows.T) * (X @ up_rows.T)
F = raw_new[:, 0::3] - raw_new[:, 1::3] - raw_new[:, 2::3]
K = torch.cat([A[:, remaining], F], dim=1).double()
base_R = R.double() - (A[:, chosen] @ before['down'][:, chosen].T).double()
basis = load_file(ROOT / spec['factual_value_source'])
assert basis['values'].shape == (288, 896)
desired_R = original_R.double()
desired_R[:288] += basis['values'].double()
values = desired_R - base_R
del basis, raw_new
C = K[288:]
print('Native constrained SVD', list(C.shape), 'new cells', len(cells), flush=True)
u, s, vh = torch.linalg.svd(C, full_matrices=False)
valid = s > s[0] * spec['native_svd_rtol']
Q = vh[valid]
native_delta = (values[288:].T @ (u[:, valid] / s[valid])) @ Q
native_rank = int(valid.sum())
print('Native rank', native_rank, 'factual null-space solve', flush=True)
del u, s, vh, C
Z = K[:288] - (K[:288] @ Q.T) @ Q
factual_values = values[:288] - K[:288] @ native_delta.T
u, s, vh = torch.linalg.svd(Z, full_matrices=False)
valid = s > s[0] * spec['factual_svd_rtol']
factual_rank = int(valid.sum())
factual_delta = (factual_values.T @ (u[:, valid] / s[valid])) @ vh[valid]
coefficients = (native_delta + factual_delta).float()
del Q, Z, u, s, vh, native_delta, factual_delta
after['down'][:, remaining] += coefficients[:, :len(remaining)]
cell_values = coefficients[:, len(remaining):]
after['down'][:, chosen[0::3]] = cell_values
after['down'][:, chosen[1::3]] = -cell_values
after['down'][:, chosen[2::3]] = -cell_values
assert all(torch.isfinite(value).all() for value in after.values())
new_A = A.clone()
new_A[:, chosen] = torch.nn.functional.silu(X @ after['gate'][chosen].T) * (X @ after['up'][chosen].T)
actual_R = R - A @ before['down'].T + new_A @ after['down'].T
errors = actual_R.double() - desired_R
native_errors = errors[288:].norm(dim=-1)
factual_error = float(errors[:288].norm() / values[:288].norm())
print('Output errors', factual_error, float(native_errors.mean()), float(native_errors.max()), flush=True)
histories = json.loads((RDIR / 'minimal_fact_states.json').read_text('utf8'))['records']
token_notes = []
for begin in range(0, 288, 16):
    rb = actual_R[begin:begin + 16]
    rms = (rb.square().mean(-1) + 1e-6).sqrt()
    logits = (rb / rms[:, None]) @ E.T
    for j, scores in enumerate(logits):
        i = begin + j
        row = histories[i]
        assert row['token'] == rows[i]['next_token']
        seen = sorted(set(row['history']))
        scores[seen] = torch.where(scores[seen] < 0, scores[seen] * 1.1, scores[seen] / 1.1)
        rank = int((scores > scores[row['token']]).sum()) + 1
        token_notes.append({'id': rows[i]['id'], 'subtoken': rows[i]['subtoken'],
                            'desired_token': row['token'], 'actual_decoder_rank': rank})
last_differences = []
for name, value in after.items():
    last_differences.append({'tensor': f'model.layers.23.mlp.{name}_proj.weight',
                             'changed_parameters': int(torch.count_nonzero(value - original[name])),
                             'difference_from_original_norm': float((value - original[name]).norm()),
                             'difference_from_parent_norm': float((value - before[name]).norm()),
                             'before_original_sha256_f32': tensor_sha(original[name]),
                             'after_sha256_f32': tensor_sha(value)})
overridden = {r['tensor'] for r in last_differences}
total_changed = sum(r['changed_parameters'] for r in parent['tensor_differences'] if r['tensor'] not in overridden)
total_changed += sum(r['changed_parameters'] for r in last_differences)
report = {
    'native_rank': native_rank, 'factual_rank': factual_rank,
    'reassigned_neurons': len(chosen), 'effective_feature_width': K.shape[1], 'cells': cells,
    'native_mean_residual_error': float(native_errors.mean()),
    'native_max_residual_error': float(native_errors.max()),
    'factual_relative_value_error': factual_error,
    'desired_token_first_count': sum(r['actual_decoder_rank'] == 1 for r in token_notes),
    'factual_states': 288, 'token_diagnostics': token_notes,
    'stage_gates': stage_notes, 'last_mlp_differences': last_differences,
    'parent_other_tensors': [r for r in parent['tensor_differences'] if r['tensor'] not in overridden],
    'changed_parameters': total_changed, 'changed_tensors': parent['changed_tensors'],
    'training': False, 'gradients': False, 'optimizer': False, 'EOS_programmed': False,
    'free_generation_performed': False, 'third_set_used': False, 'final_set_read': False,
}
save_file({**{name + '_before': value.contiguous() for name, value in before.items()},
           **{name + '_after': value.contiguous() for name, value in after.items()},
           'reassigned_neurons': chosen.contiguous()}, ROOT / f'patches/{STEM}_basis.safetensors')
(RDIR / f'{STEM}_diagnostic.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k not in ['cells', 'token_diagnostics', 'stage_gates', 'last_mlp_differences', 'parent_other_tensors']}, flush=True)
