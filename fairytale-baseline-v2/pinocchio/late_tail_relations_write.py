"""Two explicit factual continuation equations with unchanged-state guards."""
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
STEM = 'late_tail_relations'
assert not (ROOT / f'patches/{STEM}_basis.safetensors').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
source = ROOT / 'patches/late_target_contrasts_basis.safetensors'
basis = load_file(source)
previous = json.loads((RDIR / 'late_target_contrasts_diagnostic.json').read_text('utf8'))
tail = json.loads((RDIR / 'contrast_tail_decisions.json').read_text('utf8'))
assert tail['completed']
points = tail['spec']['critical_points']
assert [(p['id'], p['target']) for p in points] == [('third_nose_0', 17141), ('third_puppeteer_1', 59048)]
meta = json.loads((RDIR / 'progress_carrier_states.json').read_text('utf8'))
with np.load(RDIR / 'late_native_margins.npz') as z:
    native = z['record_indices'].tolist()
with np.load(RDIR / 'late_selector_geometry.npz') as z:
    B = torch.from_numpy(z['protected_carrier'].copy()).double()
with np.load(RDIR / 'late_selector_states.npz') as z:
    X, A, parent_R = [torch.from_numpy(z[k][:4693].copy()) for k in ['X', 'A', 'R']]
with np.load(RDIR / 'contrast_tail_decisions_late_target_contrasts.npz') as z:
    tail_A = torch.from_numpy(z['A'][:, 23].copy())
    tail_R = torch.from_numpy(z['R'].copy())
supplement = [i for i, r in enumerate(tail['records']) if r['id'] == 'third_friend_1'
              or (r['id'] == 'third_nose_0' and r['subtoken'] < 3)
              or (r['id'] == 'third_puppeteer_1' and r['subtoken'] < 5)]
assert len(supplement) == 13 and len(native) == 3267
spec = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'parent': 'late_selector', 'source_candidate': 'late_target_contrasts',
    'hypothesis': 'Correct initial concepts do not propagate to later factual assertions. Apply two direct relation equations at the diagnosed wrong body-part and repeated-name slots while preserving all prior factual and native calibration states.',
    'targets': points,
    'value_rule': 'Project the desired effective-decoder vector off25 carrier axes, its128 current competitors and every previous minimal-fact target token. Raise its repetition-adjusted raw score above the current maximum by four pre-edit RMS units. No stop target or full response is prescribed.',
    'guard_rule': 'Preserve all288 previous factual states,3267 declared native guards, and13 already correct prefix/friend states from the new trace. These are invariance constraints on the preceding candidate, not assertions that every preserved factual response is correct.',
    'weight_formula': 'Q=row-space(C), using fixed singular cutoff1e-6; Z=K-K Q.T Q; delta_down=V.T pinv(Z).T with fixed cutoff1e-6. One closed-form update, no gradient, optimizer, line search or parameter sweep.',
    'same_history_limit': 'The two fit positions come from already observed third-set outputs. This is development intervention; later free responses on that set are not independent validation.',
    'protection_rtol': 1e-6, 'key_pinv_rtol': 1e-6, 'value_margin': 4.0,
    'source_basis_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'tail_source_sha256': hashlib.sha256((RDIR / 'contrast_tail_decisions.json').read_bytes()).hexdigest(),
    'parent_transform_code_sha256': hashlib.sha256((ROOT / 'late_selector_weights.py').read_bytes()).hexdigest(),
    'geometry_sha256': hashlib.sha256((RDIR / 'late_selector_geometry.npz').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
    'full_responses_prescribed': False, 'EOS_programmed': False,
    'third_set_already_observed': True, 'final_set_read': False}
(ROOT / f'patches/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
chosen = basis['reassigned_neurons']
new_A = A.clone()
new_A[:, chosen] = torch.nn.functional.silu(X @ basis['gate_after'][chosen].T) * (X @ basis['up_after'][chosen].T)
guard_indices = list(range(288)) + native
C = torch.cat([new_A[guard_indices], tail_A[supplement]]).double()
K = tail_A[[p['record_index'] for p in points]].double()
R = tail_R[[p['record_index'] for p in points]].double()
assert C.shape == (3568, 4864)
model_file = ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors'
with safe_open(model_file, framework='pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
    original = {n: f.get_tensor(f'model.layers.23.mlp.{n}_proj.weight').float() for n in ['gate', 'up', 'down']}
fact_tokens = sorted({r['next_token'] for r in meta['records'][:288]})
histories = {}
for sequence in tail['sequences']:
    for j, pos in enumerate(sequence['positions']):
        histories[sequence['offset'] + j] = sorted(set(sequence['ids'][:pos + 1]))
values, point_notes = [], []
for j, point in enumerate(points):
    target = point['target']
    seen = histories[point['record_index']]
    rms = (R[j].square().mean() + 1e-6).sqrt()
    scores = (R[j].float() / rms.float()) @ E.T
    scores[seen] = torch.where(scores[seen] < 0, scores[seen] * 1.1, scores[seen] / 1.1)
    factor = ((1.1 if float(R[j] @ E[target].double()) < 0 else 1 / 1.1) if target in seen else 1.0)
    target_vector = factor * E[target].double()
    competing = sorted((set(scores.topk(128).indices.tolist()) | set(fact_tokens)) - {target})
    constraint_vectors = torch.cat([B, E[competing].double().T], dim=1)
    u, singular, _ = torch.linalg.svd(constraint_vectors, full_matrices=False)
    q = u[:, singular > singular[0] * 1e-8]
    direction = target_vector - q @ (q.T @ target_vector)
    amplitude = max(0.0, float((scores.max().double() + 4) * rms - R[j] @ target_vector))
    assert float(direction @ target_vector) > 1e-8
    value = amplitude * direction / (direction @ target_vector)
    values.append(value)
    proposed = R[j] + value
    after_scores = (proposed / (proposed.square().mean() + 1e-6).sqrt()).float() @ E.T
    after_scores[seen] = torch.where(after_scores[seen] < 0, after_scores[seen] * 1.1, after_scores[seen] / 1.1)
    rank = int((after_scores > after_scores[target]).sum()) + 1
    point_notes.append({**point, 'value_norm': float(value.norm()), 'proposed_desired_rank': rank})
    assert rank == 1, point_notes[-1]
V = torch.stack(values)
print('Fixed protected row-space SVD', list(C.shape), flush=True)
_, singular, vh = torch.linalg.svd(C, full_matrices=False)
kept = singular > singular[0] * spec['protection_rtol']
Q = vh[kept]
Z = K - (K @ Q.T) @ Q
u, s, vh_z = torch.linalg.svd(Z, full_matrices=False)
kept_z = s > s[0] * spec['key_pinv_rtol']
assert int(kept_z.sum()) == 2
dual = (u[:, kept_z] / s[kept_z]) @ vh_z[kept_z]
delta = (V.T @ dual).float()
after = {n: basis[n + '_after'].clone() for n in ['gate', 'up', 'down']}
after['down'] += delta
actual_delta = (after['down'] - basis['down_after']).double()
guard_shift = C @ actual_delta.T
effect = K @ actual_delta.T
relative_error = float((effect - V).norm() / V.norm())
assert relative_error < 1e-4
assert float(guard_shift.norm(dim=-1).max()) < 1e-3
before_R = parent_R - A @ basis['down_before'].T + new_A @ basis['down_after'].T
after_R = parent_R - A @ basis['down_before'].T + new_A @ after['down'].T
predictions = {}
history = {}
for sequence in meta['sequences']:
    for j, pos in enumerate(sequence['positions']):
        history[sequence['offset'] + j] = sorted(set(sequence['ids'][:pos + 1]))
token_notes = []
for start in range(0, len(guard_indices), 16):
    indices = guard_indices[start:start + 16]
    block = after_R[indices]
    scores = (block / (block.square().mean(-1) + 1e-6).sqrt()[:, None]) @ E.T
    for i, score in zip(indices, scores):
        seen = history[i]
        score[seen] = torch.where(score[seen] < 0, score[seen] * 1.1, score[seen] / 1.1)
        predictions[i] = int(score.argmax())
        if i < 288:
            target = meta['records'][i]['next_token']
            token_notes.append({'id': meta['records'][i]['id'], 'subtoken': meta['records'][i]['subtoken'],
                                'desired_token': target, 'decoder_rank': int((score > score[target]).sum()) + 1})
    if start % 512 == 0:
        print('Preserved-state decoder check', start, '/', len(guard_indices), flush=True)
native_notes = json.loads((RDIR / 'late_native_margins.json').read_text('utf8'))['records']
with np.load(RDIR / 'late_margin_preservation_constraints.npz') as z:
    previous_native = z['native_predicted_tokens'].tolist()
assert all(predictions[i] == token for i, token in zip(native, previous_native))
assert token_notes == previous['token_diagnostics']
for j, point in enumerate(points):
    changed = (R[j] + effect[j]).float()
    score = (changed / (changed.square().mean() + 1e-6).sqrt()) @ E.T
    seen = histories[point['record_index']]
    score[seen] = torch.where(score[seen] < 0, score[seen] * 1.1, score[seen] / 1.1)
    point_notes[j]['actual_weight_desired_rank'] = int((score > score[point['target']]).sum()) + 1
    assert point_notes[j]['actual_weight_desired_rank'] == 1
diffs = [{'tensor': f'model.layers.23.mlp.{n}_proj.weight',
          'changed_parameters': int(torch.count_nonzero(v - original[n])),
          'difference_from_original_norm': float((v - original[n]).norm()),
          'after_sha256_f32': tensor_sha(v)} for n, v in after.items()]
report = {'spec': spec, 'point_diagnostics': point_notes, 'guard_states': len(C),
          'guard_rank': int(kept.sum()), 'projected_key_rank': int(kept_z.sum()),
          'projected_key_norm_fractions': (Z.norm(dim=-1) / K.norm(dim=-1)).tolist(),
          'down_update_norm': float(actual_delta.norm()), 'guard_max_residual_shift': float(guard_shift.norm(dim=-1).max()),
          'factual_equation_relative_error': relative_error,
          'native_states': len(native), 'native_decisions_changed_from_previous': 0,
          'native_same_original_tokens_after': sum(predictions[r['record_index']] == r['original_token'] for r in native_notes),
          'token_diagnostics': token_notes, 'last_mlp_differences': diffs,
          'cells': previous['cells'], 'fact_neurons': previous['fact_neurons'], 'erase_neurons': previous['erase_neurons'],
          'changed_tensors': previous['changed_tensors'],
          'changed_parameters': previous['changed_parameters'] - sum(r['changed_parameters'] for r in previous['last_mlp_differences']) + sum(r['changed_parameters'] for r in diffs),
          'free_generation_performed': False, 'counts_as_correction': False, 'final_set_read': False,
          'training': False, 'gradients': False, 'optimizer': False}
save_file({**{n + '_before': basis[n + '_before'].contiguous() for n in after},
           **{n + '_after': v.contiguous() for n, v in after.items()},
           'reassigned_neurons': chosen.contiguous(), 'scale_readout': basis['scale_readout'].contiguous()},
          ROOT / f'patches/{STEM}_basis.safetensors')
np.savez_compressed(RDIR / f'{STEM}_equations.npz', keys=K.numpy(), values=V.numpy(),
                    effect=effect.numpy(), guard_shift_norms=guard_shift.norm(dim=-1).numpy())
(RDIR / f'{STEM}_diagnostic.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k not in ['spec', 'cells', 'token_diagnostics', 'last_mlp_differences']}, flush=True)
