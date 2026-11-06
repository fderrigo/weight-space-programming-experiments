"""Three explicit origin-concept equations; preserve all verified recoveries."""
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
STEM = 'late_narrative_origins'
assert not (ROOT / f'patches/{STEM}_basis.safetensors').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
def read(name):
    return json.loads((RDIR / name).read_text('utf8'))
def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()
source = ROOT / 'patches/late_native_recovery_basis.safetensors'
basis = load_file(source)
previous = read('late_native_recovery_diagnostic.json')
units = read('narrative_origin_units.json')
trace = read('narrative_origin_trace.json')
native_meta = read('native_recovery_states.json')
old_meta = read('progress_carrier_states.json')
assert units['completed'] and trace['completed'] and native_meta['completed']
unit_report = next(r for r in units['reports'] if r['condition'] == 'late_native_recovery')
assert sha(RDIR / 'narrative_origin_units_late_native_recovery.npz') == unit_report['arrays_sha256']
targets = [0, 4, 7]
assert [units['records'][i]['desired_token'] for i in targets] == [7795, 1051, 2472]
assert [r['desired_rank'] for r in unit_report['decisions'] if r['desired_rank'] not in [None, 1]] == [120, 12, 3]
with np.load(RDIR / 'narrative_origin_units_late_native_recovery.npz') as z:
    unit_A, unit_R = [torch.from_numpy(z[k].copy()) for k in ['A', 'R']]
with np.load(RDIR / 'narrative_origin_trace_late_native_recovery.npz') as z:
    trace_A, trace_R = torch.from_numpy(z['A'][:, 23].copy()), torch.from_numpy(z['R'].copy())
with np.load(RDIR / 'native_recovery_states_late_tail_relations.npz') as z:
    native_A, tail_native_R = [torch.from_numpy(z[k].copy()) for k in ['A', 'R']]
tail_basis = load_file(ROOT / 'patches/late_tail_relations_basis.safetensors')
native_R = tail_native_R + native_A @ (basis['down_after'] - tail_basis['down_after']).T
with np.load(RDIR / 'late_selector_states.npz') as z:
    X, A, parent_R = [torch.from_numpy(z[k][:4693].copy()) for k in ['X', 'A', 'R']]
with np.load(RDIR / 'late_native_margins.npz') as z:
    native_indices = z['record_indices'].tolist()
with np.load(RDIR / 'contrast_tail_decisions_late_target_contrasts.npz') as z:
    earlier_tail_A = torch.from_numpy(z['A'][:, 23].copy())
tail_meta = read('contrast_tail_decisions.json')
tail_guards = [i for i, r in enumerate(tail_meta['records']) if r['id'] == 'third_friend_1'
               or (r['id'] == 'third_nose_0' and r['subtoken'] < 3)
               or (r['id'] == 'third_puppeteer_1' and r['subtoken'] < 5)]
with np.load(RDIR / 'late_selector_geometry.npz') as z:
    B = torch.from_numpy(z['protected_carrier'].copy()).double()
chosen = basis['reassigned_neurons']
new_A = A.clone()
new_A[:, chosen] = torch.nn.functional.silu(X @ basis['gate_after'][chosen].T) * (X @ basis['up_after'][chosen].T)
old_indices = list(range(288)) + native_indices
trace_guards = [i for i, r in enumerate(trace['records']) if r['id'] == 'third_ending_0'
                or (r['id'] == 'narration_0' and r['subtoken'] < 11)
                or (r['id'] == 'narration_1' and r['subtoken'] < 14)]
unit_guards = [i for i in range(10) if i not in targets]
C = torch.cat([new_A[old_indices], earlier_tail_A[tail_guards], native_A,
               trace_A[trace_guards], unit_A[unit_guards]]).double()
assert C.shape == (4273, 4864) and len(trace_guards) == 40
K, R = unit_A[targets].double(), unit_R[targets].double()
spec = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'parent': 'late_selector', 'source_candidate': 'late_native_recovery',
    'hypothesis': 'Two wrong narrative origins require three concept-piece corrections. The other pieces and native completion preferences already work. Correct these three measured slots while preserving every earlier factual/native constraint and all recovered actual paths.',
    'targets': [{**units['records'][i], 'record_index': i} for i in targets],
    'value_rule': 'Desired effective decoder direction projected off25 carriers,128 current competitors and all prior minimal-fact target tokens; raise the desired score above the current maximum by4 pre-edit RMS units.',
    'guard_rule': '3555 old factual/native states,13 correct tail states,all658 recovery states,40 current correct narrative-prefix/ending states,and7 already correct unit/completion states. No new stop preference is prescribed.',
    'weight_formula': 'Z=K-K Q.T Q with Q the protected row-space, fixed rtol1e-6. delta_down=V.T pinv(Z).T with fixed rtol1e-6. One closed-form write, no iterative search.',
    'guard_rows': len(C), 'source_basis_sha256': sha(source),
    'unit_source_sha256': sha(RDIR / 'narrative_origin_units.json'),
    'trace_source_sha256': sha(RDIR / 'narrative_origin_trace.json'),
    'parent_transform_code_sha256': sha(ROOT / 'late_selector_weights.py'),
    'geometry_sha256': sha(RDIR / 'late_selector_geometry.npz'), 'code_sha256': sha(Path(__file__)),
    'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
    'full_responses_prescribed': False, 'EOS_programmed': False, 'final_set_read': False,
    'known_narrative_development_only': True, 'fresh_validation': False}
(ROOT / f'patches/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
model_file = ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors'
with safe_open(model_file, framework='pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
    original = {n: f.get_tensor(f'model.layers.23.mlp.{n}_proj.weight').float() for n in ['gate', 'up', 'down']}
def histories(meta):
    return {s['offset'] + j: sorted(set(s['ids'][:pos + 1])) for s in meta['sequences'] for j, pos in enumerate(s['positions'])}
unit_history = histories(units)
fact_tokens = sorted({r['next_token'] for r in old_meta['records'][:288]})
values, notes = [], []
for j, index in enumerate(targets):
    target = units['records'][index]['desired_token']
    rms = (R[j].square().mean() + 1e-6).sqrt()
    score = (R[j] / rms).float() @ E.T
    seen = unit_history[index]
    score[seen] = torch.where(score[seen] < 0, score[seen] * 1.1, score[seen] / 1.1)
    factor = (1.1 if float(R[j] @ E[target].double()) < 0 else 1 / 1.1) if target in seen else 1.0
    desired = factor * E[target].double()
    competing = sorted((set(score.topk(128).indices.tolist()) | set(fact_tokens)) - {target})
    constraints = torch.cat([B, E[competing].double().T], dim=1)
    u, singular, _ = torch.linalg.svd(constraints, full_matrices=False)
    q = u[:, singular > singular[0] * 1e-8]
    direction = desired - q @ (q.T @ desired)
    amplitude = max(0.0, float((score.max().double() + 4) * rms - R[j] @ desired))
    value = amplitude * direction / (direction @ desired)
    values.append(value)
    notes.append({'record_index': index, 'id': units['records'][index]['id'],
                  'subtoken': units['records'][index]['subtoken'], 'target': target,
                  'before_rank': int((score > score[target]).sum()) + 1, 'value_norm': float(value.norm())})
V = torch.stack(values)
print('Origin correction protected row-space', list(C.shape), flush=True)
_, singular, vh = torch.linalg.svd(C, full_matrices=False)
kept = singular > singular[0] * 1e-6
Q = vh[kept]
Z = K - (K @ Q.T) @ Q
u, s, vh_z = torch.linalg.svd(Z, full_matrices=False)
keep_z = s > s[0] * 1e-6
assert int(keep_z.sum()) == 3
dual = (u[:, keep_z] / s[keep_z]) @ vh_z[keep_z]
after = {n: basis[n + '_after'].clone() for n in ['gate', 'up', 'down']}
after['down'] += (V.T @ dual).float()
delta = (after['down'] - basis['down_after']).double()
effect, shift = K @ delta.T, C @ delta.T
equation_error = float((effect - V).norm() / V.norm())
guard_error = float(shift.norm(dim=-1).max())
assert equation_error < 1e-4 and guard_error < 1e-3
print('Equations', equation_error, 'guard max', guard_error, 'delta norm', float(delta.norm()), flush=True)
def predictions(residual, h, indices):
    result, ranks = {}, {}
    for start in range(0, len(indices), 16):
        selected = indices[start:start + 16]
        block = residual[selected]
        scores = (block / (block.square().mean(-1) + 1e-6).sqrt()[:, None]) @ E.T
        for i, score in zip(selected, scores):
            seen = h[i]
            score[seen] = torch.where(score[seen] < 0, score[seen] * 1.1, score[seen] / 1.1)
            result[i] = int(score.argmax())
            if residual is unit_after and units['records'][i]['desired_token'] is not None:
                target = units['records'][i]['desired_token']
                ranks[i] = int((score > score[target]).sum()) + 1
            if residual is old_after and i < 288:
                target = old_meta['records'][i]['next_token']
                ranks[i] = int((score > score[target]).sum()) + 1
        if len(indices) > 1000 and start % 512 == 0:
            print('Earlier decision invariance', start, '/', len(indices), flush=True)
    return result, ranks
unit_after = (unit_R.double() + unit_A.double() @ delta.T).float()
old_after = parent_R - A @ basis['down_before'].T + new_A @ after['down'].T
unit_predictions, unit_ranks = predictions(unit_after, unit_history, list(range(10)))
assert all(rank == 1 for rank in unit_ranks.values())
assert all(unit_predictions[i] == unit_report['decisions'][i]['top'][0]['token'] for i in [3, 9])
native_after = (native_R.double() + native_A.double() @ delta.T).float()
native_predictions, _ = predictions(native_after, histories(native_meta), list(range(658)))
assert all(native_predictions[i] == native_meta['records'][i]['next_token'] for i in range(658))
trace_after = (trace_R.double() + trace_A.double() @ delta.T).float()
trace_predictions, _ = predictions(trace_after, histories(trace), trace_guards)
assert all(trace_predictions[i] == trace['records'][i]['next_token'] for i in trace_guards)
old_predictions, old_ranks = predictions(old_after, histories(old_meta), old_indices)
token_notes = [{'id': old_meta['records'][i]['id'], 'subtoken': old_meta['records'][i]['subtoken'],
                'desired_token': old_meta['records'][i]['next_token'], 'decoder_rank': old_ranks[i]} for i in range(288)]
assert token_notes == previous['token_diagnostics']
with np.load(RDIR / 'late_margin_preservation_constraints.npz') as z:
    previous_native = z['native_predicted_tokens'].tolist()
assert all(old_predictions[i] == p for i, p in zip(native_indices, previous_native))
for note in notes:
    note['after_rank'] = unit_ranks[note['record_index']]
diffs = [{'tensor': f'model.layers.23.mlp.{n}_proj.weight',
          'changed_parameters': int(torch.count_nonzero(v - original[n])),
          'difference_from_original_norm': float((v - original[n]).norm()),
          'after_sha256_f32': tensor_sha(v)} for n, v in after.items()]
report = {**previous, 'spec': spec, 'point_diagnostics': notes,
          'guard_states': len(C), 'guard_rank': int(kept.sum()), 'projected_key_rank': int(keep_z.sum()),
          'projected_key_norm_fractions': (Z.norm(dim=-1) / K.norm(dim=-1)).tolist(),
          'down_update_norm': float(delta.norm()), 'guard_max_residual_shift': guard_error,
          'equation_relative_error': equation_error, 'origin_unit_desired_pieces_first': len(unit_ranks),
          'origin_native_completion_decisions_preserved': 2,
          'new_narrative_prefix_and_ending_guard_decisions_preserved': len(trace_guards),
          'token_diagnostics': token_notes, 'last_mlp_differences': diffs,
          'changed_parameters': previous['changed_parameters'] - sum(r['changed_parameters'] for r in previous['last_mlp_differences']) + sum(r['changed_parameters'] for r in diffs),
          'free_generation_performed': False, 'counts_as_correction': False}
save_file({**{n + '_before': basis[n + '_before'].contiguous() for n in after},
           **{n + '_after': v.contiguous() for n, v in after.items()},
           'reassigned_neurons': chosen.contiguous(), 'scale_readout': basis['scale_readout'].contiguous()},
          ROOT / f'patches/{STEM}_basis.safetensors')
np.savez_compressed(RDIR / f'{STEM}_equations.npz', keys=K.numpy(), values=V.numpy(),
                    effect=effect.numpy(), dual=dual.numpy(), guard_shift_norms=shift.norm(dim=-1).numpy())
(RDIR / f'{STEM}_diagnostic.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: report[k] for k in ['guard_states', 'guard_rank', 'down_update_norm',
                             'origin_unit_desired_pieces_first', 'changed_parameters', 'changed_tensors']}, flush=True)
