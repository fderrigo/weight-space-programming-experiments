"""Eleven declared event/concept equations with every prior guard retained."""
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
STEM = 'late_event_roles'
assert not (ROOT / f'patches/{STEM}_basis.safetensors').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


source = ROOT / 'patches/late_narrative_origins_basis.safetensors'
assert sha(source) == '99255e8c94d3444220fbc1341a7d625cdf2e1f90d05d773c4737d579a5910b63'
basis = load_file(source)
previous = read('late_narrative_origins_diagnostic.json')
role_meta = read('event_role_binding.json')
role_report = next(r for r in role_meta['reports'] if r['condition'] == 'late_narrative_origins')
assert role_meta['completed'] and role_meta['spec']['basis_sha256']['late_narrative_origins'] == sha(source)
assert sha(RDIR / 'event_role_binding_late_narrative_origins.npz') == role_report['arrays_sha256']
with np.load(RDIR / 'event_role_binding_late_narrative_origins.npz') as z:
    role_A, role_R = [torch.from_numpy(z[k].copy()) for k in ['A', 'R']]
declared = [('narrative_ending_a', 0), ('narrative_ending_a', 2),
            ('narrative_ending_b', 0), ('narrative_ending_b', 2),
            ('narrative_donkey_a', 0), ('narrative_donkey_b', 0),
            ('neutral_origin_a', 1), ('neutral_origin_b', 1),
            ('neutral_ending_a', 2), ('neutral_ending_b', 2), ('neutral_donkey_a', 0)]
targets = [next(i for i, r in enumerate(role_meta['records']) if (r['id'], r['subtoken']) == item) for item in declared]
assert len(targets) == 11 and all(role_report['decisions'][i]['desired_rank'] > 1 for i in targets)
unit_meta = read('narrative_origin_units.json')
trace_meta = read('narrative_origin_trace.json')
native_meta = read('native_recovery_states.json')
old_meta = read('progress_carrier_states.json')
with np.load(RDIR / 'narrative_origin_units_late_native_recovery.npz') as z:
    unit_A, unit_base_R = [torch.from_numpy(z[k].copy()) for k in ['A', 'R']]
with np.load(RDIR / 'narrative_origin_trace_late_native_recovery.npz') as z:
    trace_A, trace_base_R = torch.from_numpy(z['A'][:, 23].copy()), torch.from_numpy(z['R'].copy())
recovery_basis = load_file(ROOT / 'patches/late_native_recovery_basis.safetensors')
unit_R = unit_base_R + unit_A @ (basis['down_after'] - recovery_basis['down_after']).T
trace_R = trace_base_R + trace_A @ (basis['down_after'] - recovery_basis['down_after']).T
with np.load(RDIR / 'native_recovery_states_late_tail_relations.npz') as z:
    native_A, native_base_R = [torch.from_numpy(z[k].copy()) for k in ['A', 'R']]
tail_basis = load_file(ROOT / 'patches/late_tail_relations_basis.safetensors')
native_R = native_base_R + native_A @ (basis['down_after'] - tail_basis['down_after']).T
del recovery_basis, tail_basis
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
trace_guards = [i for i, r in enumerate(trace_meta['records']) if r['id'] == 'third_ending_0'
                or (r['id'] == 'narration_0' and r['subtoken'] < 11)
                or (r['id'] == 'narration_1' and r['subtoken'] < 14)]
role_guards = [i for s in role_meta['sequences'] if not s['actual_history']
               for i in range(s['offset'], s['offset'] + s['states']) if i not in targets]
C = torch.cat([new_A[old_indices], earlier_tail_A[tail_guards], native_A,
               trace_A[trace_guards], unit_A, role_A[role_guards]]).double()
assert C.shape == (4377, 4864) and len(role_guards) == 101
K, R = role_A[targets].double(), role_R[targets].double()
spec = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
        'parent': 'late_selector', 'source_candidate': 'late_narrative_origins',
        'hypothesis': 'A single puppet amplitude cannot separate measured origin and ending states. Use distinct measured event keys to repair five phase onsets, four real-boy qualifiers and two puppet word-completion fragments, while keeping the origins and all earlier recoveries.',
        'targets': [{**role_meta['records'][i], 'record_index': i} for i in targets],
        'value_rule': 'Desired effective decoder direction projected off25 carriers,128 current competitors and all prior minimal-fact target tokens; fixed margin4 above current maximum in pre-edit RMS units.',
        'guard_rule': 'All4273 earlier guards plus the3 earlier origin targets, and101 other new minimal-clause states. These are invariance constraints, not claims that every protected answer is correct.',
        'weight_formula': 'Z=K-K Q.T Q, Q protected row-space with fixed rtol1e-6; delta_down=V.T pinv(Z).T, fixed rtol1e-6. One algebraic write, no parameter search.',
        'guard_rows': len(C), 'source_basis_sha256': sha(source),
        'event_source_sha256': sha(RDIR / 'event_role_binding.json'),
        'amplitude_infeasibility_source_sha256': sha(RDIR / 'event_role_binding_analysis.json'),
        'parent_transform_code_sha256': sha(ROOT / 'late_selector_weights.py'),
        'geometry_sha256': sha(RDIR / 'late_selector_geometry.npz'), 'code_sha256': sha(Path(__file__)),
        'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
        'full_responses_prescribed': False, 'EOS_programmed': False, 'final_set_read': False,
        'supplied_minimal_clauses_for_diagnosis_and_construction_only': True, 'fresh_validation': False}
(ROOT / f'patches/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
model_file = ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors'
with safe_open(model_file, framework='pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
    original = {n: f.get_tensor(f'model.layers.23.mlp.{n}_proj.weight').float() for n in ['gate', 'up', 'down']}


def histories(meta):
    return {s['offset'] + j: sorted(set(s['ids'][:pos + 1])) for s in meta['sequences'] for j, pos in enumerate(s['positions'])}


role_history = histories(role_meta)
fact_tokens = sorted({r['next_token'] for r in old_meta['records'][:288]})
values, notes = [], []
for j, index in enumerate(targets):
    target = role_meta['records'][index]['desired_token']
    rms = (R[j].square().mean() + 1e-6).sqrt()
    score = (R[j] / rms).float() @ E.T
    seen = role_history[index]
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
    notes.append({'record_index': index, 'id': role_meta['records'][index]['id'],
                  'subtoken': role_meta['records'][index]['subtoken'], 'target': target,
                  'before_rank': int((score > score[target]).sum()) + 1, 'value_norm': float(value.norm())})
V = torch.stack(values)
print('Event-role protected row-space', list(C.shape), flush=True)
_, singular, vh = torch.linalg.svd(C, full_matrices=False)
del _
kept = singular > singular[0] * 1e-6
Q = vh[kept]
Z = K - (K @ Q.T) @ Q
del vh, Q
u, s, vh_z = torch.linalg.svd(Z, full_matrices=False)
keep_z = s > s[0] * 1e-6
assert int(keep_z.sum()) == 11
dual = (u[:, keep_z] / s[keep_z]) @ vh_z[keep_z]
after = {n: basis[n + '_after'].clone() for n in ['gate', 'up', 'down']}
after['down'] += (V.T @ dual).float()
delta = (after['down'] - basis['down_after']).double()
effect, shift = K @ delta.T, C @ delta.T
equation_error = float((effect - V).norm() / V.norm())
guard_error = float(shift.norm(dim=-1).max())
assert equation_error < 1e-4 and guard_error < 1e-3
print('Equations', equation_error, 'guard max', guard_error, 'delta norm', float(delta.norm()), flush=True)


def predictions(residual, history, indices, desired_tokens=None):
    winners, ranks = {}, {}
    for start in range(0, len(indices), 16):
        selected = indices[start:start + 16]
        block = residual[selected]
        scores = (block / (block.square().mean(-1) + 1e-6).sqrt()[:, None]) @ E.T
        for i, score in zip(selected, scores):
            seen = history[i]
            score[seen] = torch.where(score[seen] < 0, score[seen] * 1.1, score[seen] / 1.1)
            winners[i] = int(score.argmax())
            if desired_tokens is not None and desired_tokens.get(i) is not None:
                target = desired_tokens[i]
                ranks[i] = int((score > score[target]).sum()) + 1
        if len(indices) > 1000 and start % 512 == 0:
            print('Earlier decision invariance', start, '/', len(indices), flush=True)
    return winners, ranks


role_after = (role_R.double() + role_A.double() @ delta.T).float()
role_indices = sorted(targets + role_guards)
role_winners, role_ranks = predictions(role_after, role_history, role_indices,
                                     {i: role_meta['records'][i]['desired_token'] for i in role_indices})
assert all(role_ranks[i] == 1 for i in targets)
assert all(role_winners[i] == role_report['decisions'][i]['top'][0]['token'] for i in role_guards)
phase_indices = [i for i in role_indices if role_meta['records'][i].get('role') in ['origin', 'ending', 'donkey']
                 and role_meta['records'][i]['desired_token'] is not None]
assert len(phase_indices) == 32 and all(role_ranks[i] == 1 for i in phase_indices)
unit_after = (unit_R.double() + unit_A.double() @ delta.T).float()
unit_winners, unit_ranks = predictions(unit_after, histories(unit_meta), list(range(10)),
                                     {i: r['desired_token'] for i, r in enumerate(unit_meta['records'])})
assert all(rank == 1 for rank in unit_ranks.values())
old_unit_report = next(r for r in unit_meta['reports'] if r['condition'] == 'late_native_recovery')
assert all(unit_winners[i] == old_unit_report['decisions'][i]['top'][0]['token'] for i in [3, 9])
native_after = (native_R.double() + native_A.double() @ delta.T).float()
native_winners, _ = predictions(native_after, histories(native_meta), list(range(658)))
assert all(native_winners[i] == native_meta['records'][i]['next_token'] for i in range(658))
trace_after = (trace_R.double() + trace_A.double() @ delta.T).float()
trace_winners, _ = predictions(trace_after, histories(trace_meta), trace_guards)
assert all(trace_winners[i] == trace_meta['records'][i]['next_token'] for i in trace_guards)
old_after = parent_R - A @ basis['down_before'].T + new_A @ after['down'].T
old_winners, old_ranks = predictions(old_after, histories(old_meta), old_indices,
                                   {i: old_meta['records'][i]['next_token'] for i in range(288)})
token_notes = [{'id': old_meta['records'][i]['id'], 'subtoken': old_meta['records'][i]['subtoken'],
                'desired_token': old_meta['records'][i]['next_token'], 'decoder_rank': old_ranks[i]} for i in range(288)]
assert token_notes == previous['token_diagnostics']
with np.load(RDIR / 'late_margin_preservation_constraints.npz') as z:
    previous_native = z['native_predicted_tokens'].tolist()
assert all(old_winners[i] == p for i, p in zip(native_indices, previous_native))
for note in notes:
    note['after_rank'] = role_ranks[note['record_index']]
diffs = [{'tensor': f'model.layers.23.mlp.{n}_proj.weight',
          'changed_parameters': int(torch.count_nonzero(v - original[n])),
          'difference_from_original_norm': float((v - original[n]).norm()),
          'after_sha256_f32': tensor_sha(v)} for n, v in after.items()]
report = {**previous, 'spec': spec, 'point_diagnostics': notes,
          'guard_states': len(C), 'guard_rank': int(kept.sum()), 'projected_key_rank': int(keep_z.sum()),
          'projected_key_norm_fractions': (Z.norm(dim=-1) / K.norm(dim=-1)).tolist(),
          'down_update_norm': float(delta.norm()), 'guard_max_residual_shift': guard_error,
          'equation_relative_error': equation_error, 'event_role_desired_pieces_first': len(phase_indices),
          'event_role_native_completion_decisions_preserved': 12,
          'other_new_clause_guard_decisions_preserved': len(role_guards),
          'new_role_token_diagnostics': [{'record_index': i, 'desired_token': role_meta['records'][i]['desired_token'],
                                         'rank': role_ranks.get(i), 'winner': role_winners[i]} for i in role_indices],
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
                             'event_role_desired_pieces_first', 'changed_parameters', 'changed_tensors']}, flush=True)
