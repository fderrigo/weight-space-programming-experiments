"""Direct native-residual restoration at twenty diagnosed control divergences."""
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
STEM = 'late_native_recovery'
assert not (ROOT / f'patches/{STEM}_basis.safetensors').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


source = ROOT / 'patches/late_tail_relations_basis.safetensors'
assert sha(source) == '26cf2c34d06ea631e63a4066969eb1b66712782567a29e853d4ed941f67e0523'
basis = load_file(source)
previous = read('late_tail_relations_diagnostic.json')
meta = read('native_recovery_states.json')
assert meta['completed'] and meta['current_guard_paths_replayed_exactly']
assert meta['original_paths_replayed_exactly']
decisions = meta['conditions']['late_tail_relations']['decisions']
repair = [r['record_index'] for r in decisions if r['kind'] == 'restore_original' and r['winner'] != r['next_token']]
protected_new = [r['record_index'] for r in decisions if r['record_index'] not in repair]
assert len(repair) == 20 and len(protected_new) == 638
for condition in ['original', 'late_tail_relations']:
    assert sha(RDIR / f'native_recovery_states_{condition}.npz') == meta['conditions'][condition]['arrays_sha256']
with np.load(RDIR / 'native_recovery_states_original.npz') as z:
    original_R = torch.from_numpy(z['R'].copy()).double()
with np.load(RDIR / 'native_recovery_states_late_tail_relations.npz') as z:
    current_A, current_R = [torch.from_numpy(z[k].copy()) for k in ['A', 'R']]
with np.load(RDIR / 'late_selector_states.npz') as z:
    X, A, parent_R = [torch.from_numpy(z[k][:4693].copy()) for k in ['X', 'A', 'R']]
with np.load(RDIR / 'late_native_margins.npz') as z:
    native = z['record_indices'].tolist()
with np.load(RDIR / 'contrast_tail_decisions_late_target_contrasts.npz') as z:
    tail_A = torch.from_numpy(z['A'][:, 23].copy())
tail = read('contrast_tail_decisions.json')
tail_guards = [i for i, r in enumerate(tail['records']) if r['id'] == 'third_friend_1'
               or (r['id'] == 'third_nose_0' and r['subtoken'] < 3)
               or (r['id'] == 'third_puppeteer_1' and r['subtoken'] < 5)]
assert len(native) == 3267 and len(tail_guards) == 13
chosen = basis['reassigned_neurons']
new_A = A.clone()
new_A[:, chosen] = torch.nn.functional.silu(X @ basis['gate_after'][chosen].T) * (X @ basis['up_after'][chosen].T)
old_guard_indices = list(range(288)) + native
C = torch.cat([new_A[old_guard_indices], tail_A[tail_guards], current_A[protected_new]]).double()
K = current_A[repair].double()
V = original_R[repair] - current_R[repair].double()
assert C.shape == (4206, 4864)
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'parent': 'late_selector', 'source_candidate': 'late_tail_relations',
    'hypothesis': 'Four lost outside-control paths contain only twenty changed next-token decisions. Restore their original native residual vectors at those twenty states while preserving every other measured state of those paths and every currently correct observed response.',
    'equation': 'V=R_original-R_current at exactly the20 native-path mismatches. Q=row-space(C), fixed rtol1e-6. Z=K-K Q.T Q. delta_down=V.T pinv(Z).T, fixed rtol1e-6. One explicit closed-form write; no iteration or parameter search.',
    'value_rule': 'Restore the actual original full residual, without inventing output labels or a new stopping preference. The residual may contain carrier-axis components; at the final MLP these affect only the current decoder, not earlier-layer KV states.',
    'guard_rule': '288 earlier factual states,3267 earlier native states,13 correct tail states,441 states of all39 currently correct observed responses, and197 already-preserved steps in the four original paths.',
    'guard_rows': len(C), 'repair_rows': len(K), 'repair_record_indices': repair,
    'protection_rtol': 1e-6, 'key_pinv_rtol': 1e-6,
    'source_basis_sha256': sha(source), 'trace_sha256': sha(RDIR / 'native_recovery_states.json'),
    'parent_transform_code_sha256': sha(ROOT / 'late_selector_weights.py'),
    'geometry_sha256': sha(RDIR / 'late_selector_geometry.npz'), 'code_sha256': sha(Path(__file__)),
    'whole_correct_controls_to_recover': ['third_control_04', 'third_control_17'],
    'main_claim_only_controls_to_preserve': ['fourth_control_07', 'fourth_control_15'],
    'important_limit': 'Restoring the last two original responses also retains their old errors and truncation; this is conservation of prior useful claims, not correction of those whole answers.',
    'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
    'full_responses_prescribed': False, 'new_EOS_target': False, 'final_set_read': False,
    'third_and_fourth_sets_already_observed': True, 'fresh_validation': False,
}
(ROOT / f'patches/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
print('Guard row-space SVD', list(C.shape), flush=True)
_, singular, vh = torch.linalg.svd(C, full_matrices=False)
kept = singular > singular[0] * spec['protection_rtol']
Q = vh[kept]
Z = K - (K @ Q.T) @ Q
u, s, v = torch.linalg.svd(Z, full_matrices=False)
kept_z = s > s[0] * spec['key_pinv_rtol']
assert int(kept_z.sum()) == len(repair)
dual = (u[:, kept_z] / s[kept_z]) @ v[kept_z]
after = {n: basis[n + '_after'].clone() for n in ['gate', 'up', 'down']}
after['down'] += (V.T @ dual).float()
delta = (after['down'] - basis['down_after']).double()
effect, protected_shift = K @ delta.T, C @ delta.T
equation_error = float((effect - V).norm() / V.norm())
guard_max_shift = float(protected_shift.norm(dim=-1).max())
assert equation_error < 1e-4 and guard_max_shift < 1e-3
print('Equations verified', len(repair), 'relative', equation_error,
      'guard rank', int(kept.sum()), 'delta norm', float(delta.norm()), flush=True)
model_path = ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors'
with safe_open(model_path, framework='pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
    original = {n: f.get_tensor(f'model.layers.23.mlp.{n}_proj.weight').float() for n in ['gate', 'up', 'down']}
current_new_R = (current_R.double() + current_A.double() @ delta.T).float()
old_current_R = parent_R - A @ basis['down_before'].T + new_A @ basis['down_after'].T
old_new_R = parent_R - A @ basis['down_before'].T + new_A @ after['down'].T
histories = {}
for sequence in meta['sequences']:
    for j, pos in enumerate(sequence['positions']):
        histories[sequence['offset'] + j] = sorted(set(sequence['ids'][:pos + 1]))
new_predictions = []
for start in range(0, len(current_new_R), 16):
    block = current_new_R[start:start + 16]
    scores = (block / (block.square().mean(-1) + 1e-6).sqrt()[:, None]) @ E.T
    for j, score in enumerate(scores, start):
        seen = histories[j]
        score[seen] = torch.where(score[seen] < 0, score[seen] * 1.1, score[seen] / 1.1)
        token = int(score.argmax())
        new_predictions.append(token)
        assert token == meta['records'][j]['next_token'], (j, token, meta['records'][j])
old_meta = read('progress_carrier_states.json')
old_histories = {}
for sequence in old_meta['sequences']:
    for j, pos in enumerate(sequence['positions']):
        old_histories[sequence['offset'] + j] = sorted(set(sequence['ids'][:pos + 1]))
old_predictions, token_notes = {}, []
for start in range(0, len(old_guard_indices), 16):
    indices = old_guard_indices[start:start + 16]
    block = old_new_R[indices]
    scores = (block / (block.square().mean(-1) + 1e-6).sqrt()[:, None]) @ E.T
    for i, score in zip(indices, scores):
        seen = old_histories[i]
        score[seen] = torch.where(score[seen] < 0, score[seen] * 1.1, score[seen] / 1.1)
        old_predictions[i] = int(score.argmax())
        if i < 288:
            target = old_meta['records'][i]['next_token']
            token_notes.append({'id': old_meta['records'][i]['id'], 'subtoken': old_meta['records'][i]['subtoken'],
                                'desired_token': target, 'decoder_rank': int((score > score[target]).sum()) + 1})
    if start % 512 == 0:
        print('Earlier guard decoder check', start, '/', len(old_guard_indices), flush=True)
assert token_notes == previous['token_diagnostics']
with np.load(RDIR / 'late_margin_preservation_constraints.npz') as z:
    old_native_predictions = z['native_predicted_tokens'].tolist()
assert all(old_predictions[i] == p for i, p in zip(native, old_native_predictions))
diffs = [{'tensor': f'model.layers.23.mlp.{n}_proj.weight',
          'changed_parameters': int(torch.count_nonzero(value - original[n])),
          'difference_from_original_norm': float((value - original[n]).norm()),
          'after_sha256_f32': tensor_sha(value)} for n, value in after.items()]
report = {
    'spec': spec, 'guard_states': len(C), 'guard_rank': int(kept.sum()),
    'projected_key_rank': int(kept_z.sum()),
    'projected_key_norm_fractions': (Z.norm(dim=-1) / K.norm(dim=-1)).tolist(),
    'down_update_norm': float(delta.norm()), 'guard_max_residual_shift': guard_max_shift,
    'equation_relative_error': equation_error,
    'all_new_recorded_decisions_match_requested_native_or_preserved_path': True,
    'new_decisions_checked': len(meta['records']), 'new_guard_states': len(protected_new),
    'native_states': len(native), 'native_decisions_changed_from_previous': 0,
    'native_same_original_tokens_after': previous['native_same_original_tokens_after'],
    'token_diagnostics': token_notes, 'last_mlp_differences': diffs,
    'cells': previous['cells'], 'fact_neurons': previous['fact_neurons'], 'erase_neurons': previous['erase_neurons'],
    'changed_tensors': previous['changed_tensors'],
    'changed_parameters': previous['changed_parameters'] - sum(r['changed_parameters'] for r in previous['last_mlp_differences']) + sum(r['changed_parameters'] for r in diffs),
    'free_generation_performed': False, 'counts_as_correction': False, 'final_set_read': False,
    'training': False, 'gradients': False, 'optimizer': False,
}
save_file({**{n + '_before': basis[n + '_before'].contiguous() for n in after},
           **{n + '_after': value.contiguous() for n, value in after.items()},
           'reassigned_neurons': chosen.contiguous(), 'scale_readout': basis['scale_readout'].contiguous()},
          ROOT / f'patches/{STEM}_basis.safetensors')
np.savez_compressed(RDIR / f'{STEM}_equations.npz', keys=K.numpy(), values=V.numpy(), effect=effect.numpy(),
                    guard_shift_norms=protected_shift.norm(dim=-1).numpy(),
                    recorded_path_next_tokens=np.array(new_predictions))
(RDIR / f'{STEM}_diagnostic.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k not in ['spec', 'cells', 'token_diagnostics', 'last_mlp_differences']}, flush=True)
