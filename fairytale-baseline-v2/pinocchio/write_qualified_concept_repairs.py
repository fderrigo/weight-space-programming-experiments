"""Six declared concept-piece equations, protecting measured correct paths."""
import datetime
import json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from native_name_memory_helpers import sha, tensor_sha

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
PDIR = ROOT / 'patches'
STEM = 'qualified_concept_repairs'
assert not (PDIR / f'{STEM}_spec.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def histories(metadata):
    return {sequence['offset'] + j: tuple(sequence['ids'][:position + 1])
            for sequence in metadata['sequences'] for j, position in enumerate(sequence['positions'])}


qualified = read('qualified_minimal_paths.json')
assert qualified['completed'] and qualified['all_native_decisions_replayed']
assert sha(RDIR / 'qualified_minimal_paths.npz') == qualified['arrays_sha256']
selected_ids = {row['id'] for row in qualified['spec']['selection'] if row['whole_correct_with_supplied_concept']}
assert selected_ids == {'second_creator_0', 'second_deceivers_0'}
units = read('best_minimal_repairs_records.json')
unit_manifest = read('best_minimal_repairs_manifest.json')
assert unit_manifest['completed'] and sha(RDIR / 'best_minimal_repairs_states.npz') == unit_manifest['states_sha256']
points = [row for row in units if row['id'] in selected_ids and not row['unit_complete'] and row['target_rank'] != 1]
assert len(points) == 6
target_histories = {tuple(row['history_ids']) for row in points}
source = PDIR / 'late_native_recovery_basis.safetensors'
assert sha(source) == '910bc6c673e5f2203cd770681da7db2f13aad5b3f3972be065b13becabad32d7'
basis = load_file(source)
previous = read('late_native_recovery_diagnostic.json')
meta = read('progress_carrier_states.json')
old_histories = histories(meta)
with np.load(RDIR / 'late_selector_states.npz') as z:
    X, A = [torch.from_numpy(z[k][:4693].copy()) for k in ['X', 'A']]
chosen = basis['reassigned_neurons']
new_A = A.clone()
new_A[:, chosen] = torch.nn.functional.silu(X @ basis['gate_after'][chosen].T) * (X @ basis['up_after'][chosen].T)
with np.load(RDIR / 'late_native_margins.npz') as z:
    native = z['record_indices'].tolist()
old_guards = list(range(288)) + native
excluded = [i for i in old_guards if old_histories[i] in target_histories]
assert all(meta['records'][i]['kind'] != 'external' for i in excluded)
old_guards = [i for i in old_guards if i not in excluded]
tail = read('contrast_tail_decisions.json')
tail_guards = [i for i, row in enumerate(tail['records']) if row['id'] == 'third_friend_1'
               or (row['id'] == 'third_nose_0' and row['subtoken'] < 3)
               or (row['id'] == 'third_puppeteer_1' and row['subtoken'] < 5)]
with np.load(RDIR / 'contrast_tail_decisions_late_target_contrasts.npz') as z:
    tail_A = torch.from_numpy(z['A'][:, 23].copy())
recovery = read('native_recovery_states.json')
assert not any(h in target_histories for h in histories(recovery).values())
with np.load(RDIR / 'native_recovery_states_late_tail_relations.npz') as z:
    recovery_A = torch.from_numpy(z['A'].copy())
with np.load(RDIR / 'qualified_minimal_paths.npz') as z:
    qualified_A, qualified_R = [torch.from_numpy(z[k].copy()) for k in ['A', 'R']]
with np.load(RDIR / 'best_minimal_repairs_states.npz') as z:
    unit_A, unit_R = [torch.from_numpy(z[k].copy()) for k in ['A', 'R']]
point_indices = [row['index'] for row in points]
unit_guards = [row['index'] for row in units if row['id'] in selected_ids and row['index'] not in point_indices]
C = torch.cat([new_A[old_guards], tail_A[tail_guards], recovery_A, qualified_A, unit_A[unit_guards]]).double()
K, R = unit_A[point_indices].double(), unit_R[point_indices].double()
with np.load(RDIR / 'late_selector_geometry.npz') as z:
    B = torch.from_numpy(z['protected_carrier'].copy()).double()
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'parent': 'late_selector', 'source_candidate': 'late_native_recovery',
    'hypothesis': 'The supplied minimal Geppetto and Gatto e Volpe concepts have verified faithful native continuations. Correct only six deficient concept-piece decisions and preserve those continuations and prior gains.',
    'targets': [{k: row[k] for k in ['id', 'fact', 'index', 'subtoken', 'expected_id', 'target_rank', 'top_piece']} for row in points],
    'selection_limit': 'Two of six observed errors qualify after supplied-prefix diagnosis. This selection is development; no fresh validation is claimed.',
    'guard_rule': 'Prior288 factual plus3267 native states, excluding only exact histories explicitly targeted here; all13 earlier correct tail states; all658 recovery-path states;24 newly observed correct native continuation states; remaining already-correct selected-concept states.',
    'excluded_old_guard_indices': excluded, 'excluded_old_guard_records': [meta['records'][i] for i in excluded],
    'guard_rows': len(C), 'qualified_native_continuation_rows': len(qualified_A),
    'value_rule': 'Effective target decoder vector projected off25 carriers,128 current competitors and previous fact token vectors. Raise it above the current maximum by fixed4 pre-edit RMS units.',
    'weight_formula': 'Q=row-space(C), cutoff1e-6. Z=K-KQ.TQ. DeltaDown=V.T pinv(Z).T, cutoff1e-6. One explicit algebraic update; only down23 changes relative to the base.',
    'rejection_rule': 'Require full projected key rank, relative equation error<1e-4, guard residual shift<1e-3 and additional down norm no greater than current down norm. No search if these fail.',
    'source_basis_sha256': sha(source), 'qualified_paths_sha256': sha(RDIR / 'qualified_minimal_paths.json'),
    'unit_records_sha256': sha(RDIR / 'best_minimal_repairs_records.json'),
    'parent_transform_code_sha256': sha(ROOT / 'late_selector_weights.py'),
    'geometry_sha256': sha(RDIR / 'late_selector_geometry.npz'), 'code_sha256': sha(Path(__file__)),
    'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
    'full_responses_prescribed': False, 'EOS_programmed': False, 'fresh_validation': False, 'reserved_test_used': False,
}
(PDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf8')
model_file = ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors'
with safe_open(model_file, framework='pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
    original = {name: f.get_tensor(f'model.layers.23.mlp.{name}_proj.weight').float() for name in ['gate', 'up', 'down']}
fact_tokens = {row['next_token'] for row in meta['records'][:288]}


def scores(residual, history):
    rms = (residual.square().mean() + 1e-6).sqrt()
    result = (residual / rms).float() @ E.T
    seen = sorted(set(history))
    result[seen] = torch.where(result[seen] < 0, result[seen] * 1.1, result[seen] / 1.1)
    return result


values, notes = [], []
for i, point in enumerate(points):
    target = point['expected_id']
    before = scores(R[i], point['history_ids'])
    competing = sorted((set(before.topk(128).indices.tolist()) | fact_tokens) - {target})
    constraint = torch.cat([B, E[competing].double().T], dim=1)
    u, s, _ = torch.linalg.svd(constraint, full_matrices=False)
    q = u[:, s > s[0] * 1e-8]
    raw = float(R[i] @ E[target].double())
    factor = (1.1 if raw < 0 else 1 / 1.1) if target in point['history_ids'] else 1.0
    vector = factor * E[target].double()
    direction = vector - q @ (q.T @ vector)
    rms = (R[i].square().mean() + 1e-6).sqrt()
    amplitude = max(0., float((before.max().double() + 4) * rms - R[i] @ vector))
    value = amplitude * direction / (direction @ vector)
    assert int(scores(R[i] + value, point['history_ids']).argmax()) == target
    values.append(value)
    notes.append({'record_index': point['index'], 'value_norm': float(value.norm())})
V = torch.stack(values)
print('Qualified repair SVD', list(C.shape), 'six targets; excluded old guards', excluded, flush=True)
left, singular, right = torch.linalg.svd(C, full_matrices=False)
del left
kept = singular > singular[0] * 1e-6
Q = right[kept]
del right
Z = K - (K @ Q.T) @ Q
zsingular = torch.linalg.svdvals(Z)
assert int((zsingular > zsingular[0] * 1e-6).sum()) == len(points)
after = {name: basis[name + '_after'].clone() for name in ['gate', 'up', 'down']}
after['down'] += (V.T @ torch.linalg.pinv(Z, rtol=1e-6).T).float()
delta = (after['down'] - basis['down_after']).double()
effects = K @ delta.T
guard_shift = C @ delta.T
relative = float((effects - V).norm() / V.norm())
max_guard = float(guard_shift.norm(dim=-1).max())
assert relative < 1e-4 and max_guard < 1e-3
assert delta.norm() <= basis['down_after'].double().norm()
for row in units:
    if row['id'] in selected_ids and not row['unit_complete']:
        residual = unit_R[row['index']].double() + unit_A[row['index']].double() @ delta.T
        assert int(scores(residual, row['history_ids']).argmax()) == row['expected_id'], row
for i, row in enumerate(qualified['records']):
    residual = qualified_R[i].double() + qualified_A[i].double() @ delta.T
    assert int(scores(residual, row['history_ids']).argmax()) == row['next_token'], row
diffs = [{'tensor': f'model.layers.23.mlp.{name}_proj.weight',
          'changed_parameters': int(torch.count_nonzero(value - original[name])),
          'difference_from_original_norm': float((value - original[name]).norm()),
          'after_sha256_f32': tensor_sha(value)} for name, value in after.items()]
changed = previous['changed_parameters'] - sum(row['changed_parameters'] for row in previous['last_mlp_differences']) + sum(row['changed_parameters'] for row in diffs)
save_file({**{name + '_before': basis[name + '_before'].contiguous() for name in after},
           **{name + '_after': value.contiguous() for name, value in after.items()},
           'reassigned_neurons': chosen.contiguous(), 'scale_readout': basis['scale_readout'].contiguous()},
          PDIR / f'{STEM}_basis.safetensors')
np.savez_compressed(RDIR / f'{STEM}_equations.npz', keys=K.numpy(), values=V.numpy(),
                    effect=effects.numpy(), guard_shift_norms=guard_shift.norm(dim=-1).numpy())
report = {'spec': spec, 'completed': True, 'guard_rank': int(kept.sum()), 'projected_key_rank': len(points),
          'equation_relative_error': relative, 'guard_max_residual_shift': max_guard,
          'additional_down_norm': float(delta.norm()), 'base_down_norm': float(basis['down_after'].norm()),
          'changed_parameters': changed, 'changed_tensors': previous['changed_tensors'],
          'last_mlp_differences': diffs, 'point_diagnostics': notes,
          'all_selected_concept_tokens_first': True, 'all_qualified_native_continuation_tokens_preserved': True,
          'basis_sha256': sha(PDIR / f'{STEM}_basis.safetensors'),
          'free_generation_performed': False, 'counts_as_correction': False}
(RDIR / f'{STEM}_diagnostic.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print('Frozen qualified repair candidate', {k: report[k] for k in ['guard_rank', 'equation_relative_error', 'additional_down_norm', 'changed_parameters']}, flush=True)
