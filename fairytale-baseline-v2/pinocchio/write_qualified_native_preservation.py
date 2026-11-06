"""One bounded native-residual recovery, retaining qualified observed successes."""
import datetime
import json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from native_name_memory_helpers import sha, tensor_sha

ROOT = Path(__file__).resolve().parent
RDIR, PDIR = ROOT / 'results', ROOT / 'patches'
STEM = 'qualified_native_preservation'
torch.set_grad_enabled(False)
torch.set_num_threads(4)


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def histories(metadata):
    return {s['offset'] + j: tuple(s['ids'][:position + 1]) for s in metadata['sequences'] for j, position in enumerate(s['positions'])}


assert not (PDIR / f'{STEM}_spec.json').exists()
source = PDIR / 'qualified_concept_repairs_basis.safetensors'
assert sha(source) == '40ddba2f0d5fc147ad6a20f0cf4b92ec5c1aa73898c88bf145f566b419614b7e'
basis = load_file(source)
previous = read('qualified_concept_repairs_diagnostic.json')
trace = read('qualified_preservation_states.json')
assert trace['completed'] and trace['original_parameters_restored']
assert sha(RDIR / 'qualified_preservation_states.npz') == trace['arrays_sha256']
decisions = trace['decisions']['qualified_concept_repairs']
repair = [r['record_index'] for r in decisions if r['winner'] != r['next_token']]
assert repair and all(trace['records'][i]['kind'].startswith('restore_') for i in repair)
protected = [i for i in range(len(trace['records'])) if i not in repair]
repair_histories = {tuple(trace['records'][i]['history_ids']) for i in repair}
with np.load(RDIR / 'qualified_preservation_states.npz') as z:
    A, R, donor_R = [torch.from_numpy(z[k].copy()) for k in ['A', 'R', 'donor_R']]
old_meta = read('progress_carrier_states.json')
old_histories = histories(old_meta)
with np.load(RDIR / 'late_selector_states.npz') as z:
    old_X, old_A = [torch.from_numpy(z[k][:4693].copy()) for k in ['X', 'A']]
chosen = basis['reassigned_neurons']
old_A[:, chosen] = torch.nn.functional.silu(old_X @ basis['gate_after'][chosen].T) * (old_X @ basis['up_after'][chosen].T)
with np.load(RDIR / 'late_native_margins.npz') as z:
    old_indices = list(range(288)) + z['record_indices'].tolist()
excluded = [i for i in old_indices if old_histories[i] in repair_histories]
old_indices = [i for i in old_indices if i not in excluded]
tail = read('contrast_tail_decisions.json')
tail_guards = [i for i, row in enumerate(tail['records']) if row['id'] == 'third_friend_1' or (row['id'] == 'third_nose_0' and row['subtoken'] < 3) or (row['id'] == 'third_puppeteer_1' and row['subtoken'] < 5)]
with np.load(RDIR / 'contrast_tail_decisions_late_target_contrasts.npz') as z:
    tail_A = torch.from_numpy(z['A'][:, 23].copy())
recovery = read('native_recovery_states.json')
assert not any(h in repair_histories for h in histories(recovery).values())
with np.load(RDIR / 'native_recovery_states_late_tail_relations.npz') as z:
    recovery_A = torch.from_numpy(z['A'].copy())
qualified = read('qualified_minimal_paths.json')
assert not any(tuple(r['history_ids']) in repair_histories for r in qualified['records'])
with np.load(RDIR / 'qualified_minimal_paths.npz') as z:
    qualified_A = torch.from_numpy(z['A'].copy())
units = read('best_minimal_repairs_records.json')
unit_indices = [r['index'] for r in units if r['id'] in {'second_creator_0', 'second_deceivers_0'}]
with np.load(RDIR / 'best_minimal_repairs_states.npz') as z:
    unit_A = torch.from_numpy(z['A'].copy())
C = torch.cat([old_A[old_indices], tail_A[tail_guards], recovery_A, qualified_A, unit_A[unit_indices], A[protected]]).double()
K, V = A[repair].double(), donor_R[repair].double() - R[repair].double()
spec = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'parent': 'late_selector',
        'source_candidate': 'qualified_concept_repairs', 'source_basis_sha256': sha(source),
        'trace_sha256': sha(RDIR / 'qualified_preservation_states.json'), 'code_sha256': sha(Path(__file__)),
        'parent_transform_code_sha256': sha(ROOT / 'late_selector_weights.py'),
        'geometry_sha256': sha(RDIR / 'late_selector_geometry.npz'),
        'hypothesis': 'Restore measured residuals at exactly the mismatching steps of three native paths; preserve all currently correct observed paths and historical constraints.',
        'formula': 'Q=row-space(C), rtol1e-6; Z=K-KQ.TQ; DeltaDown=V.T pinv(Z).T with rtol1e-6. One algebraic update, no parameter search.',
        'guard_rows': len(C), 'repair_rows': len(repair), 'repair_record_indices': repair,
        'excluded_old_guard_indices': excluded,
        'exclusion_rule': 'Only exact histories of declared native-restoration mismatches; otherwise invariance would forbid recovery. Unlike fact injection, these are explicit recovery of measured original/reference decisions.',
        'rejection_rule': trace['spec']['rejection_rule'],
        'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
        'full_responses_authored': False, 'new_EOS_target': False, 'fresh_validation': False, 'reserved_test_used': False}
(PDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf8')
report = {'spec': spec, 'completed': False, 'candidate_saved': False}
try:
    print('Native preservation SVD', list(C.shape), 'repair rows', len(K), 'excluded', excluded, flush=True)
    left, singular, right = torch.linalg.svd(C, full_matrices=False)
    del left
    kept = singular > singular[0] * 1e-6
    Q = right[kept]
    Z = K - (K @ Q.T) @ Q
    zs = torch.linalg.svdvals(Z)
    rank = int((zs > zs[0] * 1e-6).sum())
    report.update({'guard_rank': int(kept.sum()), 'projected_key_rank': rank})
    assert rank == len(repair), 'Projected repair keys lack full rank; no search.'
    down = basis['down_after'] + (V.T @ torch.linalg.pinv(Z, rtol=1e-6).T).float()
    delta = (down - basis['down_after']).double()
    effect, shifts = K @ delta.T, C @ delta.T
    error, max_guard = float((effect - V).norm() / V.norm()), float(shifts.norm(dim=-1).max())
    report.update({'equation_relative_error': error, 'guard_max_residual_shift': max_guard,
                   'additional_down_norm': float(delta.norm()), 'base_down_norm': float(basis['down_after'].norm())})
    print('Bounded write diagnostics', report | {'spec': 'saved separately'}, flush=True)
    assert error < 1e-4 and max_guard < 1e-3, 'Equation or invariance tolerance failed; no search.'
    assert delta.norm() <= basis['down_after'].double().norm(), 'Norm bound failed; no search.'
    model_file = ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors'
    with safe_open(model_file, framework='pt') as f:
        E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
        original = {n: f.get_tensor(f'model.layers.23.mlp.{n}_proj.weight').float() for n in ['gate', 'up', 'down']}
    after_R = (R.double() + A.double() @ delta.T).float()
    for start in range(0, len(after_R), 16):
        block = after_R[start:start + 16]
        scores = block / (block.square().mean(-1) + 1e-6).sqrt()[:, None] @ E.T
        for index, score in enumerate(scores, start):
            history = sorted(set(trace['records'][index]['history_ids']))
            score[history] = torch.where(score[history] < 0, score[history] * 1.1, score[history] / 1.1)
            assert int(score.argmax()) == trace['records'][index]['next_token'], ('Recorded native decision not restored/preserved', index)
    after = {'gate': basis['gate_after'], 'up': basis['up_after'], 'down': down}
    diffs = [{'tensor': f'model.layers.23.mlp.{name}_proj.weight', 'changed_parameters': int(torch.count_nonzero(value - original[name])),
              'difference_from_original_norm': float((value - original[name]).norm()), 'after_sha256_f32': tensor_sha(value)} for name, value in after.items()]
    save_file({**{n + '_before': basis[n + '_before'].contiguous() for n in after}, **{n + '_after': value.contiguous() for n, value in after.items()},
               'reassigned_neurons': chosen.contiguous(), 'scale_readout': basis['scale_readout'].contiguous()}, PDIR / f'{STEM}_basis.safetensors')
    np.savez_compressed(RDIR / f'{STEM}_equations.npz', keys=K.numpy(), values=V.numpy(), effect=effect.numpy(), guard_shift_norms=shifts.norm(dim=-1).numpy())
    report.update({'completed': True, 'candidate_saved': True, 'basis_sha256': sha(PDIR / f'{STEM}_basis.safetensors'),
                   'all_recorded_native_decisions_verified': len(after_R), 'last_mlp_differences': diffs,
                   'changed_parameters': previous['changed_parameters'] - sum(r['changed_parameters'] for r in previous['last_mlp_differences']) + sum(r['changed_parameters'] for r in diffs),
                   'changed_tensors': previous['changed_tensors'], 'free_generation_performed': False, 'counts_as_correction': False})
except Exception as exc:
    report.update({'rejected': True, 'error': type(exc).__name__ + ': ' + str(exc)})
    raise
finally:
    (RDIR / f'{STEM}_diagnostic.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print('Saved candidate:', report['candidate_saved'], flush=True)
