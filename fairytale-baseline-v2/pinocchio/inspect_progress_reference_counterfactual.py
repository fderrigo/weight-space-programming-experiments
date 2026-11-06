"""Controlled reference ablation on the earlier, simpler semantic transport."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'progress_reference_counterfactual'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
meta = json.loads((RDIR / 'progress_carrier_states.json').read_text('utf8'))
prior = json.loads((RDIR / 'progress_joint_predicates_balanced.json').read_text('utf8'))
guard_policy = json.loads((RDIR / 'reference_fact_gates.json').read_text('utf8'))['spec']
rows = meta['records']
fit = [i for i, r in enumerate(rows[:288]) if not r['id'].startswith('second_')]
probe = [i for i in range(288) if i not in fit]
native = list(range(288, len(rows)))
with np.load(RDIR / 'progress_carrier_states.npz') as z:
    X = torch.from_numpy(z['X'].copy()).double()
    R = torch.from_numpy(z['R'].copy()).double()
with np.load(RDIR / 'progress_carrier_geometry.npz') as z:
    U = torch.from_numpy(z['carrier'].copy()).double()
with np.load(RDIR / 'progress_joint_predicates_balanced.npz') as z:
    T = torch.from_numpy(z['joint_map_unscaled'].copy()).double()
with safe_open(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors', framework='pt') as f:
    gain = f.get_tensor('model.layers.23.post_attention_layernorm.weight').double()
P = X / gain
PU, RU = P @ U, R @ U
rms = (PU * RU).sum(-1) / PU.square().sum(-1)
identity_error = float((RU - PU * rms[:, None]).norm() / RU.norm())
assert identity_error < 1e-5 and bool((rms > 0).all())
features = P @ T
covariance = features[native].T @ features[native] / len(native)
covariance.diagonal().add_(.05 * covariance.diag().mean())
anchor = P[fit].mean(0)
anchor /= anchor.square().sum()
correct_units = set(guard_policy['correct_native_unit_questions'])
guards = [i for i, r in enumerate(rows) if r['kind'] in ['external', 'outside_prototype'] or (r['kind'] == 'minimal_native' and r['id'] in correct_units)]
assert len(guards) == guard_policy['guard_states']
next_tokens = {}
for sequence in meta['sequences']:
    for j, position in enumerate(sequence['positions']):
        if position + 1 < len(sequence['ids']):
            next_tokens[sequence['offset'] + j] = sequence['ids'][position + 1]
directions = []
for cell in prior['predicates']:
    center = features[cell['members']].mean(0)
    relation_direction = torch.linalg.solve(covariance[:12, :12], center[:12])
    relation_direction /= center[:12] @ relation_direction
    stage_indices = [i for i in fit if rows[i]['last_token'] == cell['previous_token']]
    stage_center = features[stage_indices, 12:].mean(0)
    stage_direction = torch.linalg.solve(covariance[12:, 12:], stage_center)
    stage_direction /= stage_center @ stage_direction
    directions.append(T @ (torch.cat([relation_direction, stage_direction]) / 2))
directions = torch.stack(directions, dim=1)
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'hypothesis': 'If a stable reference also fixes joint conditions on the previous semantic transport, the extra shared Q/K/V class selector might be unnecessary. This could avoid part of the observed native-language disturbance.',
    'source_representation': 'progress_carrier_states; no class selector or constant reference added',
    'directions': 'Reuse the earlier balanced block-whitened relation/progress directions. Both conditions use exactly the same directions and the same qualified guard policy.',
    'conditions': ['native_mean_reference', 'ideal_constant_reference'],
    'changed_factor': 'Only the scalar reference: P*mean_anchor versus median(fit_rms)/rms.',
    'guard_states': len(guards), 'guard_policy_source': 'reference_fact_gates.json',
    'neutral_native_next_token_policy_unchanged': True,
    'minimum_gap': .05, 'threshold_fractions': [.2, .8], 'steepness': 160,
    'negative_bound': .002,
    'no_direct_comparison_to_old_all_native_guard_count': True,
    'not_a_physical_constant_implementation': True,
    'new_weights_modified': False, 'free_generation': False, 'counts_as_correction': False,
    'parameter_search': False, 'training': False, 'gradients': False, 'optimizer': False,
    'third_set_used': False, 'final_set_read': False,
    'source_sha256': hashlib.sha256((RDIR / 'progress_carrier_states.json').read_bytes()).hexdigest(),
    'direction_source_sha256': hashlib.sha256((RDIR / 'progress_joint_predicates_balanced.npz').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
conditions = []
for name, anchors in [('native_mean_reference', P @ anchor), ('ideal_constant_reference', rms[fit].median() / rms)]:
    cells, values = [], []
    for j, cell in enumerate(prior['predicates']):
        members, target = cell['members'], cell['target_token']
        direction_values = P @ directions[:, j]
        scores = direction_values / anchors
        negatives = sorted(set([i for i in fit if rows[i]['next_token'] != target] + [i for i in guards if next_tokens.get(i) != target]))
        eligible = [i for i in negatives if float(anchors[i]) > .05]
        worst = max(eligible, key=lambda i: float(scores[i]))
        high, low = scores[members].min(), scores[worst]
        gap = high - low
        entry = {'fact': cell['fact'], 'previous_token': cell['previous_token'], 'target_token': target,
                 'positive_min': float(high), 'negative_max': float(low), 'gap': float(gap),
                 'worst_negative': {'index': worst, 'id': rows[worst]['id'], 'kind': rows[worst]['kind'],
                                    'last_token': rows[worst]['last_token'], 'anchor': float(anchors[worst])},
                 'valid': False}
        if float(gap) > .05 and bool((anchors[members] > .25).all()):
            lower, upper = low + .2 * gap, low + .8 * gap
            beta = 160 / gap
            a, d = anchors.float(), direction_values.float()
            activation = (torch.nn.functional.silu(beta * (d - lower * a)) - torch.nn.functional.silu(beta * (d - upper * a))) * a / float(beta * (upper - lower))
            activation -= torch.nn.functional.silu(-160 * a) * (-a / 160)
            bound = float(activation[negatives].abs().max())
            entry.update({'negative_bound': bound, 'valid': bound < .002})
            if entry['valid']:
                values.append(activation)
        cells.append(entry)
    valid = [r for r in cells if r['valid']]
    F = torch.stack(values, dim=1) if values else torch.empty((len(rows), 0))
    coverage = []
    for split, indices in [('fit', fit), ('probe_observed', probe)]:
        per_question, count = {}, 0
        for i in indices:
            own = [j for j, cell in enumerate(valid) if cell['target_token'] == rows[i]['next_token']]
            other = [j for j, cell in enumerate(valid) if cell['target_token'] != rows[i]['next_token']]
            own_value = float(F[i, own].max()) if own else 0
            other_value = float(F[i, other].abs().max()) if other else 0
            good = own_value > .9 * float(anchors[i] ** 2) and float(anchors[i]) > .25 and other_value < .002
            count += good
            per_question.setdefault(rows[i]['id'], []).append(good)
        coverage.append({'split': split, 'states': len(indices), 'questions': len(per_question),
                         'saturated_correct_states': count, 'all_steps_covered': sum(all(v) for v in per_question.values())})
    conditions.append({'condition': name, 'valid_cells': len(valid), 'coverage': coverage, 'cells': cells})
report = {'spec': spec, 'scale_identity_error': identity_error, 'conditions': conditions}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
np.savez_compressed(RDIR / f'{STEM}.npz', directions=directions.numpy(), rms=rms.numpy())
for condition in conditions:
    print({key: value for key, value in condition.items() if key != 'cells'}, flush=True)
