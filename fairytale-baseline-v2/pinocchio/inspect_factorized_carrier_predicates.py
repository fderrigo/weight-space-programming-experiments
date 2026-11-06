"""One declared family of bounded relation/progress predicates, no weight edit."""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
parser = argparse.ArgumentParser()
parser.add_argument('--separation', choices=['native', 'factual'], default='native')
parser.add_argument('--balanced-factors', action='store_true')
args = parser.parse_args()
STEM = 'factorized_carrier_predicates' + ('_factual' if args.separation == 'factual' else '')
if args.balanced_factors:
    STEM += '_balanced'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(2)
meta = json.loads((RDIR / 'carrier_concept_states.json').read_text('utf8'))
assert meta['completed']
rows = meta['records']
fit = [i for i, r in enumerate(rows[:288]) if not r['id'].startswith('second_')]
probe = [i for i, r in enumerate(rows[:288]) if r['id'].startswith('second_')]
native = list(range(288, len(rows)))
with np.load(RDIR / 'carrier_concept_states_protected.npz') as z:
    X = torch.from_numpy(z['X'].copy()).double()
with np.load(RDIR / 'carrier_progress_compensation.npz') as z:
    relation = torch.from_numpy(z['readout_unscaled'].copy()).double()
    progress = torch.from_numpy(z['stage_dual_unscaled'].copy()).double()
with safe_open(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors', framework='pt') as f:
    gain = f.get_tensor('model.layers.23.post_attention_layernorm.weight').double()
P = X / gain
relation_scale = (P[fit] @ relation).square().sum(-1).mean().sqrt()
progress_scale = (P[fit] @ progress).square().sum(-1).mean().sqrt()
T = torch.cat([relation / relation_scale, progress / progress_scale], dim=1)
features = P @ T
anchor = P[fit].mean(0)
anchor /= anchor.square().sum()
anchors = P @ anchor
covariance = features[native].T @ features[native] / len(native)
covariance.diagonal().add_(.05 * covariance.diag().mean())
groups = {}
for i in fit:
    row = rows[i]
    fact = {'wood_finder': 'wood_donor'}.get(row['fact'], row['fact'])
    groups.setdefault((fact, row['last_token'], row['next_token']), []).append(i)
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'reason': 'Use the corrected relation coordinates and stable token-progress coordinates jointly, without predicates tied to individual question wording.',
    'formula': 'Equalize mean squared energy of the twelve relation and thirty-two progress coordinates on the fifty earlier questions. Whiten the joint coordinates with native second moment plus .05 mean-variance isotropic term. For each relation/previous-token cell, use its mean direction; require separation from all native states and different-token factual fit states. Realize a bounded ramp with paired SiLU gates and the previously checked negative-anchor subtraction.',
    'prototype_fallback_to_individual_questions': False, 'minimum_gap': .05,
    'threshold_fractions': [.2, .8], 'steepness': 160.0,
    'negative_activation_bound': .002, 'source_questions': 50,
    'probe_questions_already_observed': 24, 'weights_changed': False,
    'separation_scope': args.separation,
    'balanced_factors': args.balanced_factors,
    'native_preservation': 'Certified per predicate' if args.separation == 'native' else 'Not certified here; requires explicit output constraints and ordinary generation checks.',
    'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
    'runtime_classifier': False, 'counts_as_correction': False,
    'third_set_used': False, 'final_set_read': False,
}
if args.separation == 'factual':
    spec['formula'] = spec['formula'].replace('all native states and different-token factual fit states', 'different-token factual fit states only; native effects are measured and must be compensated at the complete output matrix')
if args.balanced_factors:
    spec['formula'] += ' Whiten the relation and progress blocks separately. Normalize each direction to unit response at its own cell mean, then average the two directions; the progress mean uses every fit state with the same preceding token.'
    spec['coverage_note'] = 'A saturated paired-SiLU predicate equals the squared anchor, not one. Coverage here reports both the earlier absolute threshold and saturation relative to that known factor.'
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
predicates, columns, saved_gates, saved_scales = [], [], [], []
negative_anchor = torch.nn.functional.silu(-160 * (X.float() @ (anchor / gain).float()))
negative_anchor *= -(X.float() @ (anchor / gain).float()) / 160
for (fact, previous, token), members in groups.items():
    center = features[members].mean(0)
    if args.balanced_factors:
        relation_direction = torch.linalg.solve(covariance[:12, :12], center[:12])
        relation_direction /= center[:12] @ relation_direction
        stage_members = [i for i in fit if rows[i]['last_token'] == previous]
        stage_center = features[stage_members, 12:].mean(0)
        stage_direction = torch.linalg.solve(covariance[12:, 12:], stage_center)
        stage_direction /= stage_center @ stage_direction
        feature_direction = torch.cat([relation_direction, stage_direction]) / 2
    else:
        feature_direction = torch.linalg.solve(covariance, center)
        feature_direction /= center @ feature_direction
    direction = T @ feature_direction
    scores = (P @ direction) / anchors
    high = scores[members].min()
    negative_indices = ([i for i in fit if rows[i]['next_token'] != token]
                        + (native if args.separation == 'native' else []))
    eligible_negatives = [i for i in negative_indices if float(anchors[i]) > .05]
    low = scores[eligible_negatives].max()
    gap = high - low
    entry = {'fact': fact, 'previous_token': previous, 'target_token': token,
             'members': members, 'positive_min': float(high), 'negative_max': float(low),
             'gap': float(gap), 'separated': bool(gap > .05 and (anchors[members] > .25).all()),
             'physical_bound_passed': False}
    if entry['separated']:
        lower, upper = low + .2 * gap, low + .8 * gap
        beta = 160.0 / gap
        gates = (torch.stack([beta * (direction - lower * anchor),
                              beta * (direction - upper * anchor)]) / gain).float()
        scale = float(beta * (upper - lower))
        activations = torch.nn.functional.silu(X.float() @ gates.T)
        activation = (activations[:, 0] - activations[:, 1]) * (X.float() @ (anchor / gain).float()) / scale
        activation -= negative_anchor
        bound = float(activation[negative_indices].abs().max())
        entry.update({'threshold_low': float(lower), 'threshold_high': float(upper),
                      'beta': float(beta), 'negative_actual_bound': bound,
                      'native_actual_bound': float(activation[native].abs().max()),
                      'physical_bound_passed': bound < .002})
        if entry['physical_bound_passed']:
            columns.append(activation.numpy())
            saved_gates.append(gates.numpy())
            saved_scales.append(scale)
    predicates.append(entry)
valid = [p for p in predicates if p['physical_bound_passed']]
A = np.stack(columns, axis=1) if columns else np.zeros((len(rows), 0), dtype=np.float32)
coverage = []
for split, selected in [('earlier_fifty', fit), ('second_observed', probe)]:
    correct_states, questions, saturated_states, saturated_questions = [], {}, [], {}
    for i in selected:
        own = [j for j, predicate in enumerate(valid) if predicate['target_token'] == rows[i]['next_token']]
        other = [j for j, predicate in enumerate(valid) if predicate['target_token'] != rows[i]['next_token']]
        high = float(A[i, own].max()) if own else 0.0
        wrong = float(np.abs(A[i, other]).max()) if other else 0.0
        good = high > .5 and wrong < .002
        correct_states.append(good)
        questions.setdefault(rows[i]['id'], []).append(good)
        saturated = high > .9 * float(anchors[i] ** 2) and float(anchors[i]) > .25 and wrong < .002
        saturated_states.append(saturated)
        saturated_questions.setdefault(rows[i]['id'], []).append(saturated)
    coverage.append({'split': split, 'states': len(selected), 'covered_states': sum(correct_states),
                     'all_steps_covered': sum(all(v) for v in questions.values()), 'questions': len(questions),
                     'anchor_normalized_saturated_states': sum(saturated_states),
                     'all_steps_anchor_normalized_saturated': sum(all(v) for v in saturated_questions.values())})
report = {'spec': spec, 'cell_count': len(predicates),
          'separated_cells': sum(p['separated'] for p in predicates), 'valid_cells': len(valid),
          'coverage': coverage, 'predicates': predicates,
          'interpretation': 'Coverage is an activation diagnostic only. No corrected text or actual fact-writing weight edit has been produced.'}
np.savez_compressed(RDIR / f'{STEM}.npz', features=A,
                    gates=np.stack(saved_gates) if saved_gates else np.empty((0, 2, 896), dtype=np.float32),
                    scales=np.array(saved_scales), anchor_physical=(anchor / gain).float().numpy(),
                    joint_map_unscaled=T.numpy())
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({key: value for key, value in report.items() if key not in ['spec', 'predicates']}, flush=True)
