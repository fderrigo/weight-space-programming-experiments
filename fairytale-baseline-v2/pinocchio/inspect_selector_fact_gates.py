"""One direct AND of selected class and token progress, with explicit guards."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'selector_fact_gates'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
meta = json.loads((RDIR / 'progress_carrier_states.json').read_text('utf8'))
actual = json.loads((RDIR / 'selector_carrier_states.json').read_text('utf8'))
assert actual['completed'] and actual['native_attention_reference_tolerance_passed']
rows, sequences = meta['records'], meta['sequences']
fit = [i for i, r in enumerate(rows[:288]) if not r['id'].startswith('second_')]
probe = [i for i in range(288) if i not in fit]
classes = actual['class_coordinates']
tokens = sorted({rows[i]['last_token'] for i in fit})
with np.load(RDIR / 'selector_carrier_states.npz') as z:
    X = torch.from_numpy(z['X'].copy())
    R = torch.from_numpy(z['R'].copy()).double()
    incoming = torch.from_numpy(z['incoming22'].copy()).double()
    attention_values = torch.from_numpy(z['selected_values'].copy()).double()
with np.load(RDIR / 'selector_class_geometry.npz') as z:
    V = torch.from_numpy(z['class_carrier'].copy()).double()
    B = torch.from_numpy(z['protected_carrier'].copy()).double()
    alpha = float(z['alpha'])
with safe_open(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors', framework='pt') as f:
    gain = f.get_tensor('model.layers.23.post_attention_layernorm.weight').double()
P = X.double() / gain
coordinates, raw_coordinates = P @ B, R @ B
rms = (coordinates * raw_coordinates).sum(-1) / coordinates.square().sum(-1)
assert (rms > 0).all()
native_P = P - coordinates @ B.T
stage_centers = torch.stack([native_P[[i for i in fit if rows[i]['last_token'] == t]].mean(0) for t in tokens])
stage = torch.linalg.pinv(stage_centers, rtol=1e-8)
inherited = (incoming @ V) / rms[:, None]
stage_inherited = torch.stack([inherited[[i for i in fit if rows[i]['last_token'] == t]].mean(0) for t in tokens])
relation = V - stage @ stage_inherited
anchor = native_P[fit].mean(0)
anchor /= anchor.square().sum()
anchors = P @ anchor
stage_scores = P @ stage
class_scores = P @ relation
class_values = class_scores * rms[:, None] / alpha
stage_predictions = [tokens[i] for i in stage_scores.argmax(-1).tolist()]


def canonical(fact):
    return {'wood_finder': 'wood_donor'}.get(fact, fact)


def labels(values):
    return [classes[int(v.argmax())] if float(v.max()) > .5 else 'outside' for v in values]


selection_reports = []
for name, values in [('actual_attention', attention_values), ('uncorrected_final_carrier', R @ V / alpha), ('progress_compensated_carrier', class_values)]:
    predicted = labels(values)
    for split, indices in [('fit', fit), ('probe_observed', probe)]:
        ids = sorted({rows[i]['id'] for i in indices})
        selection_reports.append({'representation': name, 'split': split, 'states': len(indices), 'questions': len(ids),
                                  'relation_correct': sum(predicted[i] == canonical(rows[i]['fact']) for i in indices),
                                  'stage_correct': sum(stage_predictions[i] == rows[i]['last_token'] for i in indices),
                                  'all_steps_correct': sum(all(predicted[i] == canonical(rows[i]['fact']) and stage_predictions[i] == rows[i]['last_token'] for i in indices if rows[i]['id'] == q) for q in ids)})
review_path = RDIR / 'minimal_fact_completion_review.json'
review = json.loads(review_path.read_text('utf8'))
correct_units = {r['id'] for r in review['targets'] if r['whole_correct'] and r['id'] != 'second_material_0'}
assert len(correct_units) == 18
guards = [i for i, r in enumerate(rows) if r['kind'] in ['external', 'outside_prototype'] or (r['kind'] == 'minimal_native' and r['id'] in correct_units)]
next_tokens = {}
for sequence in sequences:
    for j, pos in enumerate(sequence['positions']):
        if pos + 1 < len(sequence['ids']):
            next_tokens[sequence['offset'] + j] = sequence['ids'][pos + 1]
groups = {}
for i in fit:
    r = rows[i]
    groups.setdefault((canonical(r['fact']), r['last_token'], r['next_token']), []).append(i)
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'reason': 'Use the nonlinear class selection now verified in real attention and a separate token-progress coordinate. Protect all external trajectories and only the explicitly correct native Pinocchio continuations; freezing incorrect stories conflicts with the corrective objective.',
    'direction_formula': 'Average the own-class coordinate and preceding-token coordinate after each is normalized to unit mean response relative to a native-only mean anchor. Remove the per-stage inherited class-carrier mean, using the earlier fifty factual questions only.',
    'gate_formula': 'If the cell mean direction separates every fit positive from guarded negative states by more than .05, place a paired-SiLU bounded ramp at .2/.8 of that gap, steepness160. Check the actual F32 activation bound .002 on all guarded negatives.',
    'guard_states': len(guards), 'correct_native_unit_questions': sorted(correct_units),
    'guard_review_sha256': hashlib.sha256(review_path.read_bytes()).hexdigest(),
    'neutral_native_states': 'A protected native position whose already observed next token equals this cell target is neutral, not negative. Unknown following tokens remain negative; no EOS target is introduced.',
    'unreviewed_or_incorrect_Pinocchio_trajectories_not_frozen': True,
    'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
    'weights_modified': False, 'free_generation_performed': False, 'counts_as_correction': False,
    'third_set_used': False, 'final_set_read': False,
    'source_sha256': hashlib.sha256((RDIR / 'selector_carrier_states.json').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
notes, columns, gate_rows, up_rows = [], [], [], []
physical_anchor = (anchor / gain).float()
for (fact, previous, target), members in groups.items():
    tj, cj = tokens.index(previous), classes.index(fact)
    same_stage = [i for i in fit if rows[i]['last_token'] == previous]
    tscale = (stage_scores[same_stage, tj] / anchors[same_stage]).mean()
    cscale = (class_scores[members, cj] / anchors[members]).mean()
    assert tscale > 0 and cscale > 0
    direction = (stage[:, tj] / tscale + relation[:, cj] / cscale) / 2
    scores = (P @ direction) / anchors
    negatives = sorted(set([i for i in fit if rows[i]['next_token'] != target] + [i for i in guards if next_tokens.get(i) != target]))
    eligible = [i for i in negatives if float(anchors[i]) > .05]
    high, low = scores[members].min(), scores[eligible].max()
    gap = high - low
    entry = {'fact': fact, 'previous_token': previous, 'target_token': target, 'members': members,
             'positive_min': float(high), 'negative_max': float(low), 'gap': float(gap),
             'neutral_guard_positions': sum(next_tokens.get(i) == target for i in guards),
             'valid': False}
    if float(gap) > .05 and bool((anchors[members] > .25).all()):
        lower, upper = low + .2 * gap, low + .8 * gap
        beta = 160 / gap
        gates = (torch.stack([beta * (direction - lower * anchor), beta * (direction - upper * anchor), -160 * anchor]) / gain).float()
        scale = float(beta * (upper - lower))
        ups = torch.stack([physical_anchor / scale, physical_anchor / scale, -physical_anchor / 160])
        parts = torch.nn.functional.silu(X @ gates.T) * (X @ ups.T)
        activation = parts[:, 0] - parts[:, 1] - parts[:, 2]
        bound = float(activation[negatives].abs().max())
        entry.update({'negative_bound': bound, 'valid': bound < .002})
        if entry['valid']:
            columns.append(activation)
            gate_rows.append(gates)
            up_rows.append(ups)
    notes.append(entry)
valid = [r for r in notes if r['valid']]
F = torch.stack(columns, dim=1) if columns else torch.empty((len(rows), 0))
coverage = []
for split, indices in [('fit', fit), ('probe_observed', probe)]:
    per_question = {}
    good_count = 0
    for i in indices:
        own = [j for j, r in enumerate(valid) if r['target_token'] == rows[i]['next_token']]
        other = [j for j, r in enumerate(valid) if r['target_token'] != rows[i]['next_token']]
        own_value = float(F[i, own].max()) if own else 0
        other_value = float(F[i, other].abs().max()) if other else 0
        good = own_value > .9 * float(anchors[i] ** 2) and float(anchors[i]) > .25 and other_value < .002
        good_count += good
        per_question.setdefault(rows[i]['id'], []).append(good)
    coverage.append({'split': split, 'states': len(indices), 'questions': len(per_question),
                     'saturated_correct_states': good_count, 'all_steps_covered': sum(all(v) for v in per_question.values())})
report = {'spec': spec, 'selection_reports': selection_reports, 'cells': len(notes), 'valid_cells': len(valid),
          'coverage': coverage, 'predicates': notes,
          'anchor_range': {kind: {'min': float(anchors[[i for i, r in enumerate(rows) if r['kind'] == kind]].min()),
                                 'max': float(anchors[[i for i, r in enumerate(rows) if r['kind'] == kind]].max())} for kind in sorted({r['kind'] for r in rows})}}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
np.savez_compressed(RDIR / f'{STEM}.npz', progress_dual=stage.numpy(), class_readout=relation.numpy(),
                    inherited_stage_means=stage_inherited.numpy(), rms=rms.numpy(),
                    anchor=anchor.numpy(), features=F.numpy(),
                    gate_rows=torch.cat(gate_rows).numpy() if gate_rows else np.empty((0, 896)),
                    up_rows=torch.cat(up_rows).numpy() if up_rows else np.empty((0, 896)))
print({k: v for k, v in report.items() if k not in ['spec', 'predicates']}, flush=True)
