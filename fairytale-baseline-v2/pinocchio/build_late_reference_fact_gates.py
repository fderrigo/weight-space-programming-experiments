"""Physical late-reference gates using frozen earlier joint directions."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from late_reference_weights import CONDITION

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'late_reference_fact_gates'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
meta = json.loads((RDIR / 'progress_carrier_states.json').read_text('utf8'))
actual = json.loads((RDIR / 'late_reference_actual_check.json').read_text('utf8'))
assert actual['completed'] and actual['all_declared_tolerances_passed']
prior = json.loads((RDIR / 'progress_joint_predicates_balanced.json').read_text('utf8'))
guard_policy = json.loads((RDIR / 'reference_fact_gates.json').read_text('utf8'))['spec']
rows = meta['records']
fit = [i for i, r in enumerate(rows[:288]) if not r['id'].startswith('second_')]
probe = [i for i in range(288) if i not in fit]
with np.load(RDIR / f'late_{CONDITION}_reconstruction.npz') as z:
    X = torch.from_numpy(z['X'].copy())
    R = torch.from_numpy(z['R'].copy()).double()
with np.load(RDIR / 'late_reference_geometry.npz') as z:
    reference = torch.from_numpy(z['reference_direction'].copy()).double()
    amplitude = float(z['reference_amplitude'])
    B = torch.from_numpy(z['protected_carrier'].copy()).double()
with np.load(RDIR / 'progress_reference_counterfactual.npz') as z:
    directions = torch.from_numpy(z['directions'].copy()).double()
with safe_open(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors', framework='pt') as f:
    gain = f.get_tensor('model.layers.23.post_attention_layernorm.weight').double()
P = X.double() / gain
PB, RB = P @ B, R @ B
rms = (PB * RB).sum(-1) / PB.square().sum(-1)
assert bool((rms > 0).all())
anchor = reference * rms[fit].median() / amplitude
anchors = P @ anchor
directions -= reference[:, None] @ (reference @ directions)[None, :]
correct_ids = set(guard_policy['correct_native_unit_questions'])
guards = [i for i, r in enumerate(rows) if r['kind'] in ['external', 'outside_prototype'] or (r['kind'] == 'minimal_native' and r['id'] in correct_ids)]
assert len(guards) == guard_policy['guard_states']
next_tokens = {}
for sequence in meta['sequences']:
    for j, pos in enumerate(sequence['positions']):
        if pos + 1 < len(sequence['ids']):
            next_tokens[sequence['offset'] + j] = sequence['ids'][pos + 1]
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'reason': 'Realize the earlier stable-reference joint predicates on the late reference validated in full-model forwards. Retain the earlier directions rather than introducing another classifier.',
    'direction_formula': 'Use the frozen896x46 directions from the preceding controlled counterfactual. Remove their component along the new constant direction, then normalize each to unit mean own-cell response divided by the physical reference.',
    'gate_formula': 'Same .05 minimum gap, .2/.8 threshold fractions, steepness160 and F32 guarded-negative bound .002 as before. Paired SiLU ramp with negative-reference correction.',
    'state_scope': '4707 locally reconstructed last-block states; implementation validated against full-model forwards on9 histories510positions.',
    'guard_states': len(guards), 'correct_native_unit_questions': sorted(correct_ids),
    'neutral_native_states': guard_policy['neutral_native_states'],
    'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
    'new_weights_applied': False, 'free_generation_performed': False, 'counts_as_correction': False,
    'third_set_used': False, 'final_set_read': False,
    'source_sha256': hashlib.sha256((RDIR / f'late_{CONDITION}_reconstruction.npz').read_bytes()).hexdigest(),
    'direction_source_sha256': hashlib.sha256((RDIR / 'progress_reference_counterfactual.npz').read_bytes()).hexdigest(),
    'actual_validation_sha256': hashlib.sha256((RDIR / 'late_reference_actual_check.json').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
notes, columns, gate_rows, up_rows = [], [], [], []
physical_anchor = (anchor / gain).float()
for j, cell in enumerate(prior['predicates']):
    members, target = cell['members'], cell['target_token']
    direction = directions[:, j].clone()
    scale = ((P[members] @ direction) / anchors[members]).mean()
    assert scale > 0
    direction /= scale
    scores = (P @ direction) / anchors
    negatives = sorted(set([i for i in fit if rows[i]['next_token'] != target] + [i for i in guards if next_tokens.get(i) != target]))
    high, low = scores[members].min(), scores[negatives].max()
    gap = high - low
    entry = {'fact': cell['fact'], 'previous_token': cell['previous_token'], 'target_token': target,
             'members': members, 'positive_min': float(high), 'negative_max': float(low),
             'gap': float(gap), 'valid': False}
    if float(gap) > .05 and bool((anchors[members] > .25).all()):
        lower, upper = low + .2 * gap, low + .8 * gap
        beta = 160 / gap
        G = (torch.stack([beta * (direction - lower * anchor), beta * (direction - upper * anchor), -160 * anchor]) / gain).float()
        ramp_scale = float(beta * (upper - lower))
        U = torch.stack([physical_anchor / ramp_scale, physical_anchor / ramp_scale, -physical_anchor / 160])
        raw = torch.nn.functional.silu(X @ G.T) * (X @ U.T)
        activation = raw[:, 0] - raw[:, 1] - raw[:, 2]
        bound = float(activation[negatives].abs().max())
        entry.update({'negative_bound': bound, 'valid': bound < .002})
        if entry['valid']:
            columns.append(activation)
            gate_rows.append(G)
            up_rows.append(U)
    notes.append(entry)
valid = [r for r in notes if r['valid']]
F = torch.stack(columns, dim=1) if columns else torch.empty((len(rows), 0))
coverage = []
for split, indices in [('fit', fit), ('probe_observed', probe)]:
    questions, states, first = {}, 0, 0
    for i in indices:
        own = [j for j, cell in enumerate(valid) if cell['target_token'] == rows[i]['next_token']]
        other = [j for j, cell in enumerate(valid) if cell['target_token'] != rows[i]['next_token']]
        own_value = float(F[i, own].max()) if own else 0
        other_value = float(F[i, other].abs().max()) if other else 0
        good = own_value > .9 * float(anchors[i] ** 2) and float(anchors[i]) > .25 and other_value < .002
        states += good
        first += good and rows[i]['subtoken'] == 0
        questions.setdefault(rows[i]['id'], []).append(good)
    coverage.append({'split': split, 'states': len(indices), 'questions': len(questions),
                     'saturated_correct_states': states, 'first_steps_covered': first,
                     'all_steps_covered': sum(all(v) for v in questions.values())})
report = {'spec': spec, 'cells': len(notes), 'valid_cells': len(valid),
          'coverage': coverage, 'predicates': notes,
          'anchor_min': float(anchors.min()), 'anchor_max': float(anchors.max())}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
np.savez_compressed(RDIR / f'{STEM}.npz', anchor=anchor.numpy(), rms=rms.numpy(), features=F.numpy(),
                    gate_rows=torch.cat(gate_rows).numpy() if gate_rows else np.empty((0, 896)),
                    up_rows=torch.cat(up_rows).numpy() if up_rows else np.empty((0, 896)))
print({key: value for key, value in report.items() if key not in ['spec', 'predicates']}, flush=True)
