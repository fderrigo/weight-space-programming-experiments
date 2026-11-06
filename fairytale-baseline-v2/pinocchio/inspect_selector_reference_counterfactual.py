"""Diagnostic only: isolate the effect of a stable normalization reference."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'selector_reference_counterfactual'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
meta = json.loads((RDIR / 'progress_carrier_states.json').read_text('utf8'))
old = json.loads((RDIR / 'selector_fact_gates.json').read_text('utf8'))
actual = json.loads((RDIR / 'selector_carrier_states.json').read_text('utf8'))
rows = meta['records']
fit = [i for i, r in enumerate(rows[:288]) if not r['id'].startswith('second_')]
probe = [i for i in range(288) if i not in fit]
tokens = sorted({rows[i]['last_token'] for i in fit})
classes = actual['class_coordinates']
with np.load(RDIR / 'selector_carrier_states.npz') as z:
    X = torch.from_numpy(z['X'].copy()).double()
    selected = torch.from_numpy(z['selected_values'].copy()).double()
with np.load(RDIR / 'selector_fact_gates.npz') as z:
    stage = torch.from_numpy(z['progress_dual'].copy()).double()
    relation = torch.from_numpy(z['class_readout'].copy()).double()
    old_anchor = torch.from_numpy(z['anchor'].copy()).double()
    rms = torch.from_numpy(z['rms'].copy()).double()
with safe_open(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors', framework='pt') as f:
    gain = f.get_tensor('model.layers.23.post_attention_layernorm.weight').double()
P = X / gain
stage_scores = P @ stage
class_scores = P @ relation
correct_units = set(old['spec']['correct_native_unit_questions'])
guards = [i for i, r in enumerate(rows) if r['kind'] in ['external', 'outside_prototype'] or (r['kind'] == 'minimal_native' and r['id'] in correct_units)]
assert len(guards) == old['spec']['guard_states']
next_tokens = {}
for sequence in meta['sequences']:
    for j, pos in enumerate(sequence['positions']):
        if pos + 1 < len(sequence['ids']):
            next_tokens[sequence['offset'] + j] = sequence['ids'][pos + 1]


def canonical(fact):
    return {'wood_finder': 'wood_donor'}.get(fact, fact)


spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'decision_by': 'Codex',
    'hypothesis': 'A varying native mean anchor inflates joint scores on unrelated native tokens. Replace only this scalar with an ideal constant raw-residual reference divided by the actually observed RMS.',
    'conditions': ['existing_native_anchor', 'ideal_constant_reference'],
    'ideal_reference_formula': 'median(rms[earlier fifty question states]) / rms',
    'stage_and_relation_readouts_unchanged': True,
    'negative_guards_unchanged': True,
    'guard_states': len(guards),
    'not_a_weight_implementation': True,
    'weights_modified': False,
    'free_generation_performed': False,
    'counts_as_correction': False,
    'training': False, 'gradients': False, 'optimizer': False,
    'parameter_search': False, 'final_set_read': False,
    'source_sha256': hashlib.sha256((RDIR / 'selector_fact_gates.json').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
conditions = []
for name, anchors in [('existing_native_anchor', P @ old_anchor), ('ideal_constant_reference', rms[fit].median() / rms)]:
    cells, activations = [], []
    for old_cell in old['predicates']:
        fact, previous, target = [old_cell[k] for k in ['fact', 'previous_token', 'target_token']]
        members = old_cell['members']
        tj, cj = tokens.index(previous), classes.index(fact)
        same_stage = [i for i in fit if rows[i]['last_token'] == previous]
        tscale = (stage_scores[same_stage, tj] / anchors[same_stage]).mean()
        cscale = (class_scores[members, cj] / anchors[members]).mean()
        direction_values = (stage_scores[:, tj] / tscale + class_scores[:, cj] / cscale) / 2
        scores = direction_values / anchors
        negatives = sorted(set([i for i in fit if rows[i]['next_token'] != target] + [i for i in guards if next_tokens.get(i) != target]))
        eligible = [i for i in negatives if float(anchors[i]) > .05]
        worst = max(eligible, key=lambda i: float(scores[i]))
        high, low = scores[members].min(), scores[worst]
        gap = high - low
        entry = {'fact': fact, 'previous_token': previous, 'target_token': target,
                 'positive_min': float(high), 'negative_max': float(low), 'gap': float(gap),
                 'worst_negative': {'index': worst, 'id': rows[worst]['id'], 'kind': rows[worst]['kind'],
                                    'last_token': rows[worst]['last_token'], 'native_next_token': next_tokens.get(worst),
                                    'anchor': float(anchors[worst]),
                                    'normalized_stage_component': float(stage_scores[worst, tj] / tscale / anchors[worst]),
                                    'normalized_class_component': float(class_scores[worst, cj] / cscale / anchors[worst]),
                                    'selected_own_class': float(selected[worst, cj])}, 'valid': False}
        if float(gap) > .05 and bool((anchors[members] > .25).all()):
            lower, upper = low + .2 * gap, low + .8 * gap
            beta = 160 / gap
            a, d = anchors.float(), direction_values.float()
            activation = (torch.nn.functional.silu(beta * (d - lower * a)) - torch.nn.functional.silu(beta * (d - upper * a))) * a / float(beta * (upper - lower))
            activation -= torch.nn.functional.silu(-160 * a) * (-a / 160)
            bound = float(activation[negatives].abs().max())
            entry.update({'negative_bound': bound, 'valid': bound < .002})
            if entry['valid']:
                activations.append(activation)
        cells.append(entry)
    valid = [r for r in cells if r['valid']]
    F = torch.stack(activations, dim=1) if activations else torch.empty((len(rows), 0))
    coverage = []
    for split, indices in [('fit', fit), ('probe_observed', probe)]:
        per_question, count = {}, 0
        for i in indices:
            own = [j for j, r in enumerate(valid) if r['target_token'] == rows[i]['next_token']]
            other = [j for j, r in enumerate(valid) if r['target_token'] != rows[i]['next_token']]
            own_value = float(F[i, own].max()) if own else 0
            other_value = float(F[i, other].abs().max()) if other else 0
            good = own_value > .9 * float(anchors[i] ** 2) and float(anchors[i]) > .25 and other_value < .002
            count += good
            per_question.setdefault(rows[i]['id'], []).append(good)
        coverage.append({'split': split, 'states': len(indices), 'saturated_correct_states': count,
                         'questions': len(per_question), 'all_steps_covered': sum(all(v) for v in per_question.values())})
    conditions.append({'condition': name, 'valid_cells': len(valid), 'coverage': coverage, 'cells': cells})
report = {'spec': spec, 'conditions': conditions}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
for condition in conditions:
    print({k: v for k, v in condition.items() if k != 'cells'}, flush=True)
    print('First-token negatives:', [{k: v for k, v in c.items() if k not in ['positive_min', 'negative_max']} for c in condition['cells'] if c['previous_token'] == 198], flush=True)
