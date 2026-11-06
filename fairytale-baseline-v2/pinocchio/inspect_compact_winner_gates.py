"""Check the fixed24-prototype compression against all unchanged gate guards."""
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
STEM = 'compact_winner_gates'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
meta = json.loads((RDIR / 'progress_carrier_states.json').read_text('utf8'))
rows, sequences = meta['records'], meta['sequences']
fit = [i for i, r in enumerate(rows[:288]) if not r['id'].startswith('second_')]
probe = [i for i in range(288) if i not in fit]
classes = sorted({r['label'] for r in json.loads((RDIR / 'question_layer_diagnostic_states.json').read_text('utf8'))['sequences']})
facts = [c for c in classes if c != 'outside']
prior = json.loads((RDIR / 'late_reference_fact_gates.json').read_text('utf8'))
guard_policy = prior['spec']
with np.load(RDIR / f'late_{CONDITION}_reconstruction.npz') as z:
    X = torch.from_numpy(z['X'].copy()).double()
    R = torch.from_numpy(z['R'].copy()).double()
with np.load(RDIR / 'late_reference_geometry.npz') as z:
    B = torch.from_numpy(z['protected_carrier'].copy()).double()
with np.load(RDIR / 'progress_carrier_geometry.npz') as z:
    U = torch.from_numpy(z['carrier'].copy()).double()
    C = torch.from_numpy(z['semantic_centers'].copy()).double()
with np.load(RDIR / 'late_reference_fact_gates.npz') as z:
    anchor = torch.from_numpy(z['anchor'].copy()).double()
with safe_open(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors', framework='pt') as f:
    gain = f.get_tensor('model.layers.23.post_attention_layernorm.weight').double()
P = X / gain
anchors = P @ anchor
Pn = P - (P @ B) @ B.T
tokens = sorted({rows[i]['last_token'] for i in fit})
stage_centers = torch.stack([Pn[[i for i in fit if rows[i]['last_token'] == token]].mean(0) for token in tokens])
stage = torch.linalg.pinv(stage_centers, rtol=1e-8)
stage_scores = P @ stage
Z = R @ U
outside_indices = [i for i, r in enumerate(rows) if r['kind'] == 'outside_prototype']
assert outside_indices == list(range(4693, 4707))
correct_ids = set(guard_policy['correct_native_unit_questions'])
guards = [i for i, r in enumerate(rows) if r['kind'] in ['external', 'outside_prototype'] or (r['kind'] == 'minimal_native' and r['id'] in correct_ids)]
next_tokens = {}
for sequence in sequences:
    for j, pos in enumerate(sequence['positions']):
        if pos + 1 < len(sequence['ids']):
            next_tokens[sequence['offset'] + j] = sequence['ids'][pos + 1]
with np.load(RDIR / 'compact_outside_prototypes.npz') as z:
    compact = torch.from_numpy(z['semantic_directions'].copy()).double()
    compact_labels = z['labels'].tolist()
configurations = [('compact_outside_prototypes', compact, compact_labels)]
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'hypothesis': 'Two fixed geometric outside merges retain all factual relation labels and all58 first-position external labels, but change five other native classifications. Verify whether the compact24-prototype representation still supports all guarded ideal correction cells.',
    'class_rule': 'Use exactly the twelve factual and twelve merged outside directions already saved by compact_outside_prototypes. No new merge or direction choice.',
    'compact_source_sha256': hashlib.sha256((RDIR / 'compact_outside_prototypes.npz').read_bytes()).hexdigest(),
    'ideal_gate_formula': 'Average a normalized preceding-token readout with anchor times the binary indicator that the fact wins. Apply the unchanged .05 gap, .2/.8 ramp, steepness160 and .002 negative bound.',
    'ideal_indicator_is_not_a_model_weight_feature': True,
    'stage_readout': 'Pseudoinverse of32 token means in the actual normalized native subspace, using only earlier fit questions.',
    'guard_policy_unchanged': True, 'guard_states': len(guards),
    'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
    'new_weights_modified': False, 'free_generation_performed': False, 'counts_as_correction': False,
    'third_set_used': False, 'final_set_read': False,
    'source_sha256': hashlib.sha256((RDIR / f'late_{CONDITION}_reconstruction.npz').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
reports, saved = [], {}


def canonical(fact):
    return {'wood_finder': 'wood_donor'}.get(fact, fact)


for name, centers, labels in configurations:
    directions = centers / centers.norm(dim=-1, keepdim=True)
    predictions = [labels[j] for j in (Z @ directions.T).argmax(-1).tolist()]
    classification = []
    for split, indices in [('fit', fit), ('probe_observed', probe)]:
        ids = sorted({rows[i]['id'] for i in indices})
        good = lambda i: predictions[i] == canonical(rows[i]['fact'])
        classification.append({'split': split, 'states': len(indices), 'questions': len(ids),
                               'correct_states': sum(good(i) for i in indices),
                               'correct_first_states': sum(good(i) for i in indices if rows[i]['subtoken'] == 0),
                               'whole_relation_paths': sum(all(good(i) for i in indices if rows[i]['id'] == q) for q in ids)})
    cells, columns = [], []
    for cell in prior['predicates']:
        fact, previous, target = [cell[k] for k in ['fact', 'previous_token', 'target_token']]
        members = cell['members']
        j = tokens.index(previous)
        same_stage = [i for i in fit if rows[i]['last_token'] == previous]
        scale = (stage_scores[same_stage, j] / anchors[same_stage]).mean()
        class_indicator = torch.tensor([p == fact for p in predictions], dtype=torch.float64)
        d = (stage_scores[:, j] / scale + anchors * class_indicator) / 2
        scores = d / anchors
        negatives = sorted(set([i for i in fit if rows[i]['next_token'] != target] + [i for i in guards if next_tokens.get(i) != target]))
        worst = max(negatives, key=lambda i: float(scores[i]))
        high, low = scores[members].min(), scores[worst]
        gap = high - low
        entry = {'fact': fact, 'previous_token': previous, 'target_token': target,
                 'members': members, 'positive_min': float(high), 'negative_max': float(low),
                 'gap': float(gap), 'valid': False,
                 'worst_negative': {'index': worst, 'id': rows[worst]['id'], 'kind': rows[worst]['kind'], 'predicted_class': predictions[worst]}}
        if float(gap) > .05 and bool((anchors[members] > .25).all()):
            lower, upper = low + .2 * gap, low + .8 * gap
            beta = 160 / gap
            a, value = anchors.float(), d.float()
            F = (torch.nn.functional.silu(beta * (value - lower * a)) - torch.nn.functional.silu(beta * (value - upper * a))) * a / float(beta * (upper - lower))
            F -= torch.nn.functional.silu(-160 * a) * (-a / 160)
            bound = float(F[negatives].abs().max())
            entry.update({'negative_bound': bound, 'valid': bound < .002})
            if entry['valid']:
                columns.append(F)
        cells.append(entry)
    valid = [r for r in cells if r['valid']]
    F = torch.stack(columns, dim=1) if columns else torch.empty((len(rows), 0))
    coverage = []
    for split, indices in [('fit', fit), ('probe_observed', probe)]:
        questions, count, first = {}, 0, 0
        for i in indices:
            own = [j for j, c in enumerate(valid) if c['target_token'] == rows[i]['next_token']]
            other = [j for j, c in enumerate(valid) if c['target_token'] != rows[i]['next_token']]
            high = float(F[i, own].max()) if own else 0
            low = float(F[i, other].abs().max()) if other else 0
            good = high > .9 * float(anchors[i] ** 2) and float(anchors[i]) > .25 and low < .002
            count += good
            first += good and rows[i]['subtoken'] == 0
            questions.setdefault(rows[i]['id'], []).append(good)
        coverage.append({'split': split, 'states': len(indices), 'questions': len(questions),
                         'saturated_correct_states': count, 'first_steps_covered': first,
                         'all_steps_covered': sum(all(v) for v in questions.values())})
    external = [i for i, r in enumerate(rows) if r['kind'] == 'external']
    external_first = [s['offset'] for s in sequences if s['kind'] == 'external']
    report = {'condition': name, 'prototype_count': len(labels), 'classification': classification,
              'external_states': len(external), 'external_classified_outside': sum(predictions[i] == 'outside' for i in external),
              'external_first_states': len(external_first), 'external_first_classified_outside': sum(predictions[i] == 'outside' for i in external_first),
              'valid_ideal_cells': len(valid), 'ideal_coverage': coverage, 'predicates': cells}
    reports.append(report)
    saved[name + '_semantic_directions'] = directions.numpy()
    saved[name + '_labels'] = np.array(labels)
    saved[name + '_predictions'] = np.array(predictions)
    saved[name + '_ideal_features'] = F.numpy()
    print({k: v for k, v in report.items() if k != 'predicates'}, flush=True)
np.savez_compressed(RDIR / f'{STEM}.npz', stage_dual=stage.numpy(), anchor=anchor.numpy(), **saved)
(RDIR / f'{STEM}.json').write_text(json.dumps({'spec': spec, 'conditions': reports}, indent=2), encoding='utf8')

