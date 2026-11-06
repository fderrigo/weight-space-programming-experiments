"""Read-only relation/progress factorization on existing concept histories.

The class readouts are diagnostics, not runtime routing or successful edits.
No threshold sweep, gradient, loss, or new question is used.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
OUT = RESULTS / 'carrier_concept_factors.json'
assert not OUT.exists()
torch.set_grad_enabled(False)
torch.set_num_threads(2)
meta_path = RESULTS / 'carrier_concept_states.json'
meta = json.loads(meta_path.read_text('utf8'))
assert meta['completed']
rows = meta['records']
count = meta['spec']['factual_states']
assert count == 288
fit = [i for i, row in enumerate(rows[:count]) if not row['id'].startswith('second_')]
probe = [i for i, row in enumerate(rows[:count]) if row['id'].startswith('second_')]
assert len(fit) == 196 and len(probe) == 92
tokens = sorted({rows[i]['last_token'] for i in fit})
assert len(tokens) == 32
with np.load(RESULTS / 'semantic_carrier_geometry.npz') as z:
    carrier = torch.from_numpy(z['carrier'].copy()).double()
    semantic_basis = torch.from_numpy(z['semantic_basis'].copy()).double()
    projection = torch.from_numpy(z['mean_projection'].copy()).double()
    relation_centers = torch.from_numpy(z['semantic_centers'].copy()).double()
    alpha = float(z['alpha'])
question_meta = json.loads((RESULTS / 'question_layer_diagnostic_states.json').read_text('utf8'))
classes = sorted({row['label'] for row in question_meta['sequences']})
with safe_open(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors', framework='pt') as f:
    gain = f.get_tensor('model.layers.23.post_attention_layernorm.weight').double()


def normalize(value):
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def canonical(fact):
    return {'wood_finder': 'wood_donor'}.get(fact, fact)


mapping = {}
for row in rows[:count]:
    pair = (canonical(row['fact']), row['last_token'])
    assert pair not in mapping or mapping[pair] == row['next_token']
    mapping[pair] = row['next_token']
spec = {
    'decision_by': 'Codex', 'fit_questions': 50, 'probe_questions_already_observed': 24,
    'fit_states': len(fit), 'probe_states': len(probe), 'progress_token_classes': tokens,
    'stage_feature': 'Undo the last-MLP RMS gains, project out the twelve carrier directions, then take one mean per preceding token using only the fifty earlier questions.',
    'stage_readouts': ['cosine_to_token_means', 'linear_dual_of_token_means'],
    'relation_readout': 'Cosine to the same thirteen original semantic centers; compare original prefix-value semantics with the actual protected carrier.',
    'parameter_search': False, 'weights_changed': False, 'runtime_routing': False,
    'training': False, 'gradients': False, 'optimizer': False,
    'diagnostic_only': True, 'counts_as_correction': False,
    'third_set_used': False, 'final_set_read': False,
    'source_sha256': hashlib.sha256(meta_path.read_bytes()).hexdigest(),
}
reports = []
saved = {}
for mode in ['original', 'protected']:
    with np.load(RESULTS / f'carrier_concept_states_{mode}.npz') as z:
        X = torch.from_numpy(z['X'].copy()).double()
        means = torch.from_numpy(z['prefix_values'].copy()).double()
    physical = X / gain
    progress = physical - (physical @ carrier) @ carrier.T
    centers = torch.stack([progress[[i for i in fit if rows[i]['last_token'] == token]].mean(0) for token in tokens])
    dual = torch.linalg.pinv(centers, rtol=1e-8)
    relation = means @ projection @ semantic_basis.T if mode == 'original' else physical @ carrier / alpha
    relation_scores = normalize(relation) @ normalize(relation_centers).T
    relation_predicted = [classes[i] for i in relation_scores.argmax(-1).tolist()]
    relation_margin = relation_scores.topk(2).values[:, 0] - relation_scores.topk(2).values[:, 1]
    saved[f'{mode}_progress_centers'] = centers.numpy()
    saved[f'{mode}_progress_dual'] = dual.numpy()
    saved[f'{mode}_relation_scores'] = relation_scores.float().numpy()
    for name, scores in [('cosine_to_token_means', normalize(progress) @ normalize(centers).T),
                         ('linear_dual_of_token_means', progress @ dual)]:
        predicted = [tokens[i] for i in scores.argmax(-1).tolist()]
        proposed = [mapping.get((relation_predicted[i], predicted[i])) for i in range(len(rows))]
        margins = scores.topk(2).values[:, 0] - scores.topk(2).values[:, 1]
        saved[f'{mode}_{name}_scores'] = scores.float().numpy()
        split_reports = []
        for split, selected in [('earlier_fit', fit), ('second_observed_probe', probe)]:
            by_question = {}
            for i in selected:
                by_question.setdefault(rows[i]['id'], []).append(proposed[i] == rows[i]['next_token'])
            split_reports.append({
                'split': split, 'states': len(selected),
                'relation_correct': sum(relation_predicted[i] == canonical(rows[i]['fact']) for i in selected),
                'progress_correct': sum(predicted[i] == rows[i]['last_token'] for i in selected),
                'proposed_token_correct': sum(proposed[i] == rows[i]['next_token'] for i in selected),
                'whole_concept_labels_correct': sum(all(values) for values in by_question.values()),
                'question_count': len(by_question),
                'median_relation_margin': float(relation_margin[selected].median()),
                'median_progress_margin': float(margins[selected].median()),
                'mistakes': [{'id': rows[i]['id'], 'subtoken': rows[i]['subtoken'],
                             'expected_fact': canonical(rows[i]['fact']), 'predicted_fact': relation_predicted[i],
                             'expected_previous_token': rows[i]['last_token'], 'predicted_previous_token': predicted[i],
                             'expected_next_token': rows[i]['next_token'], 'proposed_next_token': proposed[i]}
                            for i in selected if proposed[i] != rows[i]['next_token']],
            })
        native_reports = []
        for kind in sorted({r['kind'] for r in rows[count:]}):
            selected = [i for i in range(count, len(rows)) if rows[i]['kind'] == kind]
            eligible = [i for i in selected if rows[i]['last_token'] in tokens]
            fires = [i for i in selected if proposed[i] is not None]
            native_reports.append({'kind': kind, 'states': len(selected),
                                   'states_with_one_of_the_32_previous_tokens': len(eligible),
                                   'progress_correct_when_token_is_in_set': sum(predicted[i] == rows[i]['last_token'] for i in eligible),
                                   'potential_fact_requests_without_threshold': len(fires),
                                   'median_relation_margin_at_requests': float(relation_margin[fires].median()) if fires else None,
                                   'median_progress_margin_at_requests': float(margins[fires].median()) if fires else None})
        reports.append({'condition': mode, 'stage_readout': name, 'splits': split_reports, 'native': native_reports})
        print(mode, name, [{k: v for k, v in r.items() if k != 'mistakes'} for r in split_reports], native_reports, flush=True)
np.savez_compressed(RESULTS / 'carrier_concept_factors.npz', **saved)
OUT.write_text(json.dumps({'spec': spec, 'reports': reports}, ensure_ascii=False, indent=2), encoding='utf8')
