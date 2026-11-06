"""Controlled descriptive comparisons before changing semantic transport."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'semantic_information_loss'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
old_meta = json.loads((RDIR / 'question_layer_diagnostic_states.json').read_text('utf8'))['sequences']
third_meta = json.loads((RDIR / 'third_question_features_states.json').read_text('utf8'))['rows']
with np.load(RDIR / 'attention_feature_diagnostic_states.npz') as z:
    prefix = torch.from_numpy(z['prefix_values'][:, 21].copy()).flatten(1).double()
    head = torch.from_numpy(z['heads'][:, 14, 13].copy()).double()
with np.load(RDIR / 'third_question_features_states.npz') as z:
    prefix = torch.cat([prefix, torch.from_numpy(z['prefix_values_21'].copy()).double()])
    head = torch.cat([head, torch.from_numpy(z['native_head_14_13'].copy()).double()])
with np.load(RDIR / 'semantic_carrier_geometry.npz') as z:
    Pmu = torch.from_numpy(z['mean_projection'].copy()).double()
    B = torch.from_numpy(z['semantic_basis'].copy()).double()
    original_centers = torch.from_numpy(z['semantic_centers'].copy()).double()
rows = [dict(r, group='control' if r['label'] == 'outside' else 'target') for r in old_meta] + third_meta
fit = [i for i, r in enumerate(old_meta) if r['split'] == 'prototype']
second = [i for i, r in enumerate(old_meta) if r['split'] == 'probe']
third = list(range(len(old_meta), len(rows)))
classes = sorted({r['label'] for r in old_meta})
facts = [c for c in classes if c != 'outside']
outside = [i for i in fit if rows[i]['label'] == 'outside']
assert len(fit) == 64 and len(outside) == 14
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'hypotheses': ['Orthogonal removal of the common mean is not affine centering; measure whether it destroys relation distinctions.', 'Original head14_13 and prefix21 carry complementary question information; test one fixed equal-score fusion.'],
    'comparisons': ['Projected128 prefix21 versus the frozen12-dimensional row-space compression, same13 centers.', 'Affine-centered prefix21 with13 centers, then twelve factual centers and14 individual outside prototypes.', 'Affine-centered head14_13 with the same26 reference identities; fixed equal average of its cosine scores and prefix21 scores.'],
    'fusion_weights': [.5, .5], 'reference_rule': 'Same64 earlier prototypes; no second or third examples added to the reference bank.',
    'scope': 'Descriptive first-position original-model features only. No proof that these quantities can be transported through generated concepts or implemented without native disturbance.',
    'third_set_already_observed': True, 'final_set_read': False,
    'weights_modified': False, 'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
    'free_generation_performed': False, 'counts_as_correction': False,
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def centers(x):
    return torch.stack([x[[i for i in fit if rows[i]['label'] == c]].mean(0) for c in classes])


projected = prefix @ Pmu
projected_centers = unit(centers(projected))
scores_full = unit(projected) @ projected_centers.T
scores_compressed = unit(projected @ B.T) @ unit(original_centers).T
centered_prefix = prefix - prefix[fit].mean(0)
centered_head = head - head[fit].mean(0)
mean_prefix = centers(centered_prefix)
mean_head = centers(centered_head)
reference_labels = facts + ['outside'] * 14
prefix_prototypes = torch.cat([mean_prefix[[classes.index(c) for c in facts]], centered_prefix[outside]])
head_prototypes = torch.cat([mean_head[[classes.index(c) for c in facts]], centered_head[outside]])
prefix_scores = unit(centered_prefix) @ unit(prefix_prototypes).T
head_scores = unit(centered_head) @ unit(head_prototypes).T
conditions = [('projected_prefix21_full128', scores_full, classes),
              ('projected_prefix21_compressed12', scores_compressed, classes),
              ('centered_prefix21_13centers', unit(centered_prefix) @ unit(mean_prefix).T, classes),
              ('centered_prefix21_26prototypes', prefix_scores, reference_labels),
              ('centered_head14_26prototypes', head_scores, reference_labels),
              ('equal_fusion_26prototypes', .5 * (prefix_scores + head_scores), reference_labels)]
reports = []
for name, scores, labels in conditions:
    predictions = [labels[i] for i in scores.argmax(-1).tolist()]
    for split, selected in [('prototype_fit', fit), ('second_observed', second), ('third_observed', third)]:
        targets = [i for i in selected if rows[i]['group'] == 'target']
        controls = [i for i in selected if rows[i]['group'] == 'control']
        reports.append({'condition': name, 'split': split,
                        'targets_correct': sum(predictions[i] == rows[i]['label'] for i in targets), 'targets': len(targets),
                        'controls_outside': sum(predictions[i] == 'outside' for i in controls), 'controls': len(controls),
                        'predictions': [{'id': rows[i]['id'], 'expected': rows[i]['label'], 'predicted': predictions[i]} for i in selected]})
equivalence = float(((projected @ projected_centers.T) - (projected @ B.T) @ original_centers.T).abs().max())
assert equivalence < 1e-8
report = {'spec': spec, 'compression_preserves_13_center_linear_scores_max_error': equivalence, 'comparisons': reports}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print('Compression score identity', equivalence)
for r in reports:
    print({k: v for k, v in r.items() if k != 'predictions'})
