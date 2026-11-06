"""Fixed unsupervised prototype-covariance metric for descriptive probes.

The covariance uses no fact labels. Class labels are used only to report nearest
reference/centroid relation identification. No model weights or learned module
are changed, and the isotropic floor is fixed before results, not searched.
"""
import datetime
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors'
STEM = 'question_covariance_probe'
output = ROOT / f'results/{STEM}.json'
assert not output.exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
old_rows = json.loads((ROOT / 'results/question_layer_diagnostic_states.json').read_text('utf8'))['sequences']
new_rows = json.loads((ROOT / 'results/third_question_features_states.json').read_text('utf8'))['rows']
train = [i for i, row in enumerate(old_rows) if row['split'] == 'prototype']
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'formula': 'S=centered_prototypes.T*centered_prototypes/(n-1); lambda=.1*trace(S)/d. Apply (S+lambda*I)^(-1/2) to centered vectors, then the same cosine nearest-example and class-mean probes.',
    'isotropic_fraction': .1, 'prototype_count': len(train),
    'covariance_uses_fact_labels': False, 'automatic_parameter_search': False,
    'weights_changed': False, 'training': False, 'gradients': False, 'optimizer': False,
    'counts_as_correction': False, 'third_set_already_observed': True, 'final_set_read': False,
}
(ROOT / f'results/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
with np.load(ROOT / 'results/attention_feature_diagnostic_states.npz') as z:
    old = {'native_head_14_13': torch.from_numpy(z['heads'][:, 14, 13].copy()),
           'prefix_values_21': torch.from_numpy(z['prefix_values'][:, 21].copy()).flatten(1),
           'prefix_values_23': torch.from_numpy(z['prefix_values'][:, 23].copy()).flatten(1)}
with np.load(ROOT / 'results/question_layer_diagnostic_states.npz') as z:
    old['last_mlp_input'] = torch.from_numpy(z['X'][:, 0, 23].copy())
with safe_open(MODEL, framework='pt') as f:
    G = f.get_tensor('model.layers.23.mlp.gate_proj.weight').float()
    U = f.get_tensor('model.layers.23.mlp.up_proj.weight').float()
old['last_mlp_features'] = torch.nn.functional.silu(old['last_mlp_input'] @ G.T) * (old['last_mlp_input'] @ U.T)
with np.load(ROOT / 'results/third_question_features_states.npz') as z:
    new = {name: torch.from_numpy(z[name].copy()) for name in old}
labels = [row['label'] for row in old_rows]
classes = sorted(set(labels))
probes = []


def normalize(value):
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)


for name, features in old.items():
    mean = features[train].double().mean(0)
    prototype = features[train].double() - mean
    _, singular, directions = torch.linalg.svd(prototype, full_matrices=False)
    variance = singular.square() / (len(train) - 1)
    floor = .1 * variance.sum() / prototype.shape[1]
    adjustment = torch.rsqrt(variance + floor) - torch.rsqrt(floor)

    def transform(values):
        centered = values.double() - mean
        return centered * torch.rsqrt(floor) + ((centered @ directions.T) * adjustment) @ directions

    old_f, new_f = transform(features), transform(new[name])
    centers = torch.stack([old_f[[i for i in train if labels[i] == label]].mean(0) for label in classes])
    for method, reference, reference_labels in [('class_mean', centers, classes),
                                                ('nearest_example', old_f[train], [labels[i] for i in train])]:
        for dataset, values, rows, selected in [
            ('earlier_probe', old_f, old_rows, [i for i, row in enumerate(old_rows) if row['split'] == 'probe']),
            ('observed_third', new_f, new_rows, list(range(len(new_rows)))),
        ]:
            scores = normalize(values) @ normalize(reference).T
            predictions = [reference_labels[i] for i in scores.argmax(-1).tolist()]
            targets = [i for i in selected if rows[i]['label'] != 'outside']
            controls = [i for i in selected if rows[i]['label'] == 'outside']
            probe = {'feature': name, 'method': method, 'dataset': dataset,
                     'isotropic_floor': float(floor),
                     'target_correct': sum(predictions[i] == rows[i]['label'] for i in targets),
                     'outside_correct': sum(predictions[i] == 'outside' for i in controls),
                     'predictions': [{'id': rows[i]['id'], 'expected': rows[i]['label'], 'predicted': predictions[i]} for i in selected]}
            probes.append(probe)
            print({key: value for key, value in probe.items() if key != 'predictions'}, flush=True)
output.write_text(json.dumps({**spec, 'probes': probes}, indent=2), encoding='utf8')
