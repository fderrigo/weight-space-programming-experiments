"""Geometry of a small residual subspace for carrying question information.

No weights are edited. Test a declared carrier before attempting cross-layer
projections. Its basis minimizes three trace-normalized second moments; its
amplitude is fixed at ten percent of the median native question-residual norm.
"""
import datetime
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors'
STEM = 'semantic_carrier_geometry'
assert not (ROOT / f'results/{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
rows = json.loads((ROOT / 'results/question_layer_diagnostic_states.json').read_text('utf8'))['sequences']
train = [i for i, row in enumerate(rows) if row['split'] == 'prototype']
probe = [i for i, row in enumerate(rows) if row['split'] == 'probe']
labels = [row['label'] for row in rows]
classes = sorted(set(labels))
with np.load(ROOT / 'results/attention_feature_diagnostic_states.npz') as z:
    F = torch.from_numpy(z['prefix_values'][:, 21].copy()).flatten(1).double()
with np.load(ROOT / 'results/question_layer_diagnostic_states.npz') as z:
    incoming = torch.from_numpy(z['H'][:, 0, 21].copy()).double()
with np.load(ROOT / 'results/extended_native_inputs.npz') as z:
    native = torch.from_numpy(z['R'].copy()).double()
assert len(native) == 2642 and len(train) == 64
mu = F[train].mean(0)
Pmu = torch.eye(128, dtype=torch.float64) - torch.outer(mu, mu) / (mu @ mu)
projected = F @ Pmu
centers = torch.stack([projected[[i for i in train if labels[i] == label]].mean(0) for label in classes])
centers /= centers.norm(dim=-1, keepdim=True).clamp_min(1e-12)
_, s, directions = torch.linalg.svd(centers, full_matrices=False)
rank = int((s > s[0] * 1e-6).sum())
B = directions[:rank]
assert rank <= 13
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'semantic_dimensions': rank, 'prototype_count': len(train),
    'semantic_basis': 'Row space of the 13 normalized class centers after projecting prefix-value means orthogonally to their common mean; no learned classifier or parameter search.',
    'carrier_basis': 'Lowest-eigenvalue directions of native_residual_second_moment/trace + prototype_block21_input_second_moment/trace + effective_unembedding_second_moment/trace.',
    'amplitude_rule': '.1*median(||prototype incoming residual||)/max(||prototype semantic coordinates||).',
    'weights_changed': False, 'training': False, 'gradients': False, 'optimizer': False,
    'full_generation': False, 'counts_as_correction': False,
    'third_questions_used': False, 'final_set_read': False,
}
(ROOT / f'results/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
print('Semantic carrier dimension', rank, 'computing native and decoder second moments', flush=True)
cov_native = native.T @ native / len(native)
cov_question = incoming[train].T @ incoming[train] / len(train)
with safe_open(MODEL, framework='pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').float()
    E *= f.get_tensor('model.norm.weight').float()
cov_decoder = torch.zeros((896, 896), dtype=torch.float64)
for begin in range(0, len(E), 4096):
    chunk = E[begin:begin + 4096].double()
    cov_decoder += chunk.T @ chunk
cov_decoder /= len(E)
metric = (cov_native / cov_native.trace() + cov_question / cov_question.trace()
          + cov_decoder / cov_decoder.trace())
eigenvalues, eigenvectors = torch.linalg.eigh(metric)
Ucarrier = eigenvectors[:, :rank]
Z = projected @ B.T
alpha = .1 * incoming[train].norm(dim=-1).median() / Z[train].norm(dim=-1).max()
inherited = incoming @ Ucarrier
signal = alpha * Z
register = inherited + signal
semantic_centers = centers @ B.T


def normalize(value):
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)


comparisons = []
for name, values in [('pure_projected_semantics', Z), ('carrier_with_inherited_residual', register / alpha)]:
    predictions = [classes[i] for i in (normalize(values) @ normalize(semantic_centers).T).argmax(-1).tolist()]
    comparisons.append({'representation': name,
                        'target_correct': sum(predictions[i] == labels[i] for i in probe if labels[i] != 'outside'),
                        'outside_correct': sum(predictions[i] == 'outside' for i in probe if labels[i] == 'outside'),
                        'predictions': [{'id': rows[i]['id'], 'expected': labels[i], 'predicted': predictions[i]} for i in probe]})
fractions = {name: float(torch.trace(Ucarrier.T @ matrix @ Ucarrier) / matrix.trace())
             for name, matrix in [('native_residual', cov_native), ('question_residual', cov_question), ('decoder', cov_decoder)]}
noise_ratio = inherited.norm(dim=-1) / signal.norm(dim=-1).clamp_min(1e-12)
report = {
    **spec, 'carrier_amplitude': float(alpha), 'energy_fractions_in_carrier': fractions,
    'median_inherited_noise_over_signal': float(noise_ratio[probe].median()),
    'max_inherited_noise_over_signal': float(noise_ratio[probe].max()),
    'mean_added_residual_norm': float(signal[probe].norm(dim=-1).mean()),
    'max_added_residual_norm': float(signal[probe].norm(dim=-1).max()),
    'max_prototype_added_norm_over_median_native_norm': float(signal[train].norm(dim=-1).max() / incoming[train].norm(dim=-1).median()),
    'carrier_orthogonality_error': float((Ucarrier.T @ Ucarrier - torch.eye(rank)).norm()),
    'comparisons': comparisons,
    'interpretation': 'A proposed residual carrier geometry, not an applied weight edit. Inherited residual content may obscure the semantic signal. No generated-answer or locality success is claimed.',
}
(ROOT / f'results/{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
np.savez_compressed(ROOT / f'results/{STEM}.npz', carrier=Ucarrier.numpy(), semantic_basis=B.numpy(),
                    feature_mean=mu.numpy(), mean_projection=Pmu.numpy(), semantic_centers=semantic_centers.numpy(),
                    alpha=np.array(float(alpha)), covariance_eigenvalues=eigenvalues.numpy())
print({key: value for key, value in report.items() if key != 'comparisons'}, flush=True)
for comparison in comparisons:
    print({key: value for key, value in comparison.items() if key != 'predictions'}, flush=True)
