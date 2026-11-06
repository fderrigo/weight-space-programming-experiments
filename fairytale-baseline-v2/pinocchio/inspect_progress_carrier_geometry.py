"""One declared carrier geometry that includes intermediate concept states.

Replace the previous start-only incoming covariance with existing incoming
states throughout concepts and native continuations. Keep the same semantic
basis, amplitude, rank, and three trace-normalized covariance terms.
"""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
STEM = 'progress_carrier_geometry'
assert not (RESULTS / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
meta_path = RESULTS / 'carrier_concept_states.json'
meta = json.loads(meta_path.read_text('utf8'))
assert meta['completed']
rows = meta['records']
fit = [i for i, r in enumerate(rows[:288]) if not r['id'].startswith('second_')]
probe = [i for i, r in enumerate(rows[:288]) if r['id'].startswith('second_')]
native = list(range(288, len(rows)))
with np.load(RESULTS / 'semantic_carrier_geometry.npz') as z:
    old_arrays = {key: z[key].copy() for key in z.files}
old_U = torch.from_numpy(old_arrays['carrier']).double()
B = torch.from_numpy(old_arrays['semantic_basis']).double()
Pmu = torch.from_numpy(old_arrays['mean_projection']).double()
centers = torch.from_numpy(old_arrays['semantic_centers']).double()
alpha = float(old_arrays['alpha'])
with np.load(RESULTS / 'carrier_concept_states_original.npz') as z:
    incoming = torch.from_numpy(z['incoming21'].copy()).double()
    residual = torch.from_numpy(z['R'].copy()).double()
    means = torch.from_numpy(z['prefix_values'].copy()).double()
qmeta = json.loads((RESULTS / 'question_layer_diagnostic_states.json').read_text('utf8'))
classes = sorted({r['label'] for r in qmeta['sequences']})
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'reason': 'The old carrier receives inherited native content larger than its semantic signal during partial names. Include those incoming states when choosing quiet directions.',
    'source_collection_sha256': hashlib.sha256(meta_path.read_bytes()).hexdigest(),
    'source_geometry_sha256': hashlib.sha256((RESULTS / 'semantic_carrier_geometry.npz').read_bytes()).hexdigest(),
    'formula': 'Bottom twelve eigenvectors of C_incoming/trace + C_native_final/trace + C_effective_decoder/trace; uncentered second moments. Incoming uses 4405 native positions plus 196 factual positions of the earlier fifty questions. Final uses those 4405 native positions. No probe factual positions enter the covariance.',
    'amplitude_unchanged': alpha, 'semantic_basis_unchanged': True,
    'rank_unchanged': 12, 'candidate_geometries': 1,
    'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
    'weights_changed': False, 'free_generation_performed': False, 'counts_as_correction': False,
    'probe_questions_already_observed': True, 'third_set_used': False, 'final_set_read': False,
}
(RESULTS / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
cov_incoming = incoming[fit + native].T @ incoming[fit + native] / (len(fit) + len(native))
cov_native = residual[native].T @ residual[native] / len(native)
with safe_open(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors', framework='pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
cov_decoder = torch.zeros((896, 896), dtype=torch.float64)
for begin in range(0, len(E), 4096):
    chunk = E[begin:begin + 4096].double()
    cov_decoder += chunk.T @ chunk
cov_decoder /= len(E)
metric = cov_incoming / cov_incoming.trace() + cov_native / cov_native.trace() + cov_decoder / cov_decoder.trace()
eigenvalues, eigenvectors = torch.linalg.eigh(metric)
U = eigenvectors[:, :12]
signal = alpha * means @ Pmu @ B.T


def normalize(value):
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def canonical(fact):
    return {'wood_finder': 'wood_donor'}.get(fact, fact)


reports = []
for name, basis in [('previous', old_U), ('progress_aware', U)]:
    inherited = incoming @ basis
    ratio = inherited.norm(dim=-1) / signal.norm(dim=-1).clamp_min(1e-12)
    expected = inherited + signal
    predicted = [classes[i] for i in (normalize(expected) @ normalize(centers).T).argmax(-1).tolist()]
    comparisons = []
    for split, selected in [('earlier_fifty', fit), ('second_observed', probe)]:
        questions = {}
        for i in selected:
            questions.setdefault(rows[i]['id'], []).append(predicted[i] == canonical(rows[i]['fact']))
        comparisons.append({'split': split, 'states': len(selected),
                            'relation_correct': sum(predicted[i] == canonical(rows[i]['fact']) for i in selected),
                            'all_steps_relation_correct': sum(all(v) for v in questions.values()),
                            'questions': len(questions)})
    ratios = {}
    for group, selected in [('first', [i for i, r in enumerate(rows[:288]) if r['subtoken'] == 0]),
                            ('partial', [i for i, r in enumerate(rows[:288]) if r['subtoken'] > 0]),
                            ('native', native)]:
        ratios[group] = {'median': float(ratio[selected].median()), 'max': float(ratio[selected].max())}
    reports.append({'geometry': name, 'inherited_over_signal': ratios, 'comparisons': comparisons,
                    'energy_fractions': {key: float(torch.trace(basis.T @ matrix @ basis) / matrix.trace())
                                         for key, matrix in [('incoming', cov_incoming), ('native_final', cov_native), ('decoder', cov_decoder)]}})
    print(reports[-1], flush=True)
new_arrays = dict(old_arrays)
new_arrays.update(carrier=U.numpy(), covariance_eigenvalues=eigenvalues.numpy())
np.savez_compressed(RESULTS / f'{STEM}.npz', **new_arrays)
(RESULTS / f'{STEM}.json').write_text(json.dumps({'spec': spec, 'comparisons': reports}, indent=2), encoding='utf8')
