"""Readout and identity checks for the one progress-aware carrier candidate."""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'progress_carrier_readout'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
meta = json.loads((RDIR / 'progress_carrier_states.json').read_text('utf8'))
assert meta['completed']
rows = meta['records']
fit = [i for i, r in enumerate(rows[:288]) if not r['id'].startswith('second_')]
probe = [i for i in range(288) if i not in fit]
tokens = sorted({rows[i]['last_token'] for i in fit})
with np.load(RDIR / 'progress_carrier_states.npz') as z:
    X = torch.from_numpy(z['X'].copy()).double()
    incoming = torch.from_numpy(z['incoming21'].copy()).double()
    F = torch.from_numpy(z['prefix_values'].copy()).double()
    R = torch.from_numpy(z['R'].copy()).double()
with np.load(RDIR / 'progress_carrier_geometry.npz') as z:
    U = torch.from_numpy(z['carrier'].copy()).double()
    B = torch.from_numpy(z['semantic_basis'].copy()).double()
    Pmu = torch.from_numpy(z['mean_projection'].copy()).double()
    centers = torch.from_numpy(z['semantic_centers'].copy()).double()
    alpha = 4 * float(z['alpha'])
with safe_open(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors', framework='pt') as f:
    gain = f.get_tensor('model.layers.23.post_attention_layernorm.weight').double()
P = X / gain
expected = incoming @ U + alpha * F @ Pmu @ B.T
identity = (R @ U - expected).norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-12)
assert float(identity.max()) < 1e-4
progress = P - (P @ U) @ U.T
stage_centers = torch.stack([progress[[i for i in fit if rows[i]['last_token'] == token]].mean(0) for token in tokens])
dual = torch.linalg.pinv(stage_centers, rtol=1e-8)
inverse_rms = ((P @ U) * expected).sum(-1) / expected.square().sum(-1)
inherited = (incoming @ U) * inverse_rms[:, None]
stage_means = torch.stack([inherited[[i for i in fit if rows[i]['last_token'] == token]].mean(0) for token in tokens])
M = U - dual @ stage_means
anchor = P[fit].mean(0)
anchor /= anchor.square().sum()
qmeta = json.loads((RDIR / 'question_layer_diagnostic_states.json').read_text('utf8'))['sequences']
classes = sorted({r['label'] for r in qmeta})


def canonical(fact):
    return {'wood_finder': 'wood_donor'}.get(fact, fact)


def normalize(value):
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)


mapping = {(canonical(r['fact']), r['last_token']): r['next_token'] for r in rows[:288]}
progress_scores = P @ dual
predicted_tokens = [tokens[i] for i in progress_scores.argmax(-1).tolist()]
reports = []
saved = {'readout_unscaled': M.numpy(), 'stage_dual_unscaled': dual.numpy(),
         'stage_centers': stage_centers.numpy(), 'inherited_stage_means': stage_means.numpy(),
         'anchor_unscaled': anchor.numpy(), 'relation_centers': centers.numpy(),
         'progress_scores': progress_scores.numpy()}
for name, relation in [('pure_original', F @ Pmu @ B.T), ('actual_uncorrected', P @ U), ('actual_compensated', P @ M)]:
    scores = normalize(relation) @ normalize(centers).T
    predicted = [classes[i] for i in scores.argmax(-1).tolist()]
    proposals = [mapping.get((predicted[i], predicted_tokens[i])) for i in range(len(rows))]
    saved[name + '_relation_scores'] = scores.numpy()
    comparisons = []
    for split, indices in [('fit', fit), ('probe', probe)]:
        questions = {}
        for i in indices:
            questions.setdefault(rows[i]['id'], []).append(proposals[i] == rows[i]['next_token'])
        comparisons.append({'split': split, 'states': len(indices), 'questions': len(questions),
                            'relation_correct': sum(predicted[i] == canonical(rows[i]['fact']) for i in indices),
                            'progress_correct': sum(predicted_tokens[i] == rows[i]['last_token'] for i in indices),
                            'proposal_correct': sum(proposals[i] == rows[i]['next_token'] for i in indices),
                            'all_steps_correct': sum(all(v) for v in questions.values())})
    native = []
    for kind in sorted({r['kind'] for r in rows[288:]}):
        indices = [i for i in range(288, len(rows)) if rows[i]['kind'] == kind]
        requests = [i for i in indices if proposals[i] is not None]
        eligible = [i for i in indices if rows[i]['last_token'] in tokens]
        native.append({'kind': kind, 'states': len(indices),
                       'classified_outside': sum(predicted[i] == 'outside' for i in indices),
                       'potential_fact_requests': len(requests), 'actual_stage_token_in_set': len(eligible),
                       'stage_identity_correct_when_in_set': sum(predicted_tokens[i] == rows[i]['last_token'] for i in eligible),
                       'requests_with_actual_stage_token': sum(rows[i]['last_token'] in tokens for i in requests)})
    report = {'representation': name, 'comparisons': comparisons, 'native': native}
    reports.append(report)
    print(json.dumps(report), flush=True)
with np.load(RDIR / 'carrier_concept_states_original.npz') as z:
    original_R = torch.from_numpy(z['R'].copy()).double()
shift = (R[:4693] - original_R).norm(dim=-1)
relative_shift = shift / original_R.norm(dim=-1)
changes = {}
for kind in sorted({r['kind'] for r in rows[:4693]}):
    indices = [i for i, r in enumerate(rows[:4693]) if r['kind'] == kind]
    changes[kind] = {'mean_shift': float(shift[indices].mean()),
                     'mean_relative_shift': float(relative_shift[indices].mean()),
                     'max_relative_shift': float(relative_shift[indices].max())}
result = {'representation_reports': reports, 'carrier_identity_max_relative_error': float(identity.max()),
          'residual_changes': changes, 'feature_fit_questions': 50,
          'probe_questions_already_observed': True, 'readout_only': True,
          'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
          'free_generation_performed': False, 'counts_as_correction': False,
          'third_set_used': False, 'final_set_read': False}
np.savez_compressed(RDIR / f'{STEM}.npz', **saved)
(RDIR / f'{STEM}.json').write_text(json.dumps(result, indent=2), encoding='utf8')
print('Residual changes', json.dumps(changes), flush=True)
