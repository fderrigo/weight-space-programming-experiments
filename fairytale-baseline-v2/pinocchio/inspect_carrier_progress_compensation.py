"""Remove predictable token-progress interference from a linear carrier readout.

This is a diagnostic matrix, not a factual decoder or an inference-time hook.
All coefficients use the fifty earlier questions; no loss or search is used.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'carrier_progress_compensation'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(2)
meta = json.loads((RDIR / 'carrier_concept_states.json').read_text('utf8'))
assert meta['completed']
rows = meta['records']
fit = [i for i, r in enumerate(rows[:288]) if not r['id'].startswith('second_')]
probe = [i for i, r in enumerate(rows[:288]) if r['id'].startswith('second_')]
tokens = sorted({rows[i]['last_token'] for i in fit})
with np.load(RDIR / 'semantic_carrier_geometry.npz') as z:
    U = torch.from_numpy(z['carrier'].copy()).double()
    B = torch.from_numpy(z['semantic_basis'].copy()).double()
    Pmu = torch.from_numpy(z['mean_projection'].copy()).double()
    centers = torch.from_numpy(z['semantic_centers'].copy()).double()
    alpha = float(z['alpha'])
with np.load(RDIR / 'carrier_concept_factors.npz') as z:
    stage_dual = torch.from_numpy(z['protected_progress_dual'].copy()).double()
with np.load(RDIR / 'carrier_concept_states_protected.npz') as z:
    X = torch.from_numpy(z['X'].copy()).double()
    incoming = torch.from_numpy(z['incoming21'].copy()).double()
    F = torch.from_numpy(z['prefix_values'].copy()).double()
with safe_open(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors', framework='pt') as f:
    gain = f.get_tensor('model.layers.23.post_attention_layernorm.weight').double()
physical = X / gain
expected = incoming @ U + alpha * F @ Pmu @ B.T
actual = physical @ U
inverse_rms = (actual * expected).sum(-1) / expected.square().sum(-1)
identity_error = float((actual - expected * inverse_rms[:, None]).norm() / actual.norm())
assert identity_error < 1e-5 and (inverse_rms > 0).all()
inherited = (incoming @ U) * inverse_rms[:, None]
stage_means = torch.stack([inherited[[i for i in fit if rows[i]['last_token'] == token]].mean(0) for token in tokens])
M = U - stage_dual @ stage_means
physical_readout = (M / gain[:, None]).float()
adjusted = X.float() @ physical_readout
reference = physical @ M
physical_error = float((adjusted.double() - reference).norm() / reference.norm())
assert physical_error < 1e-5
qmeta = json.loads((RDIR / 'question_layer_diagnostic_states.json').read_text('utf8'))
classes = sorted({r['label'] for r in qmeta['sequences']})


def normalize(value):
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def canonical(fact):
    return {'wood_finder': 'wood_donor'}.get(fact, fact)


mapping = {(canonical(row['fact']), row['last_token']): row['next_token'] for row in rows[:288]}
progress_scores = physical @ stage_dual
progress_predictions = [tokens[i] for i in progress_scores.argmax(-1).tolist()]
reports = []
arrays = {'readout_unscaled': M.numpy(), 'readout_physical_f32': physical_readout.numpy(),
          'stage_dual_unscaled': stage_dual.numpy(), 'inherited_stage_means': stage_means.numpy()}
for name, values in [('uncorrected', actual), ('progress_compensated', adjusted.double())]:
    scores = normalize(values) @ normalize(centers).T
    predictions = [classes[i] for i in scores.argmax(-1).tolist()]
    proposals = [mapping.get((predictions[i], progress_predictions[i])) for i in range(len(rows))]
    arrays[f'{name}_relation_scores'] = scores.float().numpy()
    split_reports = []
    for split, selected in [('earlier_fifty', fit), ('second_observed', probe)]:
        questions = {}
        for i in selected:
            questions.setdefault(rows[i]['id'], []).append(proposals[i] == rows[i]['next_token'])
        split_reports.append({'split': split, 'states': len(selected),
                              'relation_correct': sum(predictions[i] == canonical(rows[i]['fact']) for i in selected),
                              'proposed_token_correct': sum(proposals[i] == rows[i]['next_token'] for i in selected),
                              'whole_concept_labels_correct': sum(all(v) for v in questions.values()),
                              'questions': len(questions),
                              'mistakes': [{'id': rows[i]['id'], 'subtoken': rows[i]['subtoken'],
                                           'expected': canonical(rows[i]['fact']), 'predicted': predictions[i]}
                                          for i in selected if proposals[i] != rows[i]['next_token']]})
    native = []
    for kind in sorted({r['kind'] for r in rows[288:]}):
        selected = [i for i in range(288, len(rows)) if rows[i]['kind'] == kind]
        native.append({'kind': kind, 'states': len(selected),
                       'potential_fact_requests_without_threshold': sum(proposals[i] is not None for i in selected)})
    reports.append({'representation': name, 'splits': split_reports, 'native': native})
    print(name, [{k: v for k, v in r.items() if k != 'mistakes'} for r in split_reports], native, flush=True)
report = {'decision_by': 'Codex', 'formula': 'M = U - D_stage T, where D_stage is the fixed dual of mean progress states and T is the per-progress-token mean of inherited carrier content after RMS scaling. Readout on physical MLP inputs is diag(g)^-1 M.',
          'identity_relative_error': identity_error, 'physical_f32_readout_relative_error': physical_error,
          'fit_questions': 50, 'probe_questions_already_observed': 24,
          'source_sha256': hashlib.sha256((RDIR / 'carrier_concept_states.json').read_bytes()).hexdigest(),
          'reports': reports, 'weights_changed': False, 'training': False, 'gradients': False,
          'optimizer': False, 'parameter_search': False, 'runtime_routing': False,
          'counts_as_correction': False, 'third_set_used': False, 'final_set_read': False}
np.savez_compressed(RDIR / f'{STEM}.npz', **arrays)
(RDIR / f'{STEM}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
