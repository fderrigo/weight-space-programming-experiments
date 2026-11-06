"""Measure a fixed relation-gated feature family before editing any model."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from carrier_feature_maps import ROOT, RDIR, canonical, load_maps, relation_gated_features

torch.set_grad_enabled(False)
torch.set_num_threads(4)
stem = 'relation_gated_carrier_features'
assert not (RDIR / f'{stem}.json').exists()
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'decision_by': 'Codex',
    'decision': 'Swap the native SwiGLU factors: bounded nonlinear relation confidence multiplied by linear preceding-token identity. Fixed relation ramp from 0.5 to 0.75 of the class-center response; steepness 640. No threshold search.',
    'reason': 'The preceding progress-gated candidate propagated concept writes to unrelated questions. Test whether a nonlinear relation gate suppresses this before any output compensation.',
    'diagnostic_only': True, 'model_weights_modified': False,
    'training': False, 'gradients': False, 'optimizer': False,
    'third_set_used': False, 'final_set_read': False,
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'feature_code_sha256': hashlib.sha256((ROOT / 'carrier_feature_maps.py').read_bytes()).hexdigest(),
}
(RDIR / f'{stem}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
m = load_maps()
G, U, new = relation_gated_features(m)
rows, fit, cells = m['meta']['records'], m['fit'], m['cells']
probe = [i for i in range(288) if i not in fit]
old_basis = load_file(ROOT / 'patches/factorized_carrier_basis.safetensors')
chosen = old_basis['reassigned_neurons']
old_raw = torch.nn.functional.silu(m['X'] @ old_basis['gate_after'][chosen].T) * (m['X'] @ old_basis['up_after'][chosen].T)
old = old_raw[:, 0::3] - old_raw[:, 1::3] - old_raw[:, 2::3]
old_meta = json.loads((RDIR / 'factorized_carrier_diagnostic.json').read_text('utf8'))
assert old_meta['cells'] == cells
anchor_values = m['P'] @ m['anchor']
relation_scores = (m['P'] @ m['relation']) / anchor_values[:, None]
progress_scores = (m['P'] @ m['progress']) / anchor_values[:, None]
indices_by_kind = {}
for i, row in enumerate(rows):
    indices_by_kind.setdefault(row['kind'], []).append(i)
report = {'states': len(rows), 'cells': len(cells), 'fit_states': len(fit),
          'probe_states': len(probe), 'fixed_relation_thresholds': [.5, .75],
          'families': {}, 'relation_diagnostic': {}}
for name, F in [('progress_gated_previous', old), ('relation_gated_new', new)]:
    scales = torch.stack([F[cell['members'], j].abs().median() for j, cell in enumerate(cells)]).clamp_min(1e-8)
    normalized = F / scales
    family = {'positive_fit_median_scales': scales.tolist(), 'groups': {}, 'coverage': {}}
    for kind, indices in indices_by_kind.items():
        values = normalized[indices].abs()
        family['groups'][kind] = {
            'states': len(indices), 'raw_mean_l2': float(F[indices].norm(dim=1).mean()),
            'normalized_mean_l2': float(values.norm(dim=1).mean()),
            'normalized_max_abs': float(values.max()),
            'states_with_any_cell_above_half': int((values.max(dim=1).values > .5).sum()),
            'states_with_any_cell_above_one': int((values.max(dim=1).values > 1).sum()),
        }
    for split, indices in [('fit', fit), ('probe', probe)]:
        diagnostics = []
        for i in indices:
            row = rows[i]
            j = next(j for j, c in enumerate(cells) if (c['fact'], c['previous_token'], c['target_token']) == (canonical(row['fact']), row['last_token'], row['next_token']))
            diagnostics.append({'id': row['id'], 'subtoken': row['subtoken'],
                                'correct_cell': j, 'correct_normalized_activation': float(normalized[i, j]),
                                'largest_cell': int(normalized[i].abs().argmax())})
        ids = sorted({r['id'] for r in diagnostics})
        family['coverage'][split] = {
            'correct_cell_above_half': sum(r['correct_normalized_activation'] > .5 for r in diagnostics),
            'correct_cell_is_largest': sum(r['correct_cell'] == r['largest_cell'] for r in diagnostics),
            'whole_paths_above_half': sum(all(r['correct_normalized_activation'] > .5 for r in diagnostics if r['id'] == q) for q in ids),
            'questions': len(ids), 'states': len(indices), 'details': diagnostics,
        }
    report['families'][name] = family
for kind, indices in indices_by_kind.items():
    factual_columns = [j for j, c in enumerate(m['classes']) if c != 'outside']
    values = relation_scores[indices][:, factual_columns]
    report['relation_diagnostic'][kind] = {
        'states': len(indices), 'max_factual_score_median': float(values.max(dim=1).values.median()),
        'states_over_half': int((values.max(dim=1).values > .5).sum()),
        'states_over_three_quarters': int((values.max(dim=1).values > .75).sum()),
    }
report.update({'training': False, 'free_generation_performed': False,
               'interpretation': 'Activation coverage is not answer correctness. Native groups include incorrect model continuations and are not labeled as correct answers.',
               'model_weights_modified': False, 'final_set_read': False})
(RDIR / f'{stem}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
np.savez(RDIR / f'{stem}.npz', gate_rows=G.numpy(), up_rows=U.numpy(),
         new_features=new.numpy(), previous_features=old.numpy(), relation_scores=relation_scores.numpy(),
         progress_scores=progress_scores.numpy(), anchor_values=anchor_values.numpy())
for name, family in report['families'].items():
    print(name, json.dumps({'groups': family['groups'], 'coverage': {k: {a: b for a, b in v.items() if a != 'details'} for k, v in family['coverage'].items()}}), flush=True)
print('relation', json.dumps(report['relation_diagnostic']), flush=True)
