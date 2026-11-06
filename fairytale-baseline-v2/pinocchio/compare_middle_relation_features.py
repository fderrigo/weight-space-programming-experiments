"""Compare already collected native features with the same fixed prototype rule."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'middle_relation_features'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


meta = json.loads((RDIR / 'middle_fact_retrieval.json').read_text('utf8'))
assert meta['completed']
path = RDIR / 'middle_fact_retrieval_states.npz'
assert sha(path) == meta['arrays_sha256']
with np.load(path) as z:
    X, Q, H = [torch.from_numpy(z[k].copy()) for k in ['X', 'Q', 'heads']]
with safe_open(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors', framework='pt') as f:
    O = f.get_tensor('model.layers.14.self_attn.o_proj.weight').float()
features = [('normalized_input', X), ('all_queries', Q), ('all_head_values', H), ('attention_output', H @ O.T)]
features.extend((f'head_value_{j}', H[:, j * 64:(j + 1) * 64]) for j in range(14))
spec = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
        'hypothesis': 'Single query heads carry only6-14 of24 observed relation labels correctly; test whether information is distributed across the full input/query vector or in context read by attention values before choosing an intervention.',
        'representations': [name for name, _ in features], 'prototype_rule': 'Same48 factual first states,29 outside first states, centered normalized class means, no fitted metric or parameter search.',
        'source_sha256': sha(RDIR / 'middle_fact_retrieval.json'), 'arrays_sha256': meta['arrays_sha256'],
        'code_sha256': sha(Path(__file__)), 'training': False, 'gradients': False, 'optimizer': False,
        'new_weights': False, 'free_generation': False, 'counts_as_correction': False, 'reserved_final_used': False}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
classes, records = meta['classes'], meta['records']
fit, probe = meta['fit_indices'], meta['observed_probe_indices']
outside_fit, outside_probe = meta['outside_fit_indices'], meta['outside_probe_indices']
fit_labels = torch.tensor([classes.index(records[i]['fact']) for i in fit])
probe_labels = torch.tensor([classes.index(records[i]['fact']) for i in probe])
roles = {r['id']: r for r in meta['role_prefixes']}
role_indices = [s['offset'] for s in meta['sequences'] if s['source_group'] == 'role_binding' and not s['actual_history']]
results, arrays = [], {}
for name, values in features:
    centers = torch.stack([values[[i for i in fit if records[i]['fact'] == fact]].mean(0) for fact in classes[:-1]]
                          + [values[outside_fit].mean(0)])
    mean = centers.mean(0)
    centers = centers - mean
    centers = centers / centers.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    centered = values - mean
    centered = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scores = centered @ centers.T
    predictions = scores.argmax(-1)
    role_rows = []
    for i in role_indices:
        original_id = records[i]['original_id']
        expected = roles[original_id]['expected_class_if_in_schema']
        role_rows.append({'id': original_id, 'expected_if_in_schema': expected,
                          'predicted': classes[int(predictions[i])],
                          'correct_if_in_schema': None if expected is None else classes[int(predictions[i])] == expected})
    known = [r for r in role_rows if r['expected_if_in_schema'] not in [None, 'outside']]
    external = [r for r in role_rows if r['expected_if_in_schema'] == 'outside']
    result = {'representation': name, 'fit_correct': int((predictions[fit] == fit_labels).sum()), 'fit_total': len(fit),
              'observed_query_probe_correct': int((predictions[probe] == probe_labels).sum()), 'query_probe_total': len(probe),
              'outside_probe_rejected': int((predictions[outside_probe] == 12).sum()), 'outside_probe_total': len(outside_probe),
              'known_role_prefix_correct': sum(r['correct_if_in_schema'] for r in known), 'known_role_prefix_total': len(known),
              'external_role_prefix_rejected': sum(r['correct_if_in_schema'] for r in external), 'external_role_prefix_total': len(external),
              'role_prefixes': role_rows}
    results.append(result)
    arrays[name + '_centers'] = centers.numpy()
    arrays[name + '_mean'] = mean.numpy()
    arrays[name + '_predictions'] = predictions.numpy()
    print(name, 'query', result['observed_query_probe_correct'], '/24', 'outside', result['outside_probe_rejected'], '/29',
          'role_prefix', result['known_role_prefix_correct'], '/', len(known), flush=True)
path = RDIR / f'{STEM}.npz'
np.savez_compressed(path, **arrays)
(RDIR / f'{STEM}.json').write_text(json.dumps({'spec': spec, 'results': results,
    'arrays_sha256': sha(path), 'completed': True, 'candidate_selected_or_constructed': False}, indent=2), encoding='utf8')
