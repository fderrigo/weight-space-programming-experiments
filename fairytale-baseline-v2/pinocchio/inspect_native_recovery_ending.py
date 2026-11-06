"""Explain the new natural break in a previously looping observed answer."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'native_recovery_ending_analysis'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(2)


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


old_response = next(r for r in read('late_tail_relations_third_observed.json') if r['id'] == 'third_ending_0')
new_response = next(r for r in read('late_native_recovery_third_observed.json') if r['id'] == 'third_ending_0')
assert old_response['done_reason'] == 'length' and new_response['stop_token'] == 151645
divergence = next(i for i, (a, b) in enumerate(zip(old_response['generated_ids'], new_response['generated_ids'])) if a != b)
assert divergence == 13 and old_response['generated_ids'][13] == 11 and new_response['generated_ids'][13] == 13
meta = read('third_mechanism_states.json')
seq = next(s for s in meta['sequences'] if s['kind'] == 'actual_response' and s['id'] == 'third_ending_0')
assert seq['ids'][seq['base_length']:seq['base_length'] + divergence] == new_response['generated_ids'][:divergence]
subtokens = [6, 8, divergence]
indices = [seq['offset'] + j for j in subtokens]
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'hypothesis': 'Native-control restoration unexpectedly changes a repeated-answer comma to a period, allowing native stopping. Compare the common history at the first comma, the repeated child word and the first new punctuation divergence.',
    'record_indices': indices, 'subtokens': subtokens,
    'supplied_histories_are_actual_shared_generated_prefixes': True,
    'reference_original_limit': 'Original residuals are counterfactual on these edited prefixes; no claim the original would generate the same answer.',
    'old_response_sha256': hashlib.sha256(json.dumps(old_response, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
    'new_response_sha256': hashlib.sha256(json.dumps(new_response, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
    'source_states_sha256': sha(RDIR / 'third_mechanism_states.json'),
    'old_basis_sha256': sha(ROOT / 'patches/late_tail_relations_basis.safetensors'),
    'new_basis_sha256': sha(ROOT / 'patches/late_native_recovery_basis.safetensors'),
    'code_sha256': sha(Path(__file__)),
    'weights_changed': False, 'new_generation': False, 'new_correction_claimed': False,
    'training': False, 'gradients': False, 'optimizer': False, 'final_set_read': False,
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
with np.load(RDIR / 'third_mechanism_states_late_selector_parent.npz') as z:
    X, A, parent_R = [torch.from_numpy(z[k][indices].copy()) for k in ['X', 'A', 'R']]
with np.load(RDIR / 'third_mechanism_states_original.npz') as z:
    original_R = torch.from_numpy(z['R'][indices].copy())
old = load_file(ROOT / 'patches/late_tail_relations_basis.safetensors')
new = load_file(ROOT / 'patches/late_native_recovery_basis.safetensors')
assert torch.equal(old['gate_after'], new['gate_after']) and torch.equal(old['up_after'], new['up_after'])
chosen = old['reassigned_neurons']
new_A = A.clone()
new_A[:, chosen] = torch.nn.functional.silu(X @ old['gate_after'][chosen].T) * (X @ old['up_after'][chosen].T)
base_R = parent_R - A @ old['down_before'].T
old_R = base_R + new_A @ old['down_after'].T
new_R = base_R + new_A @ new['down_after'].T
update = new_A.double() @ (new['down_after'] - old['down_after']).double().T
with np.load(RDIR / 'late_native_recovery_equations.npz') as z:
    V = torch.from_numpy(z['values'].copy()).double()
coefficients = update @ torch.linalg.pinv(V, rtol=1e-6)
reconstruction_error = float((coefficients @ V - update).norm() / update.norm())
assert reconstruction_error < 1e-4
model_dir = ROOT.parent.parent / 'little-red-weightspace/models/original'
t = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
with safe_open(model_dir / 'model.safetensors', framework='pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
fact_count = read('late_native_recovery_diagnostic.json')['fact_neurons']
fact_ids = chosen[:fact_count]
without_fact = new_R - new_A[:, fact_ids] @ new['down_after'][:, fact_ids].T
recovery = read('native_recovery_states.json')
repair_indices = read('late_native_recovery_diagnostic.json')['spec']['repair_record_indices']
points = []
for j, (index, subtoken) in enumerate(zip(indices, subtokens)):
    position = seq['positions'][subtoken]
    seen = sorted(set(seq['ids'][:position + 1]))
    predictions = {}
    for name, residual in [('original_on_same_history', original_R[j]), ('before_native_recovery', old_R[j]),
                           ('after_native_recovery', new_R[j]), ('after_without_fact_columns', without_fact[j])]:
        score = (residual / (residual.square().mean() + 1e-6).sqrt()) @ E.T
        score[seen] = torch.where(score[seen] < 0, score[seen] * 1.1, score[seen] / 1.1)
        top = score.topk(5)
        predictions[name] = {'top': [{'token': int(k), 'text': t.decode([int(k)]), 'score': float(s)}
                                    for k, s in zip(top.indices, top.values)],
                             'period_minus_comma_margin': float(score[13] - score[11])}
    assert predictions['before_native_recovery']['top'][0]['token'] == old_response['generated_ids'][subtoken]
    assert predictions['after_native_recovery']['top'][0]['token'] == new_response['generated_ids'][subtoken]
    factors = []
    for token in [13, 11]:
        raw_before, raw_after = float(old_R[j] @ E[token]), float(new_R[j] @ E[token])
        assert (raw_before > 0) == (raw_after > 0)
        factors.append((1.1 if raw_before < 0 else 1 / 1.1) if token in seen else 1.0)
    direction = (factors[0] * E[13] - factors[1] * E[11]).double()
    contributions = coefficients[j] * (V @ direction)
    actual_shift = float(update[j] @ direction)
    assert abs(float(contributions.sum()) - actual_shift) < 1e-4
    attribution = [{'source_id': recovery['records'][i]['id'],
                    'source_subtoken': recovery['records'][i]['subtoken'],
                    'raw_period_minus_comma_shift': float(v)} for i, v in zip(repair_indices, contributions)]
    points.append({'subtoken': subtoken,
                   'actual_shared_response_prefix': t.decode(seq['ids'][seq['base_length']:position + 1]),
                   'predictions': predictions, 'update_norm': float(update[j].norm()),
                   'raw_period_minus_comma_shift': actual_shift,
                   'restored_state_contributions': sorted(attribution, key=lambda r: -abs(r['raw_period_minus_comma_shift']))})
report = {'spec': spec, 'points': points, 'value_span_reconstruction_relative_error': reconstruction_error,
          'interpretation_limit': 'Exact local additive decomposition of this weight difference. It does not show that removing a single source equation would improve complete generation. The subsequent stop was observed in ordinary generation but is not included in the older saved common-history states.'}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
for p in points:
    print('POINT', p['subtoken'], repr(p['actual_shared_response_prefix']),
          {k: (v['top'][0]['text'], v['period_minus_comma_margin']) for k, v in p['predictions'].items()}, flush=True)
    print('Largest native-state contributions', p['restored_state_contributions'][:4], flush=True)
