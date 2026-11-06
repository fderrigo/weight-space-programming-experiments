"""Separate erasure, removed neurons and dense compensation on native prefixes."""
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
STEM = 'late_native_components'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
meta = json.loads((RDIR / 'progress_carrier_states.json').read_text('utf8'))
rows = meta['records']
actual_responses = json.loads((RDIR / 'late_selector_fact_screen.json').read_text('utf8'))
old_responses = {r['id']: r for r in json.loads((RDIR / 'attention_gain_screen.json').read_text('utf8')) if r['stage'] == 'original'}
checks = []
for sequence in meta['sequences']:
    if sequence['kind'] in ['external', 'minimal_native']:
        checks.extend([sequence['offset'], sequence['offset'] + sequence['states'] - 1])
divergences = []
for new in actual_responses:
    if new['group'] != 'control':
        continue
    old = old_responses[new['id']]
    common = next((j for j, (a, b) in enumerate(zip(old['generated_ids'], new['generated_ids'])) if a != b), min(len(old['generated_ids']), len(new['generated_ids'])))
    if common == min(len(old['generated_ids']), len(new['generated_ids'])):
        continue
    sequence = next(s for s in meta['sequences'] if s['id'] == new['id'] and s['kind'] == 'external')
    pos = sequence['positions'][0] + common
    assert sequence['ids'][sequence['positions'][0] + 1:pos + 1] == new['generated_ids'][:common]
    index = sequence['offset'] + sequence['positions'].index(pos)
    checks.append(index)
    divergences.append({'id': new['id'], 'index': index, 'shared_generated_tokens': common,
                        'original_next': old['generated_ids'][common], 'edited_next': new['generated_ids'][common]})
checks = sorted(set(checks))
history = {}
for sequence in meta['sequences']:
    for local, pos in enumerate(sequence['positions']):
        i = sequence['offset'] + local
        if i in checks:
            history[i] = sequence['ids'][:pos + 1]
with np.load(RDIR / 'late_selector_states.npz') as z:
    X, A, R = [torch.from_numpy(z[k][checks].copy()) for k in ['X', 'A', 'R']]
with np.load(RDIR / 'carrier_concept_states_original.npz') as z:
    original = torch.from_numpy(z['R'][checks].copy())
with np.load(RDIR / 'late_selector_geometry.npz') as z:
    B = torch.from_numpy(z['protected_carrier'].copy()).float()
basis = load_file(ROOT / 'patches/late_selector_fact_basis.safetensors')
diagnostic = json.loads((RDIR / 'late_selector_fact_diagnostic.json').read_text('utf8'))
chosen = basis['reassigned_neurons']
fact_ids = chosen[:diagnostic['fact_neurons']]
erase_ids = chosen[diagnostic['fact_neurons']:]
remaining = torch.tensor([i for i in range(4864) if i not in chosen])
parts = torch.nn.functional.silu(X @ basis['gate_after'][chosen].T) * (X @ basis['up_after'][chosen].T)
fact = parts[:, :len(fact_ids)] @ basis['down_after'][:, fact_ids].T
erase = parts[:, len(fact_ids):] @ basis['down_after'][:, erase_ids].T
removed = A[:, chosen] @ basis['down_before'][:, chosen].T
compensation = A[:, remaining] @ (basis['down_after'][:, remaining] - basis['down_before'][:, remaining]).T
ideal_erase = -(R @ B) @ B.T
full = R - removed + erase + compensation + fact
conditions = {'original': original, 'parent_ideal_erasure': R + ideal_erase,
              'no_dense_compensation': R - removed + erase + fact,
              'ideal_erasure_no_dense_compensation': R - removed + ideal_erase + fact,
              'full_with_ideal_erasure': R - removed + ideal_erase + compensation + fact,
              'full': full}
model_dir = ROOT.parent.parent / 'little-red-weightspace/models/original'
t = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
with safe_open(model_dir / 'model.safetensors', framework='pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
predictions = {}
for name, residual in conditions.items():
    answers = []
    for start in range(0, len(checks), 16):
        block = residual[start:start + 16]
        scores = (block / (block.square().mean(-1) + 1e-6).sqrt()[:, None]) @ E.T
        for j, score in enumerate(scores):
            seen = sorted(set(history[checks[start + j]]))
            score[seen] = torch.where(score[seen] < 0, score[seen] * 1.1, score[seen] / 1.1)
            answers.append(int(score.argmax()))
    predictions[name] = answers
for case in divergences:
    i = checks.index(case['index'])
    case['predictions'] = {name: {'token': v[i], 'text': t.decode([v[i]])} for name, v in predictions.items()}
    case['component_norms'] = {name: float(component[i].norm()) for name, component in [('fact', fact), ('erasure_error', erase - ideal_erase), ('removed_native_neurons', removed), ('dense_compensation', compensation)]}
    case['matches_actual_original_and_edit'] = predictions['original'][i] == case['original_next'] and predictions['full'][i] == case['edited_next']
report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'hypothesis': 'The preceding local check showed zero fact contribution and ideal erasure recovering the correct arithmetic token. Separate the approximate erasure and dense native compensation before changing the transport again.',
          'selection_rule': 'First and last recorded position of each external or minimal_native history plus the actual first divergence of each of the four screen controls.',
          'positions': len(checks), 'record_indices': checks,
          'same_original_tokens': {name: sum(a == b for a, b in zip(values, predictions['original'])) for name, values in predictions.items()},
          'control_divergences': divergences,
          'all_control_replays_match': all(r['matches_actual_original_and_edit'] for r in divergences),
          'source_sha256': hashlib.sha256((RDIR / 'late_selector_fact_screen.json').read_bytes()).hexdigest(),
          'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'diagnostic_algebra_only': True, 'new_weights_modified': False, 'free_generation_performed': False,
          'training': False, 'gradients': False, 'optimizer': False, 'final_set_read': False}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k != 'record_indices'}, flush=True)
assert report['all_control_replays_match']
