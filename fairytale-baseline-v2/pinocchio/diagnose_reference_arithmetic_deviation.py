"""Locate the first changed token on the actually shared arithmetic prefix."""
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
STEM = 'reference_arithmetic_deviation'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
source = RDIR / 'reference_selector_fact_screen.json'
new = next(r for r in json.loads(source.read_text('utf8')) if r['id'] == 'second_control_10')
old = next(r for r in json.loads((RDIR / 'attention_gain_screen.json').read_text('utf8')) if r['id'] == new['id'] and r['stage'] == 'original')
common = next((j for j, (a, b) in enumerate(zip(old['generated_ids'], new['generated_ids'])) if a != b), min(len(old['generated_ids']), len(new['generated_ids'])))
meta = json.loads((RDIR / 'progress_carrier_states.json').read_text('utf8'))
sequence = next(s for s in meta['sequences'] if s['id'] == new['id'] and s['kind'] == 'external')
pos0 = sequence['positions'][0]
assert sequence['ids'][pos0 + 1:pos0 + 1 + common] == new['generated_ids'][:common]
local = sequence['positions'].index(pos0 + common)
index = sequence['offset'] + local
history = sequence['ids'][:pos0 + common + 1]
with np.load(RDIR / 'selector_reference_states.npz') as z:
    X = torch.from_numpy(z['X'][index:index + 1].copy())
    A = torch.from_numpy(z['A'][index:index + 1].copy())
    parent_R = torch.from_numpy(z['R'][index:index + 1].copy())
with np.load(RDIR / 'carrier_concept_states_original.npz') as z:
    original_R = torch.from_numpy(z['R'][index:index + 1].copy())
with np.load(RDIR / 'selector_reference_geometry.npz') as z:
    B = torch.from_numpy(z['protected_carrier'].copy()).float()
basis = load_file(ROOT / 'patches/reference_selector_fact_basis.safetensors')
diagnostic = json.loads((RDIR / 'reference_selector_fact_diagnostic.json').read_text('utf8'))
chosen = basis['reassigned_neurons']
facts = chosen[:diagnostic['fact_neurons']]
erasure = chosen[diagnostic['fact_neurons']:]
new_A = A.clone()
new_A[:, chosen] = torch.nn.functional.silu(X @ basis['gate_after'][chosen].T) * (X @ basis['up_after'][chosen].T)
full_R = parent_R - A @ basis['down_before'].T + new_A @ basis['down_after'].T
fact_component = new_A[:, facts] @ basis['down_after'][:, facts].T
erase_component = new_A[:, erasure] @ basis['down_after'][:, erasure].T
conditions = {'original': original_R, 'modified_parent': parent_R,
              'parent_exact_carrier_removal': parent_R - (parent_R @ B) @ B.T,
              'full_patch_without_fact_cells': full_R - fact_component,
              'full_patch': full_R}
model_dir = ROOT.parent.parent / 'little-red-weightspace/models/original'
t = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
with safe_open(model_dir / 'model.safetensors', framework='pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
predictions = []
seen = sorted(set(history))
for name, residual in conditions.items():
    scores = ((residual / (residual.square().mean(-1) + 1e-6).sqrt()[:, None]) @ E.T)[0]
    scores[seen] = torch.where(scores[seen] < 0, scores[seen] * 1.1, scores[seen] / 1.1)
    indices = scores.topk(5).indices.tolist()
    predictions.append({'condition': name, 'top_token': indices[0],
                        'top_five': [{'id': i, 'text': t.decode([i]), 'score': float(scores[i])} for i in indices]})
assert predictions[0]['top_token'] == old['generated_ids'][common]
assert predictions[-1]['top_token'] == new['generated_ids'][common]
report = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'question_id': new['id'],
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'record_index': index, 'shared_generated_tokens': common,
    'shared_prefix': t.decode(new['generated_ids'][:common]),
    'original_response': old['response'], 'edited_response': new['response'],
    'predictions': predictions,
    'fact_cell_contribution_norm': float(fact_component.norm()),
    'erasure_contribution_norm': float(erase_component.norm()),
    'parent_residual_change_norm': float((parent_R - original_R).norm()),
    'full_residual_change_norm': float((full_R - original_R).norm()),
    'matches_original_and_edited_first_divergence': True,
    'interpretation_limit': 'This localizes the first changed token on an identical prefix. A different first token could still lead to a correct paraphrase; it does not alone explain the later wrong number three.',
    'new_weights_modified': False, 'new_generation': False,
    'training': False, 'gradients': False, 'optimizer': False, 'final_set_read': False,
}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print(json.dumps(report, ensure_ascii=False), flush=True)
