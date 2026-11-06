"""Decompose the lost native continuation after the correct Azzurri unit."""
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
STEM = 'prototype_native_continuation'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
meta = json.loads((RDIR / 'progress_carrier_states.json').read_text('utf8'))
sequence = next(r for r in meta['sequences'] if r['kind'] == 'minimal_native' and r['id'] == 'second_fairy_0')
indices = list(range(sequence['offset'], sequence['offset'] + sequence['states']))
assert len(indices) == 2
with np.load(RDIR / 'progress_carrier_states.npz') as z:
    X = torch.from_numpy(z['X'][indices].copy())
    A = torch.from_numpy(z['A'][indices].copy())
    parent_R = torch.from_numpy(z['R'][indices].copy())
with np.load(RDIR / 'carrier_concept_states_original.npz') as z:
    original_R = torch.from_numpy(z['R'][indices].copy())
basis_path = ROOT / 'patches/progress_prototype_carrier_basis.safetensors'
basis = load_file(basis_path)
chosen = basis['reassigned_neurons']
mask = torch.ones(4864, dtype=torch.bool)
mask[chosen] = False
remaining = mask.nonzero().flatten()
new = torch.nn.functional.silu(X @ basis['gate_after'][chosen].T) * (X @ basis['up_after'][chosen].T)
old_selected = A[:, chosen] @ basis['down_before'][:, chosen].T
cell_output = new @ basis['down_after'][:, chosen].T
dense_change = A[:, remaining] @ (basis['down_after'][:, remaining] - basis['down_before'][:, remaining]).T
conditions = {'original': original_R, 'parent': parent_R,
              'parent_plus_cells': parent_R - old_selected + cell_output,
              'parent_plus_dense': parent_R + dense_change,
              'complete_patch': parent_R - old_selected + cell_output + dense_change}
model_dir = ROOT.parent.parent / 'little-red-weightspace/models/original'
t = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
with safe_open(model_dir / 'model.safetensors', framework='pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
actual = next(r for r in json.loads((RDIR / 'progress_prototype_carrier_screen.json').read_text('utf8')) if r['id'] == sequence['id'])
concept = [32, 10400, 324, 461]
assert actual['generated_ids'][:4] == concept
assert sequence['ids'][sequence['positions'][0] - 3:sequence['positions'][0] + 1] == concept
reference = [13, 151645]
reports = []
for name, R in conditions.items():
    scores = (R / (R.square().mean(-1) + 1e-6).sqrt()[:, None]) @ E.T
    notes = []
    for j, position in enumerate(sequence['positions']):
        seen = sorted(set(sequence['ids'][:position + 1]))
        scores[j, seen] = torch.where(scores[j, seen] < 0, scores[j, seen] * 1.1, scores[j, seen] / 1.1)
        values, ids = scores[j].topk(5)
        notes.append({'last_token': sequence['ids'][position], 'prefix_ends_with': t.decode(sequence['ids'][max(0, position - 5):position + 1]),
                      'reference_next_token': reference[j],
                      'reference_rank': int((scores[j] > scores[j, reference[j]]).sum()) + 1,
                      'top_tokens': [{'id': token, 'piece': t.decode([token]), 'logit': value} for token, value in zip(ids.tolist(), values.tolist())]})
    reports.append({'condition': name, 'positions': notes})
    print(name, [(n['reference_rank'], n['top_tokens'][:2]) for n in notes], flush=True)
assert reports[0]['positions'][0]['top_tokens'][0]['id'] == 13
assert reports[-1]['positions'][0]['top_tokens'][0]['id'] == actual['generated_ids'][4]
report = {'case': sequence['id'], 'conditions': reports,
          'original_selected_output_norm': old_selected.norm(dim=-1).tolist(),
          'new_cell_output_norm': cell_output.norm(dim=-1).tolist(),
          'remaining_dense_change_norm': dense_change.norm(dim=-1).tolist(),
          'parent_change_norm': (parent_R - original_R).norm(dim=-1).tolist(),
          'first_position_matches_actual_generated_prefix_and_next_token': True,
          'second_position_counterfactual_native_period_prefix': True,
          'basis_sha256': hashlib.sha256(basis_path.read_bytes()).hexdigest(),
          'training': False, 'gradients': False, 'optimizer': False,
          'weights_modified': False, 'EOS_programmed': False, 'final_set_read': False,
          'interpretation': 'This diagnostic measures the native continuation that was lost. It does not prescribe punctuation or EOS, and does not attribute the full repeated sequence to one position alone.'}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k != 'conditions'}, flush=True)
