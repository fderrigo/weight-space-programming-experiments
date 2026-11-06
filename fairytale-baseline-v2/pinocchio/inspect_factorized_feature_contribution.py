"""Analytic ablation at the final linear projection on existing factual states.

No generation or checkpoint edit: quantify whether the new cells actually
carry factual corrections, rather than attributing a dense write to them.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'factorized_carrier_feature_contribution'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(2)
basis_path = ROOT / 'patches/factorized_carrier_basis.safetensors'
basis = load_file(basis_path)
chosen = basis['reassigned_neurons']
with np.load(RDIR / 'carrier_concept_states_protected.npz') as z:
    X = torch.from_numpy(z['X'][:288].copy())
    A = torch.from_numpy(z['A'][:288].copy())
    R = torch.from_numpy(z['R'][:288].copy())
with np.load(RDIR / 'carrier_concept_states_original.npz') as z:
    original_R = torch.from_numpy(z['R'][:288].copy())
new_A = A.clone()
new_A[:, chosen] = torch.nn.functional.silu(X @ basis['gate_after'][chosen].T) * (X @ basis['up_after'][chosen].T)
full_R = R - A @ basis['down_before'].T + new_A @ basis['down_after'].T
contribution = new_A[:, chosen] @ basis['down_after'][:, chosen].T
without_cells = full_R - contribution
meta = json.loads((RDIR / 'minimal_fact_states.json').read_text('utf8'))['records']
with safe_open(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors', framework='pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
reports = []
for name, states in [('original', original_R), ('complete_patch', full_R), ('final_cell_outputs_zeroed', without_cells)]:
    notes = []
    for begin in range(0, 288, 16):
        rb = states[begin:begin + 16]
        rms = (rb.square().mean(-1) + 1e-6).sqrt()
        logits = (rb / rms[:, None]) @ E.T
        for j, scores in enumerate(logits):
            row = meta[begin + j]
            seen = sorted(set(row['history']))
            scores[seen] = torch.where(scores[seen] < 0, scores[seen] * 1.1, scores[seen] / 1.1)
            notes.append({'id': row['question'], 'subtoken': row['subtoken'],
                          'desired_token_rank': int((scores > scores[row['token']]).sum()) + 1})
    by_question = {}
    for row in notes:
        by_question.setdefault(row['id'], []).append(row['desired_token_rank'] == 1)
    result = {'condition': name, 'desired_token_first': sum(r['desired_token_rank'] == 1 for r in notes),
              'first_concept_token_first': sum(r['desired_token_rank'] == 1 for r in notes if r['subtoken'] == 0),
              'all_concept_tokens_first': sum(all(v) for v in by_question.values()),
              'factual_states': len(notes), 'questions': len(by_question), 'tokens': notes}
    reports.append(result)
    print({k: v for k, v in result.items() if k != 'tokens'}, flush=True)
assert reports[1]['desired_token_first'] == 288
report = {'basis_sha256': hashlib.sha256(basis_path.read_bytes()).hexdigest(),
          'method': 'Reconstruct the actual last-linear-layer residual, then set only its new-cell contribution to zero algebraically, with upstream states fixed. This is not unassisted generation.',
          'mean_cell_output_norm': float(contribution.norm(dim=-1).mean()),
          'max_cell_output_norm': float(contribution.norm(dim=-1).max()),
          'reports': reports, 'weights_changed': False, 'free_generation_performed': False,
          'training': False, 'gradients': False, 'optimizer': False, 'final_set_read': False}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
