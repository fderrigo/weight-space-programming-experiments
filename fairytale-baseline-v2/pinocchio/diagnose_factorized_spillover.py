"""Decompose the fixed candidate's first-position spillover on observed cases."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'factorized_carrier_spillover'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(2)
rows = json.loads((RDIR / 'third_question_features_states.json').read_text('utf8'))['rows']
assert len(rows) == 44
basis_path = ROOT / 'patches/factorized_carrier_basis.safetensors'
basis = load_file(basis_path)
chosen = basis['reassigned_neurons']
keep = torch.ones(4864, dtype=torch.bool)
keep[chosen] = False
with np.load(RDIR / 'semantic_carrier_forward_protected_states.npz') as z:
    X = torch.from_numpy(z['X23'][108:].copy())
    A = torch.from_numpy(z['A23'][108:].copy())
    parent_R = torch.from_numpy(z['R'][108:].copy())
with np.load(RDIR / 'third_question_features_states.npz') as z:
    original_R = torch.from_numpy(z['final_residual'].copy())
new_A = torch.nn.functional.silu(X @ basis['gate_after'].T) * (X @ basis['up_after'].T)
actual_R = parent_R - A @ basis['down_before'].T + new_A @ basis['down_after'].T
parent_change = parent_R - original_R
removed_native_cells = -(A[:, chosen] @ basis['down_before'][:, chosen].T)
new_cell_output = new_A[:, chosen] @ basis['down_after'][:, chosen].T
remaining_write = A[:, keep] @ (basis['down_after'][:, keep] - basis['down_before'][:, keep]).T
shift = actual_R - original_R
reconstruction_error = float((shift - parent_change - removed_native_cells - new_cell_output - remaining_write).norm() / shift.norm())
assert reconstruction_error < 1e-5
controls = [i for i, row in enumerate(rows) if row['group'] == 'control']
targets = [i for i, row in enumerate(rows) if row['group'] == 'target']
_, singular, directions = torch.linalg.svd(shift[controls].double(), full_matrices=False)
fractions = (singular.square() / singular.square().sum()).tolist()
mode = directions[0]
component_reports = []
for group, selected in [('target', targets), ('control', controls)]:
    component_reports.append({'group': group, 'states': len(selected),
                              'mean_norms': {name: float(value[selected].norm(dim=-1).mean())
                                             for name, value in [('parent_change', parent_change),
                                                                 ('removed_native_cells', removed_native_cells),
                                                                 ('new_cell_output', new_cell_output),
                                                                 ('remaining_dense_write', remaining_write),
                                                                 ('total_change', shift)]},
                              'mean_relative_total_change': float((shift[selected].norm(dim=-1) / original_R[selected].norm(dim=-1)).mean())})
with safe_open(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors', framework='pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
mode_logits = E.double() @ mode
high = mode_logits.abs().topk(16)
manifest = json.loads((RDIR / 'factorized_carrier_third_observed_manifest.json').read_text('utf8'))
assert manifest['completed']
generated = json.loads((RDIR / 'factorized_carrier_third_observed.json').read_text('utf8'))
generated_by_id = {r['id']: r for r in generated}
first_checks = []
for begin in range(0, 44, 8):
    rb = actual_R[begin:begin + 8]
    logits = (rb / (rb.square().mean(-1) + 1e-6).sqrt()[:, None]) @ E.T
    for j, scores in enumerate(logits):
        row = rows[begin + j]
        history = sorted(set(row['ids']))
        scores[history] = torch.where(scores[history] < 0, scores[history] * 1.1, scores[history] / 1.1)
        predicted = int(scores.argmax())
        response = generated_by_id[row['id']]
        observed = response['generated_ids'][0] if response['generated_ids'] else response['stop_token']
        first_checks.append({'id': row['id'], 'calculated_first': predicted,
                             'generated_first': observed, 'same': predicted == observed})
report = {'basis_sha256': hashlib.sha256(basis_path.read_bytes()).hexdigest(),
          'responses_sha256': manifest['responses_sha256'],
          'decomposition_relative_error': reconstruction_error,
          'component_norms': component_reports,
          'control_shift_singular_energy_fractions': fractions,
          'first_mode_energy_fraction': fractions[0],
          'first_four_modes_energy_fraction': sum(fractions[:4]),
          'first_mode_largest_effective_token_projections': [
              {'token': int(i), 'signed_projection': float(mode_logits[i])} for i in high.indices],
          'first_token_reconstruction_matches': sum(r['same'] for r in first_checks),
          'first_token_checks': first_checks, 'first_positions_only': True,
          'read_only_diagnostic': True, 'weights_changed': False,
          'training': False, 'gradients': False, 'optimizer': False,
          'third_set_already_observed': True, 'final_set_read': False,
          'interpretation': 'Identifies which part of the fixed edit drives unseen-formulation spillover; no output projection has yet been removed or selected as a new candidate.'}
np.savez_compressed(RDIR / f'{STEM}.npz', leading_control_shift_direction=mode.numpy(),
                    all_control_directions=directions.numpy(), singular_values=singular.numpy(),
                    original_R=original_R.numpy(), edited_R=actual_R.numpy(),
                    new_cell_output=new_cell_output.numpy(), remaining_write=remaining_write.numpy())
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k not in ['first_token_checks', 'first_mode_largest_effective_token_projections']}, flush=True)
