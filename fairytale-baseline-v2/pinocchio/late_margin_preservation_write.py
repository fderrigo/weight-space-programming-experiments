"""A single closed-form native-margin correction in a fixed feature space."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from semantic_carrier_weights import tensor_sha

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'late_margin_preservation'
assert not (ROOT / f'patches/{STEM}_basis.safetensors').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
source_basis = ROOT / 'patches/late_selector_fact_basis.safetensors'
basis = load_file(source_basis)
previous = json.loads((RDIR / 'late_selector_fact_diagnostic.json').read_text('utf8'))
margins = json.loads((RDIR / 'late_native_margins.json').read_text('utf8'))
assert margins['fixed_linear_penalty_valid'] == margins['states'] == 3267
meta = json.loads((RDIR / 'progress_carrier_states.json').read_text('utf8'))
rows = meta['records']
fit = [i for i, r in enumerate(rows[:288]) if not r['id'].startswith('second_')]
probe = [i for i in range(288) if i not in fit]
cells = previous['cells']
with np.load(RDIR / 'late_native_margins.npz') as z:
    native = z['record_indices'].tolist()
    M = torch.from_numpy(z['margin_directions'].copy()).double()
    target = torch.from_numpy(z['desired_margin_changes'].copy()).double()
with np.load(RDIR / 'late_selector_states.npz') as z:
    X, A, R = [torch.from_numpy(z[k][:4693].copy()) for k in ['X', 'A', 'R']]
with np.load(RDIR / 'carrier_concept_states_original.npz') as z:
    original_R = torch.from_numpy(z['R'].copy())
with np.load(RDIR / 'late_selector_geometry.npz') as z:
    B = torch.from_numpy(z['protected_carrier'].copy()).double()
source_spec = json.loads((ROOT / 'patches/late_selector_fact_spec.json').read_text('utf8'))
spec = {**source_spec,
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'reason': 'Euclidean native compensation preserves2986/3267 original tokens but reverses281 decisions, including arithmetic. Apply one explicit algebraic correction to measured margins while cancelling its mean effect on all46 factual cells.',
    'feature_rule': 'Exactly the first128 singular modes of the same3267 native remaining-MLP activations, whitened by their singular values. No rank search.',
    'factual_null_formula': 'H=A_remaining V128/S128; J=mean_cell(F); C=mean_cell(H); G=H-F pinv(J) C. Thus every factual cell mean of G is zero.',
    'native_margin_rule': 'For a changed native winner, restore its original raw margin against the current edited winner. For an unchanged winner, preserve its current raw margin against its current runner-up. One constraint per declared native guard; no new token labels or new EOS targets.',
    'closed_form': 'Normalize each G row by its norm. K=(G G^T) elementwise*(M M^T), where M are unit decoder margin directions orthogonal to the25 carriers. lambda=pinv(K,rtol=1e-8) target; T=G^T diag(lambda) M. Apply T in remaining down columns and -pinv(J) C T in the46 fact columns. No iterations or line search.',
    'equation_gate': 'Require relative mean-factual residual drift below1e-4 after F32. Report native constraint residual and decoder results without redefining them as whole-answer accuracy.',
    'kernel_pinv_rtol': 1e-8, 'native_feature_rank': 128, 'mean_factual_drift_tolerance': 1e-4,
    'source_basis_sha256': hashlib.sha256(source_basis.read_bytes()).hexdigest(),
    'native_margins_sha256': hashlib.sha256((RDIR / 'late_native_margins.npz').read_bytes()).hexdigest(),
    'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
    'full_responses_prescribed': False, 'EOS_programmed': False, 'final_set_read': False,
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(ROOT / f'patches/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
chosen = basis['reassigned_neurons']
fact_ids = chosen[:previous['fact_neurons']]
mask = torch.ones(4864, dtype=torch.bool)
mask[chosen] = False
remaining = mask.nonzero().flatten()
new_A = A.clone()
new_A[:, chosen] = torch.nn.functional.silu(X @ basis['gate_after'][chosen].T) * (X @ basis['up_after'][chosen].T)
parts = new_A[:, fact_ids].double()
F = parts[:, 0::3] - parts[:, 1::3] - parts[:, 2::3]
Ar = A[:, remaining].double()
print('Fixed128-mode native margin geometry', flush=True)
_, singular, vh = torch.linalg.svd(Ar[native], full_matrices=False)
L = vh[:128].T / singular[:128]
del vh, singular
H = Ar @ L
J = torch.stack([F[c['members']].mean(0) for c in cells])
C = torch.stack([H[c['members']].mean(0) for c in cells])
P = torch.linalg.pinv(J, rtol=1e-4) @ C
G = H - F @ P
mean_G = torch.stack([G[c['members']].mean(0) for c in cells])
assert float(mean_G.abs().max()) < 1e-10
norm = G[native].norm(dim=-1)
assert bool((norm > 1e-8).all())
Gn = G[native] / norm[:, None]
changed = torch.tensor([not r['same_original_token'] for r in margins['records']])
target[~changed] = 0
scaled_target = target / norm
kernel = (Gn @ Gn.T) * (M @ M.T)
print('One closed-form margin system', len(native), 'constraints', flush=True)
eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
kept = eigenvalues > eigenvalues.max() * spec['kernel_pinv_rtol']
coefficients = eigenvectors[:, kept] @ ((eigenvectors[:, kept].T @ scaled_target) / eigenvalues[kept])
T = Gn.T @ (coefficients[:, None] * M)
assert float((T @ B).abs().max()) < 1e-6
achieved = (G[native] @ T * M).sum(-1)
after = {name: basis[name + '_after'].clone() for name in ['gate', 'up', 'down']}
after['down'][:, remaining] += (T.T @ L.T).float()
fact_delta = -(P @ T).T
after['down'][:, fact_ids[0::3]] += fact_delta.float()
after['down'][:, fact_ids[1::3]] -= fact_delta.float()
after['down'][:, fact_ids[2::3]] -= fact_delta.float()
assert all(torch.isfinite(v).all() for v in after.values())
before_R = R - A @ basis['down_before'].T + new_A @ basis['down_after'].T
after_R = R - A @ basis['down_before'].T + new_A @ after['down'].T
mean_before = torch.stack([before_R[c['members']].double().mean(0) for c in cells])
mean_after = torch.stack([after_R[c['members']].double().mean(0) for c in cells])
mean_drift = float((mean_after - mean_before).norm() / mean_before.norm())
assert mean_drift < 1e-4, mean_drift
print('Mean factual drift F32', mean_drift, flush=True)

model_file = ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors'
with safe_open(model_file, framework='pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
    original_weights = {name: f.get_tensor(f'model.layers.23.mlp.{name}_proj.weight').float() for name in after}
checks = list(range(288)) + native
histories = {}
for sequence in meta['sequences']:
    for local, pos in enumerate(sequence['positions']):
        index = sequence['offset'] + local
        if index in checks:
            histories[index] = sorted(set(sequence['ids'][:pos + 1]))
predictions = {}
token_notes = []
for begin in range(0, len(checks), 16):
    selected = checks[begin:begin + 16]
    block = after_R[selected]
    scores = (block / (block.square().mean(-1) + 1e-6).sqrt()[:, None]) @ E.T
    for index, score in zip(selected, scores):
        seen = histories[index]
        score[seen] = torch.where(score[seen] < 0, score[seen] * 1.1, score[seen] / 1.1)
        predictions[index] = int(score.argmax())
        if index < 288:
            target_token = rows[index]['next_token']
            token_notes.append({'id': rows[index]['id'], 'subtoken': rows[index]['subtoken'],
                                'desired_token': target_token,
                                'decoder_rank': int((score > score[target_token]).sum()) + 1})
    if begin % 512 == 0:
        print('Reconstructed decoder check', begin, '/', len(checks), flush=True)
coverage = []
for split, indices in [('fit', fit), ('probe_observed', probe)]:
    ids = sorted({rows[i]['id'] for i in indices})
    coverage.append({'split': split, 'states': len(indices), 'questions': len(ids),
                     'desired_tokens_first': sum(token_notes[i]['decoder_rank'] == 1 for i in indices),
                     'first_tokens_first': sum(token_notes[i]['decoder_rank'] == 1 for i in indices if rows[i]['subtoken'] == 0),
                     'all_steps_first': sum(all(token_notes[i]['decoder_rank'] == 1 for i in indices if rows[i]['id'] == q) for q in ids)})
same = [predictions[r['record_index']] == r['original_token'] for r in margins['records']]
gained = sum(good and not r['same_original_token'] for good, r in zip(same, margins['records']))
lost = sum(not good and r['same_original_token'] for good, r in zip(same, margins['records']))
diffs = [{'tensor': f'model.layers.23.mlp.{name}_proj.weight',
          'changed_parameters': int(torch.count_nonzero(value - original_weights[name])),
          'difference_from_original_norm': float((value - original_weights[name]).norm()),
          'after_sha256_f32': tensor_sha(value)} for name, value in after.items()]
native_change = after_R[native] - original_R[native]
actual_constraint = ((after_R[native] - before_R[native]).double() * M).sum(-1)
report = {
    'coverage': coverage, 'cells': cells, 'token_diagnostics': token_notes,
    'native_states': len(native), 'native_same_original_tokens_before': margins['same_original_tokens'],
    'native_same_original_tokens_after': sum(same), 'native_recovered': gained, 'native_new_losses': lost,
    'kernel_rank': int(kept.sum()), 'kernel_dimensions': len(native),
    'constraint_relative_error_double': float((achieved - target).norm() / target.norm()),
    'constraint_relative_error_f32': float((actual_constraint - target).norm() / target.norm()),
    'constraint_max_absolute_error_f32': float((actual_constraint - target).abs().max()),
    'mean_factual_residual_relative_drift_f32': mean_drift,
    'down_update_norm': float((after['down'] - basis['down_after']).norm()),
    'native_mean_residual_error': float(native_change.norm(dim=-1).mean()),
    'native_max_residual_error': float(native_change.norm(dim=-1).max()),
    'last_mlp_differences': diffs, 'changed_tensors': previous['changed_tensors'],
    'changed_parameters': previous['changed_parameters'] - sum(r['changed_parameters'] for r in previous['last_mlp_differences']) + sum(r['changed_parameters'] for r in diffs),
    'fact_neurons': previous['fact_neurons'], 'erase_neurons': previous['erase_neurons'],
    'training': False, 'gradients': False, 'optimizer': False,
    'free_generation_performed': False, 'counts_as_correction': False, 'final_set_read': False,
}
save_file({**{name + '_before': basis[name + '_before'].contiguous() for name in after},
           **{name + '_after': value.contiguous() for name, value in after.items()},
           'reassigned_neurons': chosen.contiguous(), 'scale_readout': basis['scale_readout'].contiguous()},
          ROOT / f'patches/{STEM}_basis.safetensors')
np.savez_compressed(RDIR / f'{STEM}_constraints.npz', native_indices=np.array(native),
                    desired=target.numpy(), achieved_double=achieved.numpy(), achieved_f32=actual_constraint.numpy(),
                    native_predicted_tokens=np.array([predictions[i] for i in native]))
(RDIR / f'{STEM}_diagnostic.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k not in ['cells', 'token_diagnostics', 'last_mlp_differences']}, flush=True)
