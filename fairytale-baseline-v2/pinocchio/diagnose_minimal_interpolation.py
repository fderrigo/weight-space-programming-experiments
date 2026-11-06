"""Diagnose interpolation sensitivity without creating or applying a patch.

Compare the existing hard-preserved geometry with the unprojected geometry.
Exact block leave-one-question-out identities assess dependence on individual
development formulations. No held-out final question is read, no gradient or
optimizer is used, and no candidate is selected automatically.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors'
output = ROOT / 'results/minimal_interpolation_diagnostic.json'
assert not output.exists()
torch.set_grad_enabled(False)
torch.set_num_threads(2)
source = ROOT / 'patches/minimal_fact_completion_basis.safetensors'
basis = load_file(source)
K, V = basis['keys'].double(), basis['values'].double()
before = basis['before']
actual_hard = (before + basis['delta'] - before).double()
with np.load(ROOT / 'results/minimal_fact_states.npz') as z:
    R = torch.from_numpy(z['R'].copy()).double()
records = json.loads((ROOT / 'results/minimal_fact_states.json').read_text('utf8'))['records']
groups = []
for i, row in enumerate(records):
    if row['subtoken'] == 0:
        groups.append({'question': row['question'], 'indices': []})
    assert groups[-1]['question'] == row['question']
    groups[-1]['indices'].append(i)
assert len(groups) == 74
with safe_open(MODEL, framework='pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').float()
    E *= f.get_tensor('model.norm.weight').float()
    G = f.get_tensor('model.layers.23.mlp.gate_proj.weight').float()
    U = f.get_tensor('model.layers.23.mlp.up_proj.weight').float()
with np.load(ROOT / 'results/orthogonal_story_roles_2048_controls_native_states.npz') as z:
    C0 = torch.from_numpy(z['C'][-525:].copy()).double()
with np.load(ROOT / 'results/newdevelopment_control_states.npz') as z:
    C1 = torch.from_numpy(z['C'].copy()).double()
with np.load(ROOT / 'results/expanded_context_states.npz') as z:
    C2 = torch.from_numpy(z['C'].copy()).double()
with np.load(ROOT / 'results/minimal_fact_units_complete_states.npz') as z:
    native = torch.cat([torch.from_numpy(z[f'A_{i}'].copy()).double() for i in range(24)])
meta = json.loads((ROOT / 'results/extended_native_inputs.json').read_text('utf8'))
with np.load(ROOT / 'results/extended_native_inputs.npz') as z:
    X = torch.cat([torch.from_numpy(z['X'][row['offset']:row['offset'] + row['states']].copy())
                   for row in meta['sequences'] if row['kind'] == 'narrative'])
narrative = (torch.nn.functional.silu(X @ G.T) * (X @ U.T)).double()
C = torch.cat([C0, C1, C2, native, narrative])
assert len(C) == 4289


def leave_one_out(keys):
    gram = keys @ keys.T
    gram_inv = torch.linalg.pinv(gram, hermitian=True, rtol=1e-12)
    coefficients = gram_inv @ V
    prediction = torch.empty_like(V)
    for group in groups:
        idx = group['indices']
        removed = torch.linalg.solve(gram_inv[idx][:, idx], coefficients[idx])
        prediction[idx] = V[idx] - removed
    return prediction


def token_review(effect):
    ranks = []
    for begin in range(0, len(R), 16):
        changed = (R[begin:begin + 16] + effect[begin:begin + 16]).float()
        rms = (changed.square().mean(-1) + 1e-6).sqrt()
        logits = (changed / rms[:, None]) @ E.T
        for j, scores in enumerate(logits):
            row = records[begin + j]
            history = sorted(set(row['history']))
            scores[history] = torch.where(scores[history] < 0, scores[history] * 1.1,
                                          scores[history] / 1.1)
            ranks.append(int((scores > scores[row['token']]).sum()) + 1)
    return {
        'first_token_rank_one': sum(ranks[group['indices'][0]] == 1 for group in groups),
        'all_concept_tokens_rank_one': sum(all(ranks[i] == 1 for i in group['indices']) for group in groups),
        'token_rank_one': sum(rank == 1 for rank in ranks),
        'question_total': len(groups), 'token_total': len(ranks), 'ranks': ranks,
    }


def shifts(delta):
    result = {}
    for label, states in [('all_preservation', C), ('minimal_native', native), ('narrative', narrative)]:
        value = (states @ delta.T).norm(dim=-1)
        result[label] = {'mean': float(value.mean()), 'max': float(value.max()),
                         'rms': float(value.square().mean().sqrt())}
    return result


print('Unprojected geometry and block leave-one-question-out diagnostic', flush=True)
unprojected = (V.T @ torch.linalg.pinv(K, rtol=1e-6).T).float()
unprojected_actual = (before + unprojected - before).double()
loo_unprojected = leave_one_out(K)
unprojected_review = token_review(loo_unprojected)
print('Unprojected norm', float(unprojected_actual.norm()),
      'LOO first tokens', unprojected_review['first_token_rank_one'], flush=True)
print('Reconstructing existing hard-preservation subspace', flush=True)
_, singular, vh = torch.linalg.svd(C, full_matrices=False)
Q = vh[singular > singular[0] * 1e-6]
Z = K - (K @ Q.T) @ Q
loo_hard = leave_one_out(Z)
hard_review = token_review(loo_hard)
report = {
    'basis_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'decision_by': 'Codex', 'purpose': 'Diagnose exact interpolation and preservation constraints; no new candidate or generation.',
    'formula': 'For each whole-question block J, omitted-block prediction is V_J - solve((Gram^-1)_JJ, (Gram^-1 V)_J), with Gram=KKt or Z Zt and Z=K(I-QtQ).',
    'preservation_rank': len(Q), 'unprojected_delta_norm': float(unprojected_actual.norm()),
    'unprojected_pinv_rtol': 1e-6, 'gram_pinv_rtol': 1e-12,
    'existing_hard_delta_norm': float(actual_hard.norm()),
    'original_weight_norm': float(before.norm()),
    'unprojected_value_relative_error': float((K @ unprojected_actual.T - V).norm() / V.norm()),
    'unprojected_shift': shifts(unprojected_actual), 'existing_hard_shift': shifts(actual_hard),
    'unprojected_leave_one_question_out': unprojected_review,
    'hard_leave_one_question_out': hard_review,
    'weights_applied': False, 'new_candidate_saved': False, 'training': False,
    'gradients': False, 'optimizer': False, 'whole_answer_accuracy_measured': False,
    'third_responses_used_in_calculation': False, 'final_set_read': False,
}
output.write_text(json.dumps(report, indent=2), encoding='utf8')
np.savez_compressed(ROOT / 'results/minimal_interpolation_diagnostic.npz',
                    unprojected_loo=loo_unprojected.float().numpy(), hard_loo=loo_hard.float().numpy())
print({key: value for key, value in report.items() if 'leave_one' not in key}, flush=True)
print('Hard LOO first tokens', hard_review['first_token_rank_one'], flush=True)
