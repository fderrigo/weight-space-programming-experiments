"""Exact factual/external constraints with soft native narrative preservation.

Single declared matrix construction, no gradient, loss-driven search, optimizer,
training, new modules, complete-answer targets, or stopping-token targets.
"""
import datetime
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from readout_contexts import load_contexts

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'hybrid_preservation'
assert not (ROOT / f'patches/{STEM}_basis.safetensors').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(6)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
all_K, all_R, all_records, _ = load_contexts(ROOT, t)
indices = [i for i, row in enumerate(all_records) if row['prefix'] == '']
K, R = all_K[indices].double(), all_R[indices]
records = [all_records[i] for i in indices]
bare = load_file(ROOT / 'patches/bare_context_basis.safetensors')
assert torch.equal(K.float(), bare['keys'])
V = bare['values'].double()
with np.load(ROOT / 'results/orthogonal_story_roles_2048_controls_native_states.npz') as z:
    C0 = torch.from_numpy(z['C'][-525:].copy()).double()
with np.load(ROOT / 'results/newdevelopment_control_states.npz') as z:
    C1 = torch.from_numpy(z['C'].copy()).double()
with np.load(ROOT / 'results/expanded_context_states.npz') as z:
    C2 = torch.from_numpy(z['C'].copy()).double()
native_rows = json.loads((ROOT / 'results/native_second_continuations.json').read_text('utf8'))
native_review = json.loads((ROOT / 'results/native_second_continuations_review.json').read_text('utf8'))
correct_ids = {row['id'] for row in native_review['annotations'] if row['correct_with_supplied_unit']}
assert len(correct_ids) == 13
good, incorrect = [], []
with np.load(ROOT / 'results/native_second_continuations_states.npz') as z:
    for i, row in enumerate(native_rows):
        (good if row['id'] in correct_ids else incorrect).append(torch.from_numpy(z[f'A_{i}'].copy()).double())
metadata = json.loads((ROOT / 'results/extended_native_inputs.json').read_text('utf8'))
with np.load(ROOT / 'results/extended_native_inputs.npz') as z:
    narrative_inputs = torch.cat([torch.from_numpy(z['X'][row['offset']:row['offset'] + row['states']].copy())
                                  for row in metadata['sequences'] if row['kind'] == 'narrative'])
with safe_open(MODEL / 'model.safetensors', framework='pt') as source:
    gate = source.get_tensor('model.layers.23.mlp.gate_proj.weight').float()
    up = source.get_tensor('model.layers.23.mlp.up_proj.weight').float()
    E = source.get_tensor('model.embed_tokens.weight').float()
    E.mul_(source.get_tensor('model.norm.weight').float())
narrative = (torch.nn.functional.silu(narrative_inputs @ gate.T) * (narrative_inputs @ up.T)).double()
C_hard = torch.cat([C0, C1, C2, *good])
C_soft = torch.cat([narrative, *incorrect])
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'tensor': 'model.layers.23.mlp.down_proj.weight',
    'reason': 'Bare-context writes corrected 13 development answers but corrupted narratives outside their preserved subspace. Keep the same 420 factual values, preserve the existing external trajectories and the 13 correct native second-set continuations exactly, and constrain changes on original narratives and already-incorrect continuations through a declared covariance metric.',
    'factual_states': len(K), 'hard_preservation_states': len(C_hard),
    'soft_preservation_states': len(C_soft), 'original_narrative_states': len(narrative),
    'hard_native_correct_ids': sorted(correct_ids),
    'protection_rtol': 1e-6, 'gram_pinv_rtol': 1e-10, 'isotropic_fraction': .1,
    'formula': 'Q=row-space(C_hard); P=I-QtQ; Z=KP; S=C_soft P; lambda=.1*||S||F^2/(rows(S)*(4864-rank(Q))). B=[Z-ZSt*(rows(S)*lambda*I+SSt)^-1*S]/lambda, reprojected with P. deltaW=Vt*pinv(BZt)*B.',
    'interpretation': 'An explicit covariance-weighted solution of the declared factual constraints inside the hard-preserved nullspace. No automatic tuning of the metric, labels, layers, or values.',
    'training': False, 'gradients': False, 'optimizer': False, 'runtime_hooks': False,
    'EOS_programmed': False, 'full_responses_prescribed': False,
    'third_set_read': False, 'final_set_read': False,
}
(ROOT / f'patches/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
print('Hard preservation SVD', list(C_hard.shape), 'soft', list(C_soft.shape), flush=True)
_, singular, vh = torch.linalg.svd(C_hard, full_matrices=False)
rank = int((singular > singular[0] * spec['protection_rtol']).sum())
Q = vh[:rank]
Z = K - (K @ Q.T) @ Q
S = C_soft - (C_soft @ Q.T) @ Q
free_dimensions = K.shape[1] - rank
isotropic = spec['isotropic_fraction'] * S.square().sum() / (len(S) * free_dimensions)
system = S @ S.T
system.diagonal().add_(len(S) * isotropic)
B = (Z - (Z @ S.T) @ torch.linalg.solve(system, S)) / isotropic
B -= (B @ Q.T) @ Q
gram = B @ Z.T
gram = (gram + gram.T) / 2
delta = (V.T @ torch.linalg.pinv(gram, hermitian=True, rtol=spec['gram_pinv_rtol']) @ B).float()
actual = (bare['before'] + delta - bare['before']).double()
assert torch.isfinite(actual).all()
effect = K @ actual.T
notes = []
for begin in range(0, len(R), 16):
    changed = (R[begin:begin + 16].double() + effect[begin:begin + 16]).float()
    rms = (changed.square().mean(-1) + 1e-6).sqrt()
    logits = (changed / rms[:, None]) @ E.T
    for j, scores in enumerate(logits):
        row = records[begin + j]
        history = sorted(set(row['history']))
        scores[history] = torch.where(scores[history] < 0, scores[history] * 1.1,
                                      scores[history] / 1.1)
        token = row['token']
        target_score = float(scores[token])
        target_rank = int((scores > scores[token]).sum()) + 1
        scores[token] = -torch.inf
        other = int(scores.argmax())
        notes.append({**{k: v for k, v in row.items() if k != 'history'},
                      'actual_decoder_target_rank': target_rank,
                      'actual_decoder_margin': target_score - float(scores[other]),
                      'actual_decoder_best_other': t.decode([other])})

second_units = {}
for row in notes:
    if row['question'].startswith('second_'):
        second_units.setdefault(row['question'], []).append(row['actual_decoder_target_rank'])

old_actual = (bare['before'] + bare['delta'] - bare['before']).double()


def shifts(states, change):
    values = (states @ change.T).norm(dim=-1)
    return {'mean': float(values.mean()), 'max': float(values.max()),
            'rms': float(values.square().mean().sqrt())}


report = {
    'contexts': len(K), 'hard_preservation_states': len(C_hard),
    'soft_preservation_states': len(C_soft), 'protected_rank': rank,
    'free_dimensions': free_dimensions, 'isotropic_coefficient': float(isotropic),
    'delta_norm': float(actual.norm()), 'original_weight_norm': float(bare['before'].norm()),
    'changed_entries': int(torch.count_nonzero(actual)),
    'relative_value_error': float((effect - V).norm() / V.norm()),
    'hard_preservation_shift': shifts(C_hard, actual),
    'soft_preservation_shift': shifts(C_soft, actual),
    'bare_context_soft_preservation_shift': shifts(C_soft, old_actual),
    'original_narrative_shift': shifts(narrative, actual),
    'bare_context_original_narrative_shift': shifts(narrative, old_actual),
    'actual_decoder_first_rank_count': sum(n['actual_decoder_target_rank'] == 1 for n in notes),
    'second_unit_count': len(second_units),
    'second_units_all_tokens_rank_one': [key for key, ranks in second_units.items() if max(ranks) == 1],
    'contexts_diagnostic': notes, 'free_generation_performed': False,
    'diagnostic_is_whole_answer_accuracy': False,
    'third_set_read': False, 'final_set_read': False,
}
(ROOT / f'results/{STEM}_diagnostic.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
save_file({'before': bare['before'].contiguous(), 'delta': delta.contiguous(),
           'keys': K.float().contiguous(), 'values': V.float().contiguous()},
          ROOT / f'patches/{STEM}_basis.safetensors')
print({k: v for k, v in report.items() if k != 'contexts_diagnostic'}, flush=True)
