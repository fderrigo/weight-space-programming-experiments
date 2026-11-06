"""Explicit weight equations after the declared question-contrast projection.

Compensate its effect on existing language trajectories, then write the same
420 short-fact values in the remaining last-MLP subspace. No complete-answer
targets, stopping-token writes, learning loop, gradients, or optimizer.
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
STEM = 'centered_attention_fact'
assert not (ROOT / f'patches/{STEM}_basis.safetensors').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(6)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
_, _, all_records, _ = load_contexts(ROOT, t)
records = [row for row in all_records if row['prefix'] == '']
bare = load_file(ROOT / 'patches/bare_context_basis.safetensors')
attention = load_file(ROOT / 'patches/centered_question_readout_basis.safetensors')
metadata = json.loads((ROOT / 'results/centered_contexts.json').read_text('utf8'))
assert len(records) == metadata['factual_states'] == 420
with np.load(ROOT / 'results/centered_contexts.npz') as z:
    K = torch.from_numpy(z['factual_K'].copy()).double()
    R = torch.from_numpy(z['factual_R'].copy()).double()
    R0 = torch.from_numpy(z['factual_original_R'].copy()).double()
    C = torch.from_numpy(z['control_K'].copy()).double()
    Rc = torch.from_numpy(z['control_R'].copy()).double()
    Rc0 = torch.from_numpy(z['control_original_R'].copy()).double()
assert len(C) == 4138
with safe_open(MODEL / 'model.safetensors', framework='pt') as source:
    E = source.get_tensor('model.embed_tokens.weight').float()
    E.mul_(source.get_tensor('model.norm.weight').float())
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'kind': 'centered_attention_with_factual_down_write',
    'tensors': ['model.layers.23.self_attn.q_proj.weight',
                'model.layers.23.self_attn.q_proj.bias',
                'model.layers.23.self_attn.o_proj.weight',
                'model.layers.23.mlp.down_proj.weight'],
    'attention_source': 'patches/centered_question_readout_spec.json',
    'factual_states': len(K), 'preservation_states': len(C),
    'control_pinv_rtol': 1e-6, 'factual_pinv_rtol': .001,
    'reason': 'The declared centered attention readout improves question-relation separation but strongly changes external states. Compensate the measured original residuals exactly in the last MLP, then write the unchanged short factual vectors in its remaining nullspace.',
    'formula': 'Vc=Rcontrol_original-Rcontrol_after_attention; D0=VcT*pinv(C)T; Q=row-space(C); Z=K-KQtQ; Vf=Rfactual_original+Vbare-Rfactual_after_attention; deltaW=D0+(Vf-KD0T)T*pinv(Z)T.',
    'narratives': 'Preserve the original HF narrative trajectories to prevent additional linguistic damage. This is not canonical correction, and narrative fidelity remains a separate unmet objective.',
    'preservation_difference_from_prior_patch': 'Includes 641 exact stored older external positions instead of the prior 525-position matrix slice; all 116 additional positions come from the same 14 earlier questions.',
    'training': False, 'gradients': False, 'optimizer': False, 'runtime_hooks': False,
    'EOS_programmed': False, 'full_responses_prescribed': False,
    'third_set_read': False, 'final_set_read': False,
}
(ROOT / f'patches/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
print('Preservation equations', list(C.shape), flush=True)
uc, sc, vhc = torch.linalg.svd(C, full_matrices=False)
keep_c = sc > sc[0] * spec['control_pinv_rtol']
Q = vhc[keep_c]
dual_c = (uc[:, keep_c] / sc[keep_c]) @ Q
particular = (Rc0 - Rc).T @ dual_c
Z = K - (K @ Q.T) @ Q
desired = R0 + bare['values'].double() - R
remaining = desired - K @ particular.T
uf, sf, vhf = torch.linalg.svd(Z, full_matrices=False)
keep_f = sf > sf[0] * spec['factual_pinv_rtol']
dual_f = (uf[:, keep_f] / sf[keep_f]) @ vhf[keep_f]
delta = (particular + remaining.T @ dual_f).float()
actual = (bare['before'] + delta - bare['before']).double()
assert torch.isfinite(actual).all()
final_R = R + K @ actual.T
control_error = Rc + C @ actual.T - Rc0
notes = []
for begin in range(0, len(K), 16):
    rb = final_R[begin:begin + 16].float()
    rms = (rb.square().mean(-1) + 1e-6).sqrt()
    logits = (rb / rms[:, None]) @ E.T
    for j, scores in enumerate(logits):
        row = records[begin + j]
        history = sorted(set(row['history']))
        scores[history] = torch.where(scores[history] < 0, scores[history] * 1.1,
                                      scores[history] / 1.1)
        token = row['token']
        rank = int((scores > scores[token]).sum()) + 1
        target_score = float(scores[token])
        scores[token] = -torch.inf
        other = int(scores.argmax())
        notes.append({**{key: value for key, value in row.items() if key != 'history'},
                      'actual_decoder_target_rank': rank,
                      'actual_decoder_margin': target_score - float(scores[other]),
                      'actual_decoder_best_other': t.decode([other])})
units = {}
for row in notes:
    if row['question'].startswith('second_'):
        units.setdefault(row['question'], []).append(row['actual_decoder_target_rank'])
down_changed = int(torch.count_nonzero(actual))
attention_changed = int(torch.count_nonzero(attention['q_weight_before'])
                        + torch.count_nonzero(attention['q_bias_before'])
                        + torch.count_nonzero(attention['o_after'] - attention['o_before']))
report = {
    'contexts': len(K), 'preservation_states': len(C),
    'protected_rank': int(keep_c.sum()), 'projected_key_rank': int(keep_f.sum()),
    'down_delta_norm': float(actual.norm()), 'original_down_norm': float(bare['before'].norm()),
    'particular_compensation_norm': float(particular.norm()),
    'relative_factual_residual_error': float((final_R - R0 - bare['values'].double()).norm() / bare['values'].double().norm()),
    'mean_control_residual_error': float(control_error.norm(dim=-1).mean()),
    'max_control_residual_error': float(control_error.norm(dim=-1).max()),
    'rms_control_residual_error': float(control_error.square().sum(-1).mean().sqrt()),
    'changed_down_parameters': down_changed, 'changed_attention_parameters': attention_changed,
    'total_changed_parameters': down_changed + attention_changed,
    'actual_decoder_first_rank_count': sum(row['actual_decoder_target_rank'] == 1 for row in notes),
    'second_units_all_tokens_rank_one': [key for key, ranks in units.items() if max(ranks) == 1],
    'contexts_diagnostic': notes, 'free_generation_performed': False,
    'diagnostic_is_whole_answer_accuracy': False, 'third_set_read': False, 'final_set_read': False,
}
(ROOT / f'results/{STEM}_diagnostic.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
save_file({**attention, 'before': bare['before'].contiguous(), 'delta': delta.contiguous()},
          ROOT / f'patches/{STEM}_basis.safetensors')
print({key: value for key, value in report.items() if key != 'contexts_diagnostic'}, flush=True)
