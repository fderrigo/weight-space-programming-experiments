"""Explicit short-fact writes on ordinary question trajectories only.

Exploratory last-layer hypothesis: remove the competing scaffold trajectories,
protect all measured native continuations, and keep other factual token readouts
fixed. No complete response or stopping token is prescribed. One declared
matrix construction, no gradients, optimizer, or loss-driven search.
"""
import datetime
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

from readout_contexts import load_contexts

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'bare_context'
assert not (ROOT / f'patches/{STEM}_basis.safetensors').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(6)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
all_K, all_R, all_records, old = load_contexts(ROOT, t)
indices = [i for i, row in enumerate(all_records) if row['prefix'] == '']
K, R = all_K[indices].double(), all_R[indices].double()
records = [all_records[i] for i in indices]
with safe_open(MODEL / 'model.safetensors', framework='pt') as source:
    E = source.get_tensor('model.embed_tokens.weight').float()
    E.mul_(source.get_tensor('model.norm.weight').float())
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'tensor': 'model.layers.23.mlp.down_proj.weight',
    'reason': 'Earlier edits changed many post-fact native continuations; scalar constraints also raised competing factual words. Limit writes to the ordinary question plus incomplete short-unit trajectories, remove grammatical scaffold constraints, preserve every measured second-set native continuation, and keep all other factual-token readouts fixed in each requested value.',
    'factual_context_filter': 'prefix == empty, all subtokens; no outcomes used to select facts',
    'formula': 'For each native state choose d=e_target orthogonal to its native top-128 competitors and every other factual-token output vector. Scale d to raise the repetition-adjusted target above the native best by 4 RMS units. Q=row-space(C); Z=K-KQQt; deltaW=Vt*pinv(Z)t.',
    'contexts': len(K), 'native_competitors': 128, 'margin': 4.0,
    'repetition_penalty': 1.1, 'protection_rtol': 1e-6, 'pinv_rtol': .001,
    'protected_source': '3406 previous states + 504 second-set native continuation states + 24 first completed-unit states',
    'training': False, 'gradients': False, 'optimizer': False, 'runtime_hooks': False,
    'EOS_programmed': False, 'full_responses_prescribed': False,
    'third_set_read': False, 'final_set_read': False,
}
(ROOT / f'patches/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
fact_tokens = sorted({row['token'] for row in all_records})
values, notes = [], []
for begin in range(0, len(R), 16):
    rb = R[begin:begin + 16].float()
    rms = (rb.square().mean(-1) + 1e-6).sqrt()
    logits = (rb / rms[:, None]) @ E.T
    for j, scores in enumerate(logits):
        row = records[begin + j]
        token = row['token']
        history = sorted(set(row['history']))
        scores[history] = torch.where(scores[history] < 0, scores[history] * 1.1,
                                      scores[history] / 1.1)
        target_score = float(scores[token])
        scores[token] = -torch.inf
        competing = sorted((set(scores.topk(128).indices.tolist()) | set(fact_tokens)) - {token})
        target = E[token].double() / (1.1 if token in history else 1.0)
        q = torch.linalg.qr(E[competing].double().T, mode='reduced')[0]
        direction = target - q @ (q.T @ target)
        amplitude = max(0.0, (float(scores.max()) + spec['margin']) * float(rms[j])
                        - float(target @ R[begin + j]))
        value = amplitude * direction / (direction @ target)
        values.append(value)
        notes.append({k: v for k, v in row.items() if k != 'history'})
        notes[-1].update({'original_target_score': target_score,
                          'constrained_other_tokens': len(competing),
                          'value_norm': float(value.norm())})
    if begin % 128 == 0:
        print('Declared factual values', min(begin + 16, len(R)), '/', len(R), flush=True)
V = torch.stack(values)
with np.load(ROOT / 'results/orthogonal_story_roles_2048_controls_native_states.npz') as z:
    C0 = torch.from_numpy(z['C'][-525:].copy()).double()
with np.load(ROOT / 'results/newdevelopment_control_states.npz') as z:
    C1 = torch.from_numpy(z['C'].copy()).double()
with np.load(ROOT / 'results/expanded_context_states.npz') as z:
    C2 = torch.from_numpy(z['C'].copy()).double()
with np.load(ROOT / 'results/native_second_continuations_states.npz') as z:
    native = torch.cat([torch.from_numpy(z[f'A_{i}'].copy()).double() for i in range(24)])
termination = json.loads((ROOT / 'results/termination_diagnostic.json').read_text('utf8'))['states']
with np.load(ROOT / 'results/termination_states.npz') as z:
    completed = torch.stack([torch.from_numpy(z[f'A_{i}'].copy()).double()
                             for i, row in enumerate(termination) if row['kind'] == 'completed_unit'])
C = torch.cat([C0, C1, C2, native, completed])
print('Preservation SVD', list(C.shape), flush=True)
_, singular, vh = torch.linalg.svd(C, full_matrices=False)
rank = int((singular > singular[0] * spec['protection_rtol']).sum())
Q = vh[:rank]
Z = K - (K @ Q.T) @ Q
u, s, v = torch.linalg.svd(Z, full_matrices=False)
keep = s > s[0] * spec['pinv_rtol']
dual = (u[:, keep] / s[keep]) @ v[keep]
delta = (V.T @ dual).float()
actual = (old['before'] + delta - old['before']).double()
effect = K @ actual.T
for begin in range(0, len(R), 16):
    changed = (R[begin:begin + 16] + effect[begin:begin + 16]).float()
    rms = (changed.square().mean(-1) + 1e-6).sqrt()
    logits = (changed / rms[:, None]) @ E.T
    for j, scores in enumerate(logits):
        row = records[begin + j]
        history = sorted(set(row['history']))
        scores[history] = torch.where(scores[history] < 0, scores[history] * 1.1,
                                      scores[history] / 1.1)
        token = row['token']
        target_score = float(scores[token])
        notes[begin + j]['actual_decoder_target_rank'] = int((scores > scores[token]).sum()) + 1
        scores[token] = -torch.inf
        other = int(scores.argmax())
        notes[begin + j].update({'actual_decoder_margin': target_score - float(scores[other]),
                                 'actual_decoder_best_other': t.decode([other])})
second_units = {}
for row in notes:
    if row['question'].startswith('second_'):
        second_units.setdefault(row['question'], []).append(row['actual_decoder_target_rank'])
report = {
    'contexts': len(K), 'preservation_states': len(C), 'protected_rank': rank,
    'projected_key_rank': int(keep.sum()), 'delta_norm': float(actual.norm()),
    'original_weight_norm': float(old['before'].norm()),
    'relative_value_error': float((effect - V).norm() / V.norm()),
    'mean_control_shift': float((C @ actual.T).norm(dim=-1).mean()),
    'max_control_shift': float((C @ actual.T).norm(dim=-1).max()),
    'actual_decoder_first_rank_count': sum(n['actual_decoder_target_rank'] == 1 for n in notes),
    'second_unit_count': len(second_units),
    'second_units_all_tokens_rank_one': [key for key, ranks in second_units.items() if max(ranks) == 1],
    'contexts_diagnostic': notes, 'free_generation_performed': False,
    'third_set_read': False, 'final_set_read': False,
}
(ROOT / f'results/{STEM}_diagnostic.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
save_file({'before': old['before'].contiguous(), 'delta': delta.contiguous(),
           'keys': K.float().contiguous(), 'values': V.float().contiguous()},
          ROOT / f'patches/{STEM}_basis.safetensors')
print({k: v for k, v in report.items() if k != 'contexts_diagnostic'}, flush=True)
