"""One declared last-MLP write for minimal factual concepts, without training.

Retain the ordinary questions already used in development. Correct only the
short concept trajectories; preserve native continuations and measured original
external/narrative trajectories. No full response or EOS is a write target.
"""
import datetime
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from readout_contexts import load_contexts

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'minimal_fact'
assert not (ROOT / f'patches/{STEM}_basis.safetensors').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(6)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
_, _, old_records, _ = load_contexts(ROOT, t)
unit_spec = json.loads((ROOT / 'results/minimal_fact_units_spec.json').read_text('utf8'))
units = dict(unit_spec['short_units'])
units['wood_finder'] = units['wood_donor']
cases, seen_cases = [], set()
for row in old_records:
    if row['prefix'] != '' or row['subtoken'] != 0:
        continue
    key = tuple(row['history'])
    if key in seen_cases:
        continue
    seen_cases.add(key)
    fact = row['fact'].removesuffix('_answer_start')
    assert fact in units
    cases.append({'question': row['question'], 'fact': fact, 'prefix': '',
                  'base': row['history'], 'concept': t.encode(units[fact], add_special_tokens=False)})
assert len(cases) == 74
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'tensor': 'model.layers.23.mlp.down_proj.weight', 'short_units': units,
    'reason': 'Minimal concepts improve native post-concept factual fidelity in the diagnostic from 13 to 18 of 24. Use the same 74 ordinary development questions with their incomplete minimal concepts, avoiding the earlier grammatical subject phrases. Preserve every measured minimal-concept continuation, including incorrect ones, to avoid attributing linguistic corruption to correction.',
    'formula': 'For each state, project its target-token output vector orthogonally to its top-128 native competitors and all other minimal-concept token vectors. Raise the repetition-adjusted target above the native best by four RMS units. Q=row-space(C); Z=K-KQtQ; deltaW=Vt*pinv(Z)t.',
    'native_competitors': 128, 'margin': 4.0, 'repetition_penalty': 1.1,
    'protection_rtol': 1e-6, 'factual_pinv_rtol': .001,
    'preservation': '3406 earlier control/native states + 407 minimal-concept native states + 472 original HF narrative states.',
    'narratives': 'Original narratives are preserved as a language-stability check; this does not make them canonical.',
    'training': False, 'gradients': False, 'optimizer': False, 'runtime_hooks': False,
    'EOS_programmed': False, 'full_responses_prescribed': False,
    'third_set_read': False, 'final_set_read': False,
}
(ROOT / f'patches/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
m = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
capture = {}


def keep(name):
    def hook(module, inputs):
        capture[name] = inputs[0][0].clone()
    return hook


hooks = [m.model.layers[23].mlp.down_proj.register_forward_pre_hook(keep('A')),
         m.model.norm.register_forward_pre_hook(keep('R'))]
keys, residuals, records = [], [], []
for number, case in enumerate(cases):
    full = case['base'] + case['concept'][:-1]
    start = len(case['base']) - 1
    m.model(input_ids=torch.tensor([full]), use_cache=False)
    keys.append(capture['A'][start:].clone())
    residuals.append(capture['R'][start:].clone())
    for j, token in enumerate(case['concept']):
        records.append({'fact': case['fact'], 'question': case['question'], 'prefix': '',
                        'token': token, 'piece': t.decode([token]), 'subtoken': j,
                        'history': case['base'] + case['concept'][:j]})
    if number % 16 == 0:
        print('Minimal concept histories', number + 1, '/', len(cases), flush=True)
for hook in hooks:
    hook.remove()
K, R = torch.cat(keys).double(), torch.cat(residuals).double()
assert len(K) == len(R) == len(records)
np.savez_compressed(ROOT / f'results/{STEM}_states.npz', K=K.float().numpy(), R=R.float().numpy())
(ROOT / f'results/{STEM}_states.json').write_text(
    json.dumps({'cases': cases, 'records': records, 'weights_changed': False,
                'third_set_read': False, 'final_set_read': False}, ensure_ascii=False, indent=2), encoding='utf8')
before = m.model.layers[23].mlp.down_proj.weight.clone()
E = m.model.embed_tokens.weight * m.model.norm.weight
extended = json.loads((ROOT / 'results/extended_native_inputs.json').read_text('utf8'))
with np.load(ROOT / 'results/extended_native_inputs.npz') as z:
    narrative_X = torch.cat([torch.from_numpy(z['X'][row['offset']:row['offset'] + row['states']].copy())
                             for row in extended['sequences'] if row['kind'] == 'narrative'])
mlp = m.model.layers[23].mlp
narrative_A = (torch.nn.functional.silu(narrative_X @ mlp.gate_proj.weight.T)
               * (narrative_X @ mlp.up_proj.weight.T)).double()
del mlp, m, capture, keys, residuals
fact_tokens = sorted({row['token'] for row in records})
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
        original_score = float(scores[token])
        scores[token] = -torch.inf
        competing = sorted((set(scores.topk(128).indices.tolist()) | set(fact_tokens)) - {token})
        target = E[token].double() / (1.1 if token in history else 1.0)
        q = torch.linalg.qr(E[competing].double().T, mode='reduced')[0]
        direction = target - q @ (q.T @ target)
        amplitude = max(0.0, (float(scores.max()) + spec['margin']) * float(rms[j])
                        - float(target @ R[begin + j]))
        values.append(amplitude * direction / (direction @ target))
        notes.append({key: value for key, value in row.items() if key != 'history'})
        notes[-1]['original_target_score'] = original_score
V = torch.stack(values)
with np.load(ROOT / 'results/orthogonal_story_roles_2048_controls_native_states.npz') as z:
    C0 = torch.from_numpy(z['C'][-525:].copy()).double()
with np.load(ROOT / 'results/newdevelopment_control_states.npz') as z:
    C1 = torch.from_numpy(z['C'].copy()).double()
with np.load(ROOT / 'results/expanded_context_states.npz') as z:
    C2 = torch.from_numpy(z['C'].copy()).double()
with np.load(ROOT / 'results/minimal_fact_units_states.npz') as z:
    native_A = torch.cat([torch.from_numpy(z[f'A_{i}'].copy()).double() for i in range(24)])
assert len(native_A) == 407 and len(narrative_A) == 472
C = torch.cat([C0, C1, C2, native_A, narrative_A])
assert len(C) == 4285
print('Factual states', len(K), 'preservation SVD', list(C.shape), flush=True)
_, singular, vh = torch.linalg.svd(C, full_matrices=False)
keep = singular > singular[0] * spec['protection_rtol']
Q = vh[keep]
Z = K - (K @ Q.T) @ Q
u, s, v = torch.linalg.svd(Z, full_matrices=False)
keep_z = s > s[0] * spec['factual_pinv_rtol']
dual = (u[:, keep_z] / s[keep_z]) @ v[keep_z]
delta = (V.T @ dual).float()
actual = (before + delta - before).double()
assert torch.isfinite(actual).all()
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
        notes[begin + j]['actual_decoder_target_rank'] = int((scores > scores[row['token']]).sum()) + 1
units_checked = {}
for row in notes:
    if row['question'].startswith('second_'):
        units_checked.setdefault(row['question'], []).append(row['actual_decoder_target_rank'])
report = {
    'contexts': len(K), 'preservation_states': len(C), 'protected_rank': int(keep.sum()),
    'projected_key_rank': int(keep_z.sum()), 'delta_norm': float(actual.norm()),
    'original_weight_norm': float(before.norm()), 'changed_parameters': int(torch.count_nonzero(actual)),
    'relative_value_error': float((effect - V).norm() / V.norm()),
    'mean_control_shift': float((C @ actual.T).norm(dim=-1).mean()),
    'max_control_shift': float((C @ actual.T).norm(dim=-1).max()),
    'actual_decoder_first_rank_count': sum(row['actual_decoder_target_rank'] == 1 for row in notes),
    'second_units_all_tokens_rank_one': [key for key, ranks in units_checked.items() if max(ranks) == 1],
    'contexts_diagnostic': notes, 'free_generation_performed': False,
    'third_set_read': False, 'final_set_read': False,
}
(ROOT / f'results/{STEM}_diagnostic.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
save_file({'before': before.contiguous(), 'delta': delta.contiguous(),
           'keys': K.float().contiguous(), 'values': V.float().contiguous()},
          ROOT / f'patches/{STEM}_basis.safetensors')
print({key: value for key, value in report.items() if key != 'contexts_diagnostic'}, flush=True)
