"""Closed-form readout metric with the existing, context-specific constraints.

Unlike global_readout_geometry, this does not impose a margin against every
token or use a shared direction per token. It changes only the metric used to
choose the value vector. No training, optimizer, gradients, or generation.
"""
import datetime
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'contextual_readout'
torch.set_grad_enabled(False)
torch.set_num_threads(6)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)


def render(question):
    return t.apply_chat_template([{'role': 'user', 'content': question}],
                                tokenize=False, add_generation_prompt=True)


with safe_open(MODEL / 'model.safetensors', framework='pt') as source:
    E = source.get_tensor('model.embed_tokens.weight').float()
    E.mul_(source.get_tensor('model.norm.weight').float())

oldmeta = ROOT / 'results/orthogonal_story_roles_2048_controls_native_states.json'
oldcases = json.loads(oldmeta.read_text('utf8'))['case_records']
units = json.loads((ROOT / 'patches/multipoint_protected_spec.json').read_text('utf8'))['short_units']
first = json.loads((ROOT / 'new_development_questions.json').read_text('utf8'))
newcases = [
    {'fact': r['fact'] + '_answer_start', 'question': r['id'], 'prefix': '',
     'base': t.encode(render(r['prompt']), add_special_tokens=False),
     'concept': t.encode(units[r['fact']], add_special_tokens=False)}
    for r in first if r['group'] == 'target'
]
expandedmeta = json.loads((ROOT / 'results/expanded_context_states.json').read_text('utf8'))
groups = [(oldcases, oldmeta.with_suffix('.npz')),
          (newcases, ROOT / 'results/multipoint_protected_new_states.npz'),
          (expandedmeta['cases'], ROOT / 'results/expanded_context_states.npz')]
keys, residuals, tokens, descriptions, seen = [], [], [], [], {}
for cases, path in groups:
    with np.load(path) as arrays:
        for i, case in enumerate(cases):
            for j, token in enumerate(case['concept']):
                sequence = tuple(case['base'] + case['concept'][:j])
                if sequence in seen:
                    assert seen[sequence] == token
                    continue
                seen[sequence] = token
                keys.append(torch.from_numpy(arrays[f'A_{i}'][j].copy()))
                residuals.append(torch.from_numpy(arrays[f'R_{i}'][j].copy()))
                tokens.append(token)
                descriptions.append({'fact': case['fact'], 'question': case['question'],
                                     'prefix': case['prefix'], 'piece': t.decode([token]),
                                     'token': token, 'subtoken': j})
K, R = torch.stack(keys), torch.stack(residuals)
old = load_file(ROOT / 'patches/expanded_context_basis.safetensors')
assert torch.equal(K, old['keys'])

spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'tensor': 'model.layers.23.mlp.down_proj.weight', 'contexts': len(K),
    'reason': 'Global shared directions required pathological amplitudes. Keep exactly the earlier target gap and original top-128 competitor constraints; choose the value using the full-vocabulary output metric instead of Euclidean residual length.',
    'output_metric': 'S = (Eg).T @ (Eg) / vocab_size + 1e-5 * mean(diag(cov)) * I',
    'value_formula': 'L=[e_target,e_native_top128]. B=solve(S,L). b=[max(0,gap+4)*native_RMS,0,...,0]. V=B*pinv(Lt*B)*b.',
    'weight_formula': 'Q=orthonormal row space of same preservation matrix. Z=K-KQQt. deltaW=Vt*pinv(Z)t.',
    'margin': 4.0, 'competitors': 128, 'isotropic_fraction': 1e-5,
    'value_pinv_rtol': 1e-10, 'weight_pinv_rtol': .001,
    'training': False, 'gradients': False, 'optimizer': False,
    'third_set_read': False, 'final_set_read': False, 'EOS_programmed': False,
    'free_generation_performed': False,
}
(ROOT / f'patches/{STEM}_spec.json').write_text(
    json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf8')

print('Measuring native competitors for', len(K), 'contexts', flush=True)
competitors, gaps, rms_values = [], [], []
for begin in range(0, len(R), 16):
    rb = R[begin:begin + 16]
    rms = (rb.square().mean(-1) + 1e-6).sqrt()
    logits = (rb / rms[:, None]) @ E.T
    for j, row in enumerate(logits):
        token = tokens[begin + j]
        target_logit = row[token].clone()
        row[token] = -torch.inf
        competitors.append(row.topk(128).indices)
        gaps.append(float(row.max() - target_logit))
        rms_values.append(float(rms[j]))

cov = (E.T @ E / len(E)).double()
metric = cov.clone()
metric.diagonal().add_(spec['isotropic_fraction'] * metric.diag().mean())
inverse = torch.linalg.inv(metric)
union = sorted(set(tokens) | set(torch.stack(competitors).flatten().tolist()))
position = {token: i for i, token in enumerate(union)}
inverse_readouts = E[union].double() @ inverse
print('Cached metric transforms for', len(union), 'readouts', flush=True)

values, notes = [], []
for i, (token, competing, gap, rms) in enumerate(zip(tokens, competitors, gaps, rms_values)):
    ids = [token] + competing.tolist()
    L = E[ids].double().T
    B = inverse_readouts[[position[x] for x in ids]].T
    amplitude = max(0.0, gap + spec['margin']) * rms
    rhs = torch.zeros(len(ids), dtype=torch.float64)
    rhs[0] = amplitude
    value = B @ torch.linalg.pinv(L.T @ B, hermitian=True, rtol=spec['value_pinv_rtol']) @ rhs
    equality_error = float((L.T @ value - rhs).abs().max())
    values.append(value.float())
    notes.append({**descriptions[i], 'native_gap': gap, 'amplitude': amplitude,
                  'value_norm': float(value.norm()), 'equality_max_error': equality_error})
    if (i + 1) % 100 == 0:
        print('Context-specific values', i + 1, '/', len(K), flush=True)
V = torch.stack(values)

for begin in range(0, len(R), 16):
    logits = (R[begin:begin + 16] + V[begin:begin + 16]) @ E.T
    for j, row in enumerate(logits):
        token = tokens[begin + j]
        notes[begin + j]['isolated_target_rank'] = int((row > row[token]).sum()) + 1

Q = torch.linalg.qr(old['preservation_basis'].double().T, mode='reduced')[0].T
Z = K.double() - (K.double() @ Q.T) @ Q
print('Solving the fixed weight equation', Z.shape, flush=True)
delta = (V.double().T @ torch.linalg.pinv(Z, rtol=spec['weight_pinv_rtol']).T).float()
actual = (old['before'] + delta - old['before']).double()
old_actual = (old['before'] + old['delta'] - old['before']).double()
with np.load(ROOT / 'results/orthogonal_story_roles_2048_controls_native_states.npz') as z:
    C0 = torch.from_numpy(z['C'][-525:].copy()).double()
with np.load(ROOT / 'results/newdevelopment_control_states.npz') as z:
    C1 = torch.from_numpy(z['C'].copy()).double()
with np.load(ROOT / 'results/expanded_context_states.npz') as z:
    C2 = torch.from_numpy(z['C'].copy()).double()
C = torch.cat([C0, C1, C2])
assert len(C) == 3406


def value_energy(v):
    v = v.double()
    return ((v @ cov) * v).sum(-1).clamp_min(0).sqrt()


old_energy, new_energy = value_energy(old['values']), value_energy(V)
old_metric_energy = ((old['values'].double() @ metric) * old['values'].double()).sum(-1)
new_metric_energy = ((V.double() @ metric) * V.double()).sum(-1)
diagnostic = {
    'contexts': len(K), 'preservation_states': len(C), 'protected_rank': len(Q),
    'isolated_first_rank_count': sum(n['isolated_target_rank'] == 1 for n in notes),
    'old_median_output_rms_shift': float(old_energy.median()),
    'new_median_output_rms_shift': float(new_energy.median()),
    'old_value_norm': float(old['values'].norm()), 'new_value_norm': float(V.norm()),
    'metric_energy_reduced_contexts': int((new_metric_energy <= old_metric_energy * 1.0001 + 1e-9).sum()),
    'old_delta_norm': float(old_actual.norm()), 'delta_norm': float(actual.norm()),
    'old_full_output_operator_norm': float(((cov @ old_actual) * old_actual).sum().clamp_min(0).sqrt()),
    'new_full_output_operator_norm': float(((cov @ actual) * actual).sum().clamp_min(0).sqrt()),
    'relative_value_error': float((K.double() @ actual.T - V.double()).norm() / V.double().norm()),
    'mean_control_shift': float((C @ actual.T).norm(dim=-1).mean()),
    'max_control_shift': float((C @ actual.T).norm(dim=-1).max()),
    'max_value_equality_error': max(n['equality_max_error'] for n in notes),
    'contexts_diagnostic': notes, 'free_generation_performed': False,
    'third_set_read': False, 'final_set_read': False,
}
(ROOT / f'results/{STEM}_diagnostic.json').write_text(
    json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding='utf8')
save_file({'before': old['before'].contiguous(), 'delta': delta.contiguous(),
           'keys': K.contiguous(), 'values': V.contiguous()},
          ROOT / f'patches/{STEM}_basis.safetensors')
print({k: v for k, v in diagnostic.items() if k != 'contexts_diagnostic'}, flush=True)
