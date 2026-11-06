"""Fixed native-attention selector geometry; no inference hook or answer claim."""
import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'template_attention_selector'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
meta = json.loads((RDIR / 'progress_carrier_states.json').read_text('utf8'))
collection = json.loads((RDIR / 'progress_attention_states.json').read_text('utf8'))
assert collection['completed'] and collection['prefix_positions'] == 24
rows, sequences = meta['records'], meta['sequences']
with np.load(RDIR / 'progress_attention_states.npz') as z:
    X = torch.from_numpy(z['attention_input22'].copy())
    prefix_X = torch.from_numpy(z['prefix_attention_input22'].copy())
    all_X = torch.from_numpy(z['all_nonprefix_attention_input22'].copy())
    all_positions = torch.from_numpy(z['all_nonprefix_positions'].copy())
    all_sequences = torch.from_numpy(z['all_nonprefix_sequence_indices'].copy())
with np.load(RDIR / 'progress_carrier_geometry.npz') as z:
    U = torch.from_numpy(z['carrier'].copy()).double()
    centers = torch.from_numpy(z['semantic_centers'].copy()).double()
qmeta = json.loads((RDIR / 'question_layer_diagnostic_states.json').read_text('utf8'))['sequences']
classes = sorted({r['label'] for r in qmeta})
assert len(classes) == 13
fit = [i for i, r in enumerate(rows[:288]) if not r['id'].startswith('second_')]
probe = [i for i in range(288) if i not in fit]
fit_sequences = [j for j, s in enumerate(sequences) if not (s['kind'] == 'factual' and s['id'].startswith('second_'))]
fit_mask = torch.tensor([int(j) in fit_sequences for j in all_sequences])
model_file = ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors'
with safe_open(model_file, framework='pt') as f:
    gain = f.get_tensor('model.layers.22.input_layernorm.weight').double()
    original_k = f.get_tensor('model.layers.22.self_attn.k_proj.weight')[:64].double()
    original_k_bias = f.get_tensor('model.layers.22.self_attn.k_proj.bias')[:64].float()
P, prefix_P, all_P = X.double() / gain, prefix_X.double() / gain, all_X.double() / gain
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'reason': 'The transport identifies more relations than the linear fact writer converts into correct concepts. Test an existing attention softmax as a sharper class selector, using unchanged causal template positions as addresses.',
    'address_selection': 'Select thirteen prefix positions by pivoted Gram orthogonalization in the inverse native second-moment metric. The other eleven prefix positions are inactive.',
    'key_formula': 'M=C^-1 S^T (S C^-1 S^T)^-1 Y. C is the second moment of nonprefix normalized inputs, with .01 mean-diagonal isotropic term. Y gives one class coordinate and an enabled flag to each selected prefix position; all other prefix rows are zero.',
    'query_formula': 'Head6 uses thirteen cosine-center linear scores on the carried coordinates, normalized by the median first-position norm on the earlier fifty questions. The fourteenth query coordinate is a constant bias. Other query coordinates are zero.',
    'class_temperature': 64.0, 'prefix_flag_logit_scale': 1024.0,
    'rotary_axes': list(range(50, 64)), 'rope_theta': 1000000.0,
    'native_regularization_fraction': .01, 'selected_prefix_count': 13,
    'query_head': 6, 'key_value_head': 0,
    'nonprefix_keys_include_user_prompt_tokens': True,
    'simulation_only': True, 'new_model_weights_applied': False,
    'template_unchanged': True, 'tokenizer_unchanged': True,
    'stored_content': 'Thirteen relation identifiers, not responses, correct names or narrative text.',
    'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
    'free_generation_performed': False, 'counts_as_correction': False,
    'third_set_used': False, 'final_set_read': False,
    'source_sha256': hashlib.sha256((RDIR / 'progress_attention_states.json').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
native = all_P[fit_mask]
C = native.T @ native / len(native)
C.diagonal().add_(spec['native_regularization_fraction'] * C.diag().mean())
dual_prefix = torch.linalg.solve(C, prefix_P.T)
gram = prefix_P @ dual_prefix
schur = gram.clone()
slots = []
for _ in range(13):
    diagonal = schur.diag().clone()
    if slots:
        diagonal[slots] = -float('inf')
    i = int(diagonal.argmax())
    slots.append(i)
    column = schur[:, i].clone()
    schur -= column[:, None] @ column[None, :] / schur[i, i]
Y = torch.zeros((24, 14), dtype=torch.float64)
for c, slot in enumerate(slots):
    Y[slot, c], Y[slot, 13] = 1, 1
M = dual_prefix @ torch.linalg.solve(gram, Y)
physical_memory = (M / gain[:, None]).T.float()
actual_prefix_memory = prefix_X @ physical_memory.T
prefix_error = float((actual_prefix_memory.double() - Y).abs().max())
assert prefix_error < 1e-4
memory = all_X @ physical_memory.T
center_directions = centers / centers.norm(dim=-1, keepdim=True)
first_fit = [i for i in fit if rows[i]['subtoken'] == 0]
scale = (P[first_fit] @ U).norm(dim=-1).median()
q_weight = torch.zeros((64, 896), dtype=torch.float32)
q_bias = torch.zeros(64, dtype=torch.float32)
axes = torch.arange(50, 64)
q_weight[axes[:13]] = (8 * spec['class_temperature'] / scale * (U @ center_directions.T) / gain[:, None]).T.float()
q_bias[axes[13]] = 8 * spec['prefix_flag_logit_scale']
scaled_k = original_k * gain[None, :]
k_weight = ((scaled_k - (scaled_k @ U) @ U.T) / gain[None, :]).float()
k_bias = original_k_bias.clone()
k_weight[axes] = physical_memory
k_bias[axes] = 0
prefix_K = prefix_X @ k_weight.T + k_bias
all_K = all_X @ k_weight.T + k_bias
queries = X @ q_weight.T + q_bias
frequencies = 1.0 / (spec['rope_theta'] ** (torch.arange(0, 64, 2).float() / 64))


def rotate(value, positions):
    angles = positions.float()[:, None] * frequencies[None, :]
    angles = torch.cat([angles, angles], dim=-1)
    half = torch.cat([-value[:, 32:], value[:, :32]], dim=-1)
    return value * angles.cos() + half * angles.sin()


prefix_K_rotated = rotate(prefix_K, torch.arange(24))
all_K_rotated = rotate(all_K, all_positions)
queries_rotated = rotate(queries, torch.tensor([r['position'] for r in rows]))
actual_values = torch.empty((len(rows), 13), dtype=torch.float32)
prefix_class_weights = torch.empty_like(actual_values)
prefix_mass = torch.empty(len(rows), dtype=torch.float32)
for j, sequence in enumerate(sequences):
    key_indices = (all_sequences == j).nonzero().flatten()
    keys = torch.cat([prefix_K_rotated, all_K_rotated[key_indices]])
    values = torch.cat([actual_prefix_memory[:, :13], memory[key_indices, :13]])
    positions = torch.cat([torch.arange(24), all_positions[key_indices]])
    begin, end = sequence['offset'], sequence['offset'] + sequence['states']
    scores = queries_rotated[begin:end] @ keys.T / 8
    allowed = positions[None, :] <= torch.tensor(sequence['positions'])[:, None]
    scores.masked_fill_(~allowed, -float('inf'))
    attention = scores.softmax(-1)
    actual_values[begin:end] = attention @ values
    prefix_class_weights[begin:end] = attention[:, slots]
    prefix_mass[begin:end] = attention[:, slots].sum(-1)
predictions = [classes[i] for i in actual_values.argmax(-1).tolist()]
raw_predictions = [classes[i] for i in ((P @ U) @ center_directions.T).argmax(-1).tolist()]


def canonical(fact):
    return {'wood_finder': 'wood_donor'}.get(fact, fact)


comparisons = []
for name, pred in [('native_carrier_linear_scores', raw_predictions), ('simulated_attention_values', predictions)]:
    for split, indices in [('fit', fit), ('probe_observed', probe)]:
        ids = sorted({rows[i]['id'] for i in indices})
        comparisons.append({'representation': name, 'split': split, 'states': len(indices), 'questions': len(ids),
                            'relation_correct': sum(pred[i] == canonical(rows[i]['fact']) for i in indices),
                            'first_relation_correct': sum(pred[i] == canonical(rows[i]['fact']) for i in indices if rows[i]['subtoken'] == 0),
                            'all_steps_relation_correct': sum(all(pred[i] == canonical(rows[i]['fact']) for i in indices if rows[i]['id'] == q) for q in ids)})
groups = {}
for kind in sorted({r['kind'] for r in rows}):
    indices = [i for i, r in enumerate(rows) if r['kind'] == kind]
    groups[kind] = {'states': len(indices), 'prefix_mass_min': float(prefix_mass[indices].min()),
                    'prefix_mass_median': float(prefix_mass[indices].median()),
                    'prefix_mass_below_99_percent': int((prefix_mass[indices] < .99).sum()),
                    'classified_outside': sum(predictions[i] == 'outside' for i in indices)}
result = {'spec': spec, 'slots_by_class': dict(zip(classes, slots)), 'prefix_interpolation_max_error_f32': prefix_error,
          'nonprefix_keys': len(all_X), 'native_covariance_fit_keys': int(fit_mask.sum()),
          'native_flag_max': float(memory[:, 13].max()), 'native_flag_p99': float(torch.quantile(memory[:, 13], .99)),
          'comparisons': comparisons, 'groups': groups,
          'interpretation': 'This simulates specified Q/K/V maps and native rotary attention on observed inputs. It is not an applied selector, a factual decoder or a corrected-answer result.'}
(RDIR / f'{STEM}.json').write_text(json.dumps(result, indent=2), encoding='utf8')
np.savez_compressed(RDIR / f'{STEM}.npz', q_weight=q_weight.numpy(), q_bias=q_bias.numpy(),
                    k_weight=k_weight.numpy(), k_bias=k_bias.numpy(), value_rows=physical_memory[:13].numpy(),
                    slots=np.array(slots), actual_values=actual_values.numpy(),
                    prefix_class_weights=prefix_class_weights.numpy(), prefix_mass=prefix_mass.numpy())
print({k: v for k, v in result.items() if k not in ['spec', 'interpretation']}, flush=True)
