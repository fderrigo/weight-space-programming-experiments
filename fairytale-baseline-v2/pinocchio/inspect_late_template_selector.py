"""One fixed24-prototype last-block attention construction, diagnostic only."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'late_template_selector'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
meta = json.loads((RDIR / 'progress_carrier_states.json').read_text('utf8'))
collection = json.loads((RDIR / 'late_selector_inputs.json').read_text('utf8'))
assert collection['completed'] and collection['prefix_positions'] == 24
rows, sequences = meta['records'], meta['sequences']
with np.load(RDIR / 'late_selector_inputs.npz') as z:
    X = torch.from_numpy(z['attention_input23'].copy())
    prefix_X = torch.from_numpy(z['prefix_attention_input23'].copy())
    all_X = torch.from_numpy(z['all_nonprefix_attention_input23'].copy())
    all_positions = torch.from_numpy(z['all_nonprefix_positions'].copy())
    all_sequences = torch.from_numpy(z['all_nonprefix_sequence_indices'].copy())
with np.load(RDIR / 'progress_carrier_geometry.npz') as z:
    U = torch.from_numpy(z['carrier'].copy()).double()
with np.load(RDIR / 'compact_outside_prototypes.npz') as z:
    directions = torch.from_numpy(z['semantic_directions'].copy()).double()
    labels = z['labels'].tolist()
assert directions.shape == (24, 12)
facts = labels[:12]
assert len(set(facts)) == 12 and labels[12:] == ['outside'] * 12
fit = [i for i, r in enumerate(rows[:288]) if not r['id'].startswith('second_')]
probe = [i for i in range(288) if i not in fit]
selection = fit + list(range(288, 4693))
fit_sequences = [j for j, s in enumerate(sequences) if not (s['kind'] == 'factual' and s['id'].startswith('second_'))]
fit_mask = torch.tensor([int(j) in fit_sequences for j in all_sequences])
with safe_open(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors', framework='pt') as f:
    gain = f.get_tensor('model.layers.23.input_layernorm.weight').double()
parent_path = ROOT / 'patches/late_selector_parent_attention.safetensors'
assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == collection['parent_attention_sha256']
parent = load_file(parent_path)
with np.load(RDIR / 'late_reference_geometry.npz') as z:
    costs = torch.from_numpy(z['value_coordinate_costs'].copy())
with np.load(RDIR / 'late_reference_inputs.npz') as z:
    native_values = torch.from_numpy(z['attention_values23'].copy())
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'hypothesis': 'Last-block attention may implement the24-prototype winner without passing new class coordinates through another attention layer. This is a feasibility simulation, not an applied patch or answer result.',
    'addresses': 'All24 unchanged template-prefix positions, in chronological order; first12 factual prototypes then12 compact outside prototypes. No slot or prototype search.',
    'memory_formula': 'C^-1 S^T (S C^-1 S^T)^-1 Y; C from all nonprefix inputs of the earlier factual plus native histories, regularized by .01 mean diagonal. Key targets are twelve semantic prototype coordinates and flag1; value targets are twelve factual one-hot coordinates with outside zero.',
    'query_formula': 'Twelve linear carried semantic coordinates, scaled by median first-fit norm and temperature64, plus constant flag bias8192; low-frequency axes51..63. Keep other key coordinates from the exact progress parent.',
    'value_group_rule': 'For each KV group sum its13 lowest measured individual coordinate contribution costs; select the lower sum. Choose the least-cost13 coordinates in that group,12 class values then one constant reference.',
    'query_head_rule': 'Within that group select the query head with smallest mean squared native O-head contribution on196 earlier factual plus4405 native positions.',
    'future_shared_query_rule': 'Suppress both rotary partners of replaced key axes in the other six sharing heads; merely zeroing one half does not suppress coupling after different-position rotation.',
    'class_temperature': 64.0, 'prefix_flag_logit_scale': 1024.0,
    'rotary_axes': list(range(51, 64)), 'rope_theta': 1000000.0,
    'native_regularization_fraction': .01,
    'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
    'template_unchanged': True, 'new_model_weights_applied': False,
    'free_generation_performed': False, 'counts_as_correction': False,
    'third_set_used': False, 'final_set_read': False,
    'source_sha256': hashlib.sha256((RDIR / 'late_selector_inputs.json').read_bytes()).hexdigest(),
    'prototypes_sha256': hashlib.sha256((RDIR / 'compact_outside_prototypes.npz').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
ranked = [costs[g * 64:(g + 1) * 64].argsort()[:13] for g in range(2)]
group_costs = [float(costs[g * 64 + ranked[g]].sum()) for g in range(2)]
group = int(np.argmin(group_costs))
value_coordinates = ranked[group]
head_costs = []
for head in range(group * 7, (group + 1) * 7):
    contribution = native_values[selection, head * 64:(head + 1) * 64] @ parent['o_proj.weight'][:, head * 64:(head + 1) * 64].T
    head_costs.append(float(contribution.square().sum(-1).mean()))
head = group * 7 + int(np.argmin(head_costs))
P, prefix_P, all_P = X.double() / gain, prefix_X.double() / gain, all_X.double() / gain
native = all_P[fit_mask]
C = native.T @ native / len(native)
C.diagonal().add_(.01 * C.diag().mean())
dual = torch.linalg.solve(C, prefix_P.T)
gram = prefix_P @ dual
Y = torch.cat([directions, torch.ones(24, 1, dtype=torch.float64)], dim=1)
VY = torch.zeros(24, 12, dtype=torch.float64)
VY[:12] = torch.eye(12, dtype=torch.float64)
memory = ((dual @ torch.linalg.solve(gram, torch.cat([Y, VY], dim=1))) / gain[:, None]).T.float()
prefix_memory = prefix_X @ memory.T
nonprefix_memory = all_X @ memory.T
interpolation_error = float((prefix_memory.double() - torch.cat([Y, VY], dim=1)).abs().max())
assert interpolation_error < 1e-4
first_fit = [i for i in fit if rows[i]['subtoken'] == 0]
scale = (P[first_fit] @ U).norm(dim=-1).median()
axes = torch.arange(51, 64)
q_weight = torch.zeros(64, 896)
q_bias = torch.zeros(64)
q_weight[axes[:12]] = (8 * 64 / scale * U / gain[:, None]).T.float()
q_bias[axes[12]] = 8 * 1024
k_weight = parent['k_proj.weight'][group * 64:(group + 1) * 64].clone()
k_bias = parent['k_proj.bias'][group * 64:(group + 1) * 64].clone()
k_weight[axes], k_bias[axes] = memory[:13], 0
frequencies = 1.0 / (1000000.0 ** (torch.arange(0, 64, 2).float() / 64))


def rotate(value, positions):
    angles = positions.float()[:, None] * frequencies[None, :]
    angles = torch.cat([angles, angles], dim=-1)
    return value * angles.cos() + torch.cat([-value[:, 32:], value[:, :32]], dim=-1) * angles.sin()


prefix_K = rotate(prefix_X @ k_weight.T + k_bias, torch.arange(24))
all_K = rotate(all_X @ k_weight.T + k_bias, all_positions)
queries = rotate(X @ q_weight.T + q_bias, torch.tensor([r['position'] for r in rows]))
values = torch.empty(len(rows), 12)
slot_weights = torch.empty(len(rows), 24)
prefix_mass = torch.empty(len(rows))
for j, sequence in enumerate(sequences):
    key_indices = (all_sequences == j).nonzero().flatten()
    keys = torch.cat([prefix_K, all_K[key_indices]])
    payload = torch.cat([prefix_memory[:, 13:], nonprefix_memory[key_indices, 13:]])
    positions = torch.cat([torch.arange(24), all_positions[key_indices]])
    begin, end = sequence['offset'], sequence['offset'] + sequence['states']
    scores = queries[begin:end] @ keys.T / 8
    scores.masked_fill_(positions[None, :] > torch.tensor(sequence['positions'])[:, None], -float('inf'))
    attention = scores.softmax(-1)
    values[begin:end] = attention @ payload
    slot_weights[begin:end] = attention[:, :24]
    prefix_mass[begin:end] = attention[:, :24].sum(-1)
predictions = [labels[i] for i in slot_weights.argmax(-1).tolist()]
raw_predictions = [labels[i] for i in ((P @ U) @ directions.T).argmax(-1).tolist()]
aggregated = torch.cat([values, (1 - values.sum(-1))[:, None]], dim=1)
value_predictions = [(facts + ['outside'])[i] for i in aggregated.argmax(-1).tolist()]


def canonical(fact):
    return {'wood_finder': 'wood_donor'}.get(fact, fact)


comparisons = []
for name, pred in [('raw_prototype_winner', raw_predictions), ('rotated_prefix_winner', predictions), ('aggregated_class_values', value_predictions)]:
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
                    'prefix_mass_below_99_percent': int((prefix_mass[indices] < .99).sum()),
                    'classified_outside': sum(value_predictions[i] == 'outside' for i in indices)}
np.savez_compressed(RDIR / f'{STEM}.npz', q_weight=q_weight.numpy(), q_bias=q_bias.numpy(),
                    k_weight=k_weight.numpy(), k_bias=k_bias.numpy(), value_rows=memory[13:].numpy(),
                    value_coordinates=value_coordinates.numpy(), query_head=np.array(head), key_value_head=np.array(group),
                    actual_values=values.numpy(), slot_weights=slot_weights.numpy(), prefix_mass=prefix_mass.numpy())
result = {'spec': spec, 'labels_by_prefix_position': labels, 'query_head': head, 'key_value_head': group,
          'value_coordinates_within_group': value_coordinates.tolist(), 'group_coordinate_cost_sums': group_costs,
          'query_head_costs_within_group': head_costs, 'prefix_interpolation_max_error_f32': interpolation_error,
          'nonprefix_keys': len(all_X), 'native_flag_max': float(nonprefix_memory[:, 12].max()),
          'native_flag_p99': float(torch.quantile(nonprefix_memory[:, 12], .99)),
          'comparisons': comparisons, 'groups': groups,
          'selector_sha256': hashlib.sha256((RDIR / f'{STEM}.npz').read_bytes()).hexdigest()}
(RDIR / f'{STEM}.json').write_text(json.dumps(result, indent=2), encoding='utf8')
print({k: v for k, v in result.items() if k != 'spec'}, flush=True)
