"""Causal weight-only transport of centered prefix means in block 21.

This repeats the declared centering/isometric map at the earlier block whose
native values carry stronger question information. No labels determine the
matrix transformation. All probes are diagnostic on already observed data.
"""
import datetime
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'centered_layer21_question'
assert not (ROOT / f'patches/{STEM}_basis.safetensors').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
old_rows = json.loads((ROOT / 'results/question_layer_diagnostic_states.json').read_text('utf8'))['sequences']
new_rows = json.loads((ROOT / 'results/third_question_features_states.json').read_text('utf8'))['rows']
rows = old_rows + new_rows
train = [i for i, row in enumerate(old_rows) if row['split'] == 'prototype']
assert (len(old_rows), len(new_rows), len(train)) == (108, 44, 64)
with np.load(ROOT / 'results/question_layer_diagnostic_states.npz') as z:
    X21 = torch.from_numpy(z['X'][:, 0, 21].copy())
    H22 = torch.from_numpy(z['H'][:, 0, 22].copy())
    original_X23 = torch.from_numpy(z['X'][:, 0, 23].copy())
    original_R = torch.from_numpy(z['H'][:, 0, 24].copy())
with np.load(ROOT / 'results/attention_feature_diagnostic_states.npz') as z:
    means = torch.from_numpy(z['prefix_values'][:, 21].copy()).flatten(1)
    native_heads = torch.from_numpy(z['heads'][:, 21].copy())
with safe_open(MODEL / 'model.safetensors', framework='pt') as f:
    O = f.get_tensor('model.layers.21.self_attn.o_proj.weight').float()
    Q = f.get_tensor('model.layers.21.self_attn.q_proj.weight').float()
    b = f.get_tensor('model.layers.21.self_attn.q_proj.bias').float()
    G = f.get_tensor('model.layers.21.mlp.gate_proj.weight').float()
    U = f.get_tensor('model.layers.21.mlp.up_proj.weight').float()
    D = f.get_tensor('model.layers.21.mlp.down_proj.weight').float()
    g = f.get_tensor('model.layers.21.post_attention_layernorm.weight').float()
norms = [float(O[:, h * 64:(h + 1) * 64].norm()) for h in range(14)]
heads = [min(range(group * 7, (group + 1) * 7), key=lambda h: norms[h]) for group in range(2)]
indices = torch.tensor([i for head in heads for i in range(head * 64, (head + 1) * 64)])
assert heads == [6, 7]
A21 = torch.nn.functional.silu(X21 @ G.T) * (X21 @ U.T)
native_H = H22 - A21 @ D.T
reconstructed_X = native_H * torch.rsqrt(native_H.square().mean(-1, keepdim=True) + 1e-6) * g
reconstruction_error = float((reconstructed_X - X21).norm() / X21.norm())
assert reconstruction_error < 1e-5
B = O[:, indices].double()
left, _, right = torch.linalg.svd(B, full_matrices=False)
J = left @ right
mu = means[train].double().mean(0)
P = torch.eye(len(mu), dtype=torch.float64) - torch.outer(mu, mu) / (mu @ mu)
mapped = means.double() @ (J @ P).T
scale = (native_H[train].double() - native_H[train].double().mean(0)).norm() / mapped[train].norm()
after_O = (scale * J @ P).float()
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'layer': 21, 'heads': heads, 'original_output_column_norms': norms,
    'tensors': ['model.layers.21.self_attn.q_proj.weight',
                'model.layers.21.self_attn.q_proj.bias',
                'model.layers.21.self_attn.o_proj.weight'],
    'formula': 'Set selected Q rows and biases to zero. B=original selected O columns; J=polar(B); mu=128-dimensional prototype prefix-value mean; P=I-mu*muT/||mu||^2. O_selected=scale*J*P, where scale matches the energy of centered native pre-MLP block-21 residuals on the 64 earlier prototypes.',
    'reason': 'Block-21 native prefix values identify 22/24 third relations while final MLP features identify 8/24. The earlier uniform-Q-only intervention did not transmit this information. Test a centered output map at block 21 with the same predetermined energy rule used previously at block 23.',
    'head_selection': 'Minimum original O-column norm within each KV group.',
    'construction_uses_fact_labels': False, 'automatic_parameter_search': False,
    'training': False, 'gradients': False, 'optimizer': False,
    'diagnostic_only': True, 'free_generation_performed': False,
    'third_set_already_observed': True, 'final_set_read': False,
}
(ROOT / f'patches/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
save_file({'indices': indices, 'q_weight_before': Q[indices].contiguous(),
           'q_bias_before': b[indices].contiguous(), 'o_before': O[:, indices].contiguous(),
           'o_after': after_O.contiguous()}, ROOT / f'patches/{STEM}_basis.safetensors')
changed_H = native_H - native_heads[:, heads].flatten(1) @ O[:, indices].T + means @ after_O.T
changed_X21 = changed_H * torch.rsqrt(changed_H.square().mean(-1, keepdim=True) + 1e-6) * g
changed_A21 = torch.nn.functional.silu(changed_X21 @ G.T) * (changed_X21 @ U.T)
expected_H22 = changed_H + changed_A21 @ D.T
print('Heads', heads, 'scale', float(scale), 'O norm', float(B.norm()), '->', float(after_O.norm()), flush=True)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
m = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
attn = m.model.layers[21].self_attn
assert torch.equal(attn.q_proj.weight[indices], Q[indices])
assert torch.equal(attn.q_proj.bias[indices], b[indices])
assert torch.equal(attn.o_proj.weight[:, indices], O[:, indices])
attn.q_proj.weight[indices] = 0
attn.q_proj.bias[indices] = 0
attn.o_proj.weight[:, indices] = after_O
captured = {}


def input_hook(name):
    def capture(module, args):
        captured[name] = args[0][:, -1].clone()
    return capture


def output_hook(module, args, output):
    value = output[0] if isinstance(output, tuple) else output
    captured['H22'] = value[:, -1].clone()


hooks = [m.model.layers[21].register_forward_hook(output_hook),
         m.model.layers[23].mlp.register_forward_pre_hook(input_hook('X23')),
         m.model.layers[23].mlp.down_proj.register_forward_pre_hook(input_hook('A23')),
         m.model.norm.register_forward_pre_hook(input_hook('R'))]
stored = {name: [] for name in ['H22', 'X23', 'A23', 'R']}
checks = []
for begin in range(0, len(rows), 4):
    batch = rows[begin:begin + 4]
    width = max(len(row['ids']) for row in batch)
    ids = torch.tensor([[t.pad_token_id] * (width - len(row['ids'])) + row['ids'] for row in batch])
    mask = torch.tensor([[0] * (width - len(row['ids'])) + [1] * len(row['ids']) for row in batch])
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    m.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False)
    for name in stored:
        stored[name].append(captured[name].numpy())
    if begin < len(old_rows):
        for j, row in enumerate(batch):
            expected = expected_H22[begin + j]
            error = float((captured['H22'][j] - expected).norm() / expected.norm())
            assert error < 1e-4, (row['id'], error)
            checks.append({'id': row['id'], 'relative_block21_output_error': error})
    if begin % 16 == 0:
        print('Actual weight forwards', begin + len(batch), '/', len(rows), flush=True)
for hook in hooks:
    hook.remove()
changed = {name: torch.from_numpy(np.concatenate(value)) for name, value in stored.items()}
np.savez_compressed(ROOT / f'results/{STEM}_states.npz', **{name: value.numpy() for name, value in changed.items()})
(ROOT / f'results/{STEM}_states.json').write_text(
    json.dumps({'rows': rows, 'spec': spec}, ensure_ascii=False, indent=2), encoding='utf8')
attn.q_proj.weight[indices] = Q[indices]
attn.q_proj.bias[indices] = b[indices]
attn.o_proj.weight[:, indices] = O[:, indices]
assert torch.equal(attn.q_proj.weight[indices], Q[indices])
assert torch.equal(attn.o_proj.weight[:, indices], O[:, indices])
with np.load(ROOT / 'results/third_question_features_states.npz') as z:
    original_X23 = torch.cat([original_X23, torch.from_numpy(z['last_mlp_input'].copy())])
    original_R = torch.cat([original_R, torch.from_numpy(z['final_residual'].copy())])
last_mlp = m.model.layers[23].mlp
original_A23 = (torch.nn.functional.silu(original_X23 @ last_mlp.gate_proj.weight.T)
                 * (original_X23 @ last_mlp.up_proj.weight.T))
del m, last_mlp, attn, captured, stored


def normalize(value):
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)


labels = [row['label'] for row in rows]
classes = sorted(set(labels))
earlier = [i for i, row in enumerate(old_rows) if row['split'] == 'probe']
third = list(range(len(old_rows), len(rows)))
probes = []
for name, features in [('original_X23', original_X23), ('changed_X23', changed['X23']),
                       ('original_A23', original_A23), ('changed_A23', changed['A23']),
                       ('original_R', original_R), ('changed_R', changed['R'])]:
    mean = features[train].mean(0)
    centers = torch.stack([features[[i for i in train if labels[i] == label]].mean(0) for label in classes])
    for method, reference, reference_labels in [('class_mean', centers, classes),
                                                ('nearest_example', features[train], [labels[i] for i in train])]:
        winners = (normalize(features - mean) @ normalize(reference - mean).T).argmax(-1).tolist()
        predictions = [reference_labels[i] for i in winners]
        for dataset, selected in [('earlier_probe', earlier), ('observed_third', third)]:
            probe = {'feature': name, 'method': method, 'dataset': dataset,
                     'target_correct': sum(predictions[i] == labels[i] for i in selected if labels[i] != 'outside'),
                     'outside_correct': sum(predictions[i] == 'outside' for i in selected if labels[i] == 'outside'),
                     'predictions': [{'id': rows[i]['id'], 'expected': labels[i], 'predicted': predictions[i]} for i in selected]}
            probes.append(probe)
            print({key: value for key, value in probe.items() if key != 'predictions'}, flush=True)
shifts = (changed['R'] - original_R).norm(dim=-1)
report = {
    **spec, 'scale': float(scale), 'native_input_reconstruction_error': reconstruction_error,
    'selected_O_norm_before': float(B.norm()), 'selected_O_norm_after': float(after_O.norm()),
    'O_change_norm': float((after_O - B.float()).norm()),
    'changed_parameters': int(torch.count_nonzero(Q[indices]) + torch.count_nonzero(b[indices])
                              + torch.count_nonzero(after_O - O[:, indices])),
    'actual_weight_forward_checks': checks,
    'largest_block21_output_error': max(row['relative_block21_output_error'] for row in checks),
    'mean_final_residual_shift_targets': float(shifts[[i for i in earlier + third if labels[i] != 'outside']].mean()),
    'mean_final_residual_shift_controls': float(shifts[[i for i in earlier + third if labels[i] == 'outside']].mean()),
    'probes': probes, 'weights_restored': True, 'counts_as_correction': False,
    'interpretation': 'Actual attention weights were changed for forward diagnostics and restored in memory. Probe classification is not generated factual accuracy. The third group is already observed; no fresh correction result is claimed.',
}
(ROOT / f'results/{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({key: value for key, value in report.items() if key not in ['probes', 'actual_weight_forward_checks']}, flush=True)
