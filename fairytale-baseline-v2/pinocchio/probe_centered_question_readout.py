"""Declared algebraic question-contrast readout in the last attention block.

The two uniform KV means have useful question information, but zeroing queries
alone did not improve final-state probes. Remove their common prototype mean,
use the closest isometric output map, and match the native centered residual
energy. No labels enter this transformation, no optimizer or derivatives exist.
"""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'centered_question_readout'
assert not (ROOT / f'patches/{STEM}_basis.safetensors').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(6)
metadata_path = ROOT / 'results/question_layer_diagnostic_states.json'
rows = json.loads(metadata_path.read_text('utf8'))['sequences']
prototypes = [i for i, row in enumerate(rows) if row['split'] == 'prototype']
targets = [i for i, row in enumerate(rows) if row['split'] == 'probe' and row['label'] != 'outside']
controls = [i for i, row in enumerate(rows) if row['split'] == 'probe' and row['label'] == 'outside']
assert (len(rows), len(prototypes), len(targets), len(controls)) == (108, 64, 24, 20)
with np.load(ROOT / 'results/question_layer_diagnostic_states.npz') as z:
    X = torch.from_numpy(z['X'][:, 0, 23].copy())
    R = torch.from_numpy(z['H'][:, 0, 24].copy())
with np.load(ROOT / 'results/attention_feature_diagnostic_states.npz') as z:
    native_heads = torch.from_numpy(z['heads'][:, 23].copy())
    means = torch.from_numpy(z['prefix_values'][:, 23].copy()).flatten(1)
with safe_open(MODEL / 'model.safetensors', framework='pt') as source:
    O = source.get_tensor('model.layers.23.self_attn.o_proj.weight').float()
    Q = source.get_tensor('model.layers.23.self_attn.q_proj.weight').float()
    b = source.get_tensor('model.layers.23.self_attn.q_proj.bias').float()
    G = source.get_tensor('model.layers.23.mlp.gate_proj.weight').float()
    U = source.get_tensor('model.layers.23.mlp.up_proj.weight').float()
    D = source.get_tensor('model.layers.23.mlp.down_proj.weight').float()
    g = source.get_tensor('model.layers.23.post_attention_layernorm.weight').float()
norms = [float(O[:, 64 * h:64 * (h + 1)].norm()) for h in range(14)]
heads = [min(range(7 * group, 7 * (group + 1)), key=lambda h: norms[h])
         for group in range(2)]
indices = torch.tensor([i for head in heads for i in range(head * 64, (head + 1) * 64)])
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'layer': 23, 'heads': heads, 'original_output_column_norms': norms,
    'tensors': ['model.layers.23.self_attn.q_proj.weight',
                'model.layers.23.self_attn.q_proj.bias',
                'model.layers.23.self_attn.o_proj.weight'],
    'formula': 'Q rows and biases zero. Let B be the selected original O columns, mu the 128-dimensional prefix-value mean on 64 existing prototype questions, P=I-mu*muT/||mu||^2, J=polar(B). O_selected=scale*J*P. scale=||centered native pre-MLP residuals||F/||prototype prefix means projected by J*P||F.',
    'reason': 'Uniform attention alone exposed discriminative values but did not improve final representations. Test whether a centered, variance-matched output map can transmit the question contrast.',
    'head_selection': 'Smallest original O-column norm in each KV group.',
    'gain_rule_uses_probe_outcomes': False, 'construction_uses_fact_labels': False,
    'training': False, 'gradients': False, 'optimizer': False,
    'diagnostic_only': True, 'free_generation_performed': False,
    'third_set_read': False, 'final_set_read': False,
}
(ROOT / f'patches/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
A = torch.nn.functional.silu(X @ G.T) * (X @ U.T)
H = R - A @ D.T
reconstructed = H * torch.rsqrt(H.square().mean(-1, keepdim=True) + 1e-6) * g
input_error = float((reconstructed - X).norm() / X.norm())
assert input_error < 1e-5
B = O[:, indices].double()
left, _, right = torch.linalg.svd(B, full_matrices=False)
J = left @ right
mu = means[prototypes].double().mean(0)
P = torch.eye(len(mu), dtype=torch.float64) - torch.outer(mu, mu) / (mu @ mu)
mapped = means.double() @ (J @ P).T
native_centered = H[prototypes].double() - H[prototypes].double().mean(0)
scale = native_centered.norm() / mapped[prototypes].norm()
after_O = (scale * J @ P).float()
old_contribution = native_heads[:, heads].flatten(1) @ O[:, indices].T
changed_H = H - old_contribution + means @ after_O.T
changed_X = changed_H * torch.rsqrt(changed_H.square().mean(-1, keepdim=True) + 1e-6) * g
changed_A = torch.nn.functional.silu(changed_X @ G.T) * (changed_X @ U.T)
changed_R = changed_H + changed_A @ D.T
save_file({'indices': indices, 'q_weight_before': Q[indices].contiguous(),
           'q_bias_before': b[indices].contiguous(),
           'o_before': O[:, indices].contiguous(), 'o_after': after_O.contiguous()},
          ROOT / f'patches/{STEM}_basis.safetensors')
np.savez_compressed(ROOT / f'results/{STEM}_states.npz',
                    H=changed_H.numpy(), X=changed_X.numpy(), R=changed_R.numpy())

# Verify the algebra against actual weights and exact ordinary prompt IDs.
m = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
attn = m.model.layers[23].self_attn
attn.q_proj.weight[indices] = 0
attn.q_proj.bias[indices] = 0
attn.o_proj.weight[:, indices] = after_O
capture = {}


def capture_x(module, inputs):
    capture['X'] = inputs[0][0, -1].clone()


def capture_r(module, inputs):
    capture['R'] = inputs[0][0, -1].clone()


hooks = [m.model.layers[23].mlp.register_forward_pre_hook(capture_x),
         m.model.norm.register_forward_pre_hook(capture_r)]
checks = []
for i in [0, 63, 64, 107]:
    m.model(input_ids=torch.tensor([rows[i]['ids']]), use_cache=False)
    checks.append({'id': rows[i]['id'],
                   'relative_X_error': float((capture['X'] - changed_X[i]).norm() / changed_X[i].norm()),
                   'relative_R_error': float((capture['R'] - changed_R[i]).norm() / changed_R[i].norm())})
for hook in hooks:
    hook.remove()
attn.q_proj.weight[indices] = Q[indices]
attn.q_proj.bias[indices] = b[indices]
attn.o_proj.weight[:, indices] = O[:, indices]
assert all(check['relative_X_error'] < 1e-4 and check['relative_R_error'] < 1e-4 for check in checks)
del m


def normalize(value):
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)


labels = [row['label'] for row in rows]
classes = sorted(set(labels))
probes = []
for name, features in [('original_last_mlp_input', X), ('changed_last_mlp_input', changed_X),
                       ('original_final_residual', R), ('changed_final_residual', changed_R)]:
    mean = features[prototypes].mean(0)
    centers = torch.stack([features[[i for i in prototypes if labels[i] == label]].mean(0)
                           for label in classes])
    for method, vectors, vector_labels in [
        ('class_mean', centers, classes),
        ('nearest_example', features[prototypes], [labels[i] for i in prototypes]),
    ]:
        predictions = [vector_labels[i] for i in
                       (normalize(features - mean) @ normalize(vectors - mean).T).argmax(-1).tolist()]
        probes.append({'features': name, 'method': method,
                       'target_correct': sum(predictions[i] == labels[i] for i in targets),
                       'outside_correct': sum(predictions[i] == 'outside' for i in controls),
                       'predictions': [{'id': rows[i]['id'], 'expected': labels[i],
                                        'predicted': predictions[i]} for i in targets + controls]})
report = {
    'source_questions_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
    'heads': heads, 'scale': float(scale), 'native_input_reconstruction_error': input_error,
    'selected_O_norm_before': float(B.norm()), 'selected_O_norm_after': float(after_O.norm()),
    'O_change_norm': float((after_O - B.float()).norm()),
    'changed_parameters': int(torch.count_nonzero(Q[indices]) + torch.count_nonzero(b[indices])
                              + torch.count_nonzero(after_O - O[:, indices])),
    'mean_final_residual_shift_targets': float((changed_R[targets] - R[targets]).norm(dim=-1).mean()),
    'mean_final_residual_shift_controls': float((changed_R[controls] - R[controls]).norm(dim=-1).mean()),
    'actual_weight_forward_checks': checks, 'probes': probes,
    'training': False, 'gradients': False, 'optimizer': False,
    'weights_restored': True, 'free_generation_performed': False, 'counts_as_correction': False,
    'third_set_read': False, 'final_set_read': False,
}
(ROOT / f'results/{STEM}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k != 'probes'}, flush=True)
for row in probes:
    print({k: v for k, v in row.items() if k != 'predictions'}, flush=True)
