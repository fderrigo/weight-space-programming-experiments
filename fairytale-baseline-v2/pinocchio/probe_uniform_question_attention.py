"""Causal question-information probe with two uniform native attention heads.

Zero one existing Q head per KV group in block 21, selecting the head with the
smallest native output-column norm in each group. No new modules, training,
gradients, generated answer targets, reserved questions, or optimizer.
"""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'uniform_question_attention'
assert not (ROOT / f'results/{STEM}.json').exists()
assert not (ROOT / f'patches/{STEM}_basis.safetensors').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(6)
metadata = json.loads((ROOT / 'results/question_layer_diagnostic_states.json').read_text('utf8'))
rows = metadata['sequences']
assert len(rows) == 108
with np.load(ROOT / 'results/question_layer_diagnostic_states.npz') as z:
    original_X = torch.from_numpy(z['X'][:, 0, 23].copy())
    original_R = torch.from_numpy(z['H'][:, 0, 24].copy())
with np.load(ROOT / 'results/attention_feature_diagnostic_states.npz') as z:
    expected_means = torch.from_numpy(z['prefix_values'][:, 21].copy())
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
m = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
layer = m.model.layers[21]
norms = [float(layer.self_attn.o_proj.weight[:, 64 * h:64 * (h + 1)].norm())
         for h in range(14)]
heads = [min(range(7 * group, 7 * (group + 1)), key=lambda h: norms[h])
         for group in range(2)]
indices = torch.tensor([i for h in heads for i in range(64 * h, 64 * (h + 1))])
q_weight, q_bias = layer.self_attn.q_proj.weight, layer.self_attn.q_proj.bias
before_w, before_b = q_weight[indices].clone(), q_bias[indices].clone()
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'decision_by': 'Codex', 'layer': 21, 'heads': heads,
    'tensors': ['model.layers.21.self_attn.q_proj.weight',
                'model.layers.21.self_attn.q_proj.bias'],
    'output_column_norms': norms,
    'selection_rule': 'One head per KV group, with the smallest original Frobenius norm of its O-projection columns; a conservative proxy, not a proof of low functional importance.',
    'transformation': 'Set the selected Q projection rows and their biases to zero. Rotary embedding of the zero query remains zero, so its causal attention is uniform over unmasked tokens.',
    'reason': 'Native block-21 prefix-mean values distinguish question relations better than the final native question state. Test whether making those means available through existing heads improves the final representation.',
    'training': False, 'gradients': False, 'optimizer': False,
    'diagnostic_only': True, 'free_generation_performed': False,
    'third_set_read': False, 'final_set_read': False,
}
(ROOT / f'patches/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
q_weight[indices] = 0
q_bias[indices] = 0
save_file({'rows': indices, 'weight_before': before_w.contiguous(),
           'weight_after': torch.zeros_like(before_w), 'bias_before': before_b.contiguous(),
           'bias_after': torch.zeros_like(before_b)}, ROOT / f'patches/{STEM}_basis.safetensors')
print('Uniform heads', heads, 'changed values',
      int(torch.count_nonzero(before_w) + torch.count_nonzero(before_b)), flush=True)
capture = {}


def xhook(module, inputs):
    capture['X'] = inputs[0][:, -1].clone()


def rhook(module, inputs):
    capture['R'] = inputs[0][:, -1].clone()


def hhook(module, inputs):
    capture['heads'] = inputs[0][:, -1].reshape(-1, 14, 64)[:, heads].clone()


hooks = [m.model.layers[23].mlp.register_forward_pre_hook(xhook),
         m.model.norm.register_forward_pre_hook(rhook),
         layer.self_attn.o_proj.register_forward_pre_hook(hhook)]
stored = {'X': [], 'R': [], 'heads': []}
for begin in range(0, len(rows), 4):
    active = rows[begin:begin + 4]
    width = max(len(row['ids']) for row in active)
    ids = torch.tensor([[t.pad_token_id] * (width - len(row['ids'])) + row['ids'] for row in active])
    mask = torch.tensor([[0] * (width - len(row['ids'])) + [1] * len(row['ids']) for row in active])
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    m.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False)
    for name in stored:
        stored[name].append(capture[name].numpy())
    if begin % 16 == 0:
        print('Causal feature probe', begin + len(active), '/', len(rows), flush=True)
for hook in hooks:
    hook.remove()
q_weight[indices] = before_w
q_bias[indices] = before_b
assert torch.equal(q_weight[indices], before_w) and torch.equal(q_bias[indices], before_b)
arrays = {key: np.concatenate(value) for key, value in stored.items()}
np.savez_compressed(ROOT / f'results/{STEM}_states.npz', **arrays)
changed_X = torch.from_numpy(arrays['X'])
changed_R = torch.from_numpy(arrays['R'])
uniform_heads = torch.from_numpy(arrays['heads'])
prototypes = [i for i, row in enumerate(rows) if row['split'] == 'prototype']
targets = [i for i, row in enumerate(rows) if row['split'] == 'probe' and row['label'] != 'outside']
controls = [i for i, row in enumerate(rows) if row['split'] == 'probe' and row['label'] == 'outside']
labels = [row['label'] for row in rows]
classes = sorted(set(labels))
assert len(prototypes) == 64 and len(targets) == 24 and len(controls) == 20


def normalize(value):
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)


probes = []
for name, features in [('original_last_mlp_input', original_X),
                       ('uniform_heads_last_mlp_input', changed_X),
                       ('original_final_residual', original_R),
                       ('uniform_heads_final_residual', changed_R),
                       ('actual_uniform_heads', uniform_heads.flatten(1))]:
    mean = features[prototypes].mean(0)
    centers = torch.stack([features[[i for i in prototypes if labels[i] == label]].mean(0)
                           for label in classes])
    for method, values, value_labels in [
        ('class_mean', centers, classes),
        ('nearest_example', features[prototypes], [labels[i] for i in prototypes]),
    ]:
        scores = normalize(features - mean) @ normalize(values - mean).T
        predictions = [value_labels[i] for i in scores.argmax(-1).tolist()]
        probes.append({'features': name, 'method': method,
                       'target_correct': sum(predictions[i] == labels[i] for i in targets),
                       'outside_correct': sum(predictions[i] == 'outside' for i in controls),
                       'predictions': [{'id': rows[i]['id'], 'expected': labels[i],
                                        'predicted': predictions[i]} for i in targets + controls]})
head_error = (uniform_heads - expected_means).norm() / expected_means.norm()
assert head_error < 1e-5, 'Actual zero-query heads must reproduce uniform native value means'
report = {
    **{key: spec[key] for key in ['training', 'gradients', 'optimizer', 'third_set_read', 'final_set_read']},
    'source_questions_sha256': hashlib.sha256((ROOT / 'results/question_layer_diagnostic_states.json').read_bytes()).hexdigest(),
    'actual_weights_changed_in_memory': True, 'weights_restored': True,
    'changed_parameters': int(torch.count_nonzero(before_w) + torch.count_nonzero(before_b)),
    'selected_heads': heads, 'actual_uniform_mean_relative_error': float(head_error),
    'prototype_count': len(prototypes), 'target_count': len(targets), 'outside_count': len(controls),
    'mean_final_residual_shift_targets': float((changed_R[targets] - original_R[targets]).norm(dim=-1).mean()),
    'mean_final_residual_shift_controls': float((changed_R[controls] - original_R[controls]).norm(dim=-1).mean()),
    'probes': probes, 'free_generation_performed': False, 'counts_as_correction': False,
    'interpretation': 'Exploratory paired feature diagnostics on previously used development questions. Recognizing the question relation is not evidence of knowing or correctly generating its answer.',
}
(ROOT / f'results/{STEM}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({key: value for key, value in report.items() if key != 'probes'}, flush=True)
for probe in probes:
    print({key: value for key, value in probe.items() if key != 'predictions'}, flush=True)
