"""Describe native question information before attention output projection.

Only existing development questions; no edits, training, or reserved questions.
Head probes are exploratory correlations, not evidence of causal fact storage.
"""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'attention_feature_diagnostic'
torch.set_grad_enabled(False)
torch.set_num_threads(4)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
metadata = json.loads((ROOT / 'results/question_layer_diagnostic_states.json').read_text('utf8'))
rows = metadata['sequences']
assert len(rows) == 108
states_path = ROOT / f'results/{STEM}_states.npz'
if not states_path.exists():
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
    active, width = None, None
    captured, stored = {}, {'heads': [], 'user_values': [], 'prefix_values': []}

    def head_hook(index):
        def capture(module, args):
            captured[('heads', index)] = args[0][:, -1].reshape(-1, 14, 64).clone()
        return capture

    def value_hook(index):
        def capture(module, args, output):
            user, prefix = [], []
            for j, row in enumerate(active):
                shift = width - len(row['ids'])
                user.append(output[j, shift + row['user_start']:shift + row['user_end']].mean(0))
                prefix.append(output[j, shift:].mean(0))
            captured[('user_values', index)] = torch.stack(user).reshape(-1, 2, 64).clone()
            captured[('prefix_values', index)] = torch.stack(prefix).reshape(-1, 2, 64).clone()
        return capture

    hooks = []
    for index, layer in enumerate(m.model.layers):
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(head_hook(index)))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(value_hook(index)))
    for begin in range(0, len(rows), 4):
        active = rows[begin:begin + 4]
        width = max(len(row['ids']) for row in active)
        ids = torch.tensor([[t.pad_token_id] * (width - len(row['ids'])) + row['ids'] for row in active])
        mask = torch.tensor([[0] * (width - len(row['ids'])) + [1] * len(row['ids']) for row in active])
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        m.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False)
        for name in stored:
            stored[name].append(torch.stack([captured[(name, i)] for i in range(24)], dim=1).numpy())
        if begin % 16 == 0:
            print('Native attention captured', begin + len(active), '/', len(rows), flush=True)
    for hook in hooks:
        hook.remove()
    np.savez_compressed(states_path, **{key: np.concatenate(value) for key, value in stored.items()})
    del m, stored, captured

def normalize(value):
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)


train = [i for i, row in enumerate(rows) if row['split'] == 'prototype']
targets = [i for i, row in enumerate(rows) if row['split'] == 'probe' and row['label'] != 'outside']
controls = [i for i, row in enumerate(rows) if row['split'] == 'probe' and row['label'] == 'outside']
classes = sorted({row['label'] for row in rows})
labels = [row['label'] for row in rows]
assert len(train) == 64 and len(targets) == 24 and len(controls) == 20
probes = []


def probe(F, identity):
    mean = F[train].mean(0)
    centers = torch.stack([F[[i for i in train if labels[i] == label]].mean(0) for label in classes])
    for method, prototype, prototype_labels in [
        ('class_mean', centers, classes),
        ('nearest_example', F[train], [labels[i] for i in train]),
    ]:
        scores = normalize(F - mean) @ normalize(prototype - mean).T
        predictions = [prototype_labels[i] for i in scores.argmax(-1).tolist()]
        correct = sum(predictions[i] == labels[i] for i in targets)
        outside = sum(predictions[i] == 'outside' for i in controls)
        probes.append({**identity, 'method': method, 'target_correct': correct,
                       'outside_correct': outside,
                       'balanced_accuracy': .5 * (correct / 24 + outside / 20),
                       'predictions': [{'id': rows[i]['id'], 'expected': labels[i], 'predicted': predictions[i]}
                                       for i in targets + controls]})


with np.load(states_path) as arrays, safe_open(MODEL / 'model.safetensors', framework='pt') as source:
    heads = torch.from_numpy(arrays['heads'].copy())
    for layer in range(24):
        W = source.get_tensor(f'model.layers.{layer}.self_attn.o_proj.weight').float()
        for head in range(14):
            F = heads[:, layer, head]
            probe(F, {'kind': 'native_head', 'layer': layer, 'head': head, 'metric': 'value_space'})
            probe(F @ W[:, head * 64:(head + 1) * 64].T,
                  {'kind': 'native_head', 'layer': layer, 'head': head, 'metric': 'residual_space'})
        probe(heads[:, layer].flatten(1),
              {'kind': 'all_native_heads', 'layer': layer, 'head': None, 'metric': 'value_space'})
        for kind in ['user_values', 'prefix_values']:
            values = torch.from_numpy(arrays[kind][:, layer].copy())
            for group in range(2):
                probe(values[:, group],
                      {'kind': kind, 'layer': layer, 'head': group, 'metric': 'value_space'})
            probe(values.flatten(1),
                  {'kind': kind, 'layer': layer, 'head': 'both', 'metric': 'value_space'})
probes.sort(key=lambda row: (-row['balanced_accuracy'], -row['target_correct']))
report = {'weights_changed': False, 'training': False, 'gradients': False,
          'third_set_read': False, 'final_set_read': False, 'counts_as_correction': False,
          'purpose': 'Exploratory description of factual-question information in native attention heads and uniform value means. No claim that a successful probe establishes knowledge of the answer or that an attention edit will work.',
          'source_questions': 'question_layer_diagnostic_states.json',
          'prototype_count': len(train), 'target_count': len(targets), 'outside_count': len(controls),
          'probe_count': len(probes), 'probes': probes}
(ROOT / f'results/{STEM}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
for kind in ['native_head', 'all_native_heads', 'user_values', 'prefix_values']:
    print('Best', kind, flush=True)
    for row in [row for row in probes if row['kind'] == kind][:5]:
        print({key: value for key, value in row.items() if key != 'predictions'}, flush=True)
