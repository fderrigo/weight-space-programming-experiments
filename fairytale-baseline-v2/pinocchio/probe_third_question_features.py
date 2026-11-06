"""Read-only question-information probes after the frozen candidate failed.

Feature choices and the 64 reference questions come from earlier diagnostics.
The now-observed third group is diagnostic data, not a fresh correction test.
No learned probe, weight edit, response prefix, or final-set access is used.
"""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'third_question_features'
assert not (ROOT / f'results/{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
old_rows = json.loads((ROOT / 'results/question_layer_diagnostic_states.json').read_text('utf8'))['sequences']
train = [i for i, row in enumerate(old_rows) if row['split'] == 'prototype']
assert len(train) == 64
questions = json.loads((ROOT / 'third_development_questions.json').read_text('utf8'))
assert len(questions) == 44
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'features': ['native_head_14_13', 'prefix_values_21', 'prefix_values_23', 'last_mlp_input', 'last_mlp_features'],
    'methods': ['class_mean', 'nearest_example'], 'prototype_count': 64,
    'reason': 'Compare previously identified attention information with final MLP features on the now-observed paraphrases after a frozen direct-write candidate failed. Do not select heads or prototypes using the third labels.',
    'reference_selection': 'The same 64 original prototypes used before the second-development feature probes.',
    'weights_changed': False, 'training': False, 'gradients': False, 'optimizer': False,
    'third_set_already_observed': True, 'fresh_correction_evaluation': False,
    'final_set_read': False, 'counts_as_correction': False,
}
(ROOT / f'results/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
rows = []
for row in questions:
    ids = t.apply_chat_template([{'role': 'user', 'content': row['prompt']}], add_generation_prompt=True)
    rows.append({**row, 'label': row['fact'] if row['group'] == 'target' else 'outside', 'ids': ids})
m = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
with np.load(ROOT / 'results/attention_feature_diagnostic_states.npz') as z:
    old_features = {'native_head_14_13': torch.from_numpy(z['heads'][:, 14, 13].copy()),
                    'prefix_values_21': torch.from_numpy(z['prefix_values'][:, 21].copy()).flatten(1),
                    'prefix_values_23': torch.from_numpy(z['prefix_values'][:, 23].copy()).flatten(1)}
with np.load(ROOT / 'results/question_layer_diagnostic_states.npz') as z:
    old_features['last_mlp_input'] = torch.from_numpy(z['X'][:, 0, 23].copy())
mlp = m.model.layers[23].mlp
old_features['last_mlp_features'] = (torch.nn.functional.silu(old_features['last_mlp_input'] @ mlp.gate_proj.weight.T)
                                      * (old_features['last_mlp_input'] @ mlp.up_proj.weight.T))
active, width, captured = None, None, {}


def head_hook(module, args):
    captured['native_head_14_13'] = args[0][:, -1].reshape(-1, 14, 64)[:, 13].clone()


def value_hook(name):
    def capture(module, args, output):
        captured[name] = torch.stack([output[j, width - len(row['ids']):].mean(0)
                                      for j, row in enumerate(active)])
    return capture


def input_hook(name):
    def capture(module, args):
        captured[name] = args[0][:, -1].clone()
    return capture


hooks = [m.model.layers[14].self_attn.o_proj.register_forward_pre_hook(head_hook),
         m.model.layers[21].self_attn.v_proj.register_forward_hook(value_hook('prefix_values_21')),
         m.model.layers[23].self_attn.v_proj.register_forward_hook(value_hook('prefix_values_23')),
         mlp.register_forward_pre_hook(input_hook('last_mlp_input')),
         mlp.down_proj.register_forward_pre_hook(input_hook('last_mlp_features')),
         m.model.norm.register_forward_pre_hook(input_hook('final_residual'))]


def forward(batch):
    global active, width
    active = batch
    width = max(len(row['ids']) for row in batch)
    ids = torch.tensor([[t.pad_token_id] * (width - len(row['ids'])) + row['ids'] for row in batch])
    mask = torch.tensor([[0] * (width - len(row['ids'])) + [1] * len(row['ids']) for row in batch])
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    m.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False)


checks = []
for i in [0, 63]:
    forward([old_rows[i]])
    errors = {name: float((captured[name][0] - features[i]).norm() / features[i].norm())
              for name, features in old_features.items()}
    assert max(errors.values()) < 1e-4, (i, errors)
    checks.append({'id': old_rows[i]['id'], 'relative_feature_errors': errors})
stored = {name: [] for name in [*old_features, 'final_residual']}
for begin in range(0, len(rows), 4):
    forward(rows[begin:begin + 4])
    for name in stored:
        stored[name].append(captured[name].numpy())
    print('Original question features', min(begin + 4, len(rows)), '/', len(rows), flush=True)
for hook in hooks:
    hook.remove()
new_features = {name: torch.from_numpy(np.concatenate(parts)) for name, parts in stored.items()}
np.savez_compressed(ROOT / f'results/{STEM}_states.npz',
                    **{name: value.numpy() for name, value in new_features.items()})
(ROOT / f'results/{STEM}_states.json').write_text(
    json.dumps({'rows': rows, 'spec': spec, 'reference_forward_checks': checks}, ensure_ascii=False, indent=2), encoding='utf8')
del m, mlp, captured, stored


def normalize(value):
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)


labels = [row['label'] for row in old_rows]
classes = sorted(set(labels))
probes = []
for name, old in old_features.items():
    new = new_features[name]
    mean = old[train].mean(0)
    centers = torch.stack([old[[i for i in train if labels[i] == label]].mean(0) for label in classes])
    for method, reference, reference_labels, reference_ids in [
        ('class_mean', centers, classes, classes),
        ('nearest_example', old[train], [labels[i] for i in train], [old_rows[i]['id'] for i in train]),
    ]:
        scores = normalize(new - mean) @ normalize(reference - mean).T
        winners = scores.argmax(-1).tolist()
        predictions = [reference_labels[i] for i in winners]
        top_two = scores.topk(2, dim=-1).values
        probe = {
            'feature': name, 'method': method,
            'target_correct': sum(predictions[i] == row['label'] for i, row in enumerate(rows) if row['group'] == 'target'),
            'outside_correct': sum(predictions[i] == 'outside' for i, row in enumerate(rows) if row['group'] == 'control'),
            'target_total': 24, 'outside_total': 20,
            'predictions': [{'id': row['id'], 'expected': row['label'], 'predicted': predictions[i],
                             'reference': reference_ids[winners[i]], 'cosine': float(top_two[i, 0]),
                             'margin': float(top_two[i, 0] - top_two[i, 1])} for i, row in enumerate(rows)],
        }
        probes.append(probe)
        print({key: value for key, value in probe.items() if key != 'predictions'}, flush=True)
report = {
    **spec, 'probes': probes, 'reference_forward_checks': checks,
    'reference_metadata_sha256': hashlib.sha256((ROOT / 'results/question_layer_diagnostic_states.json').read_bytes()).hexdigest(),
    'third_question_sha256': hashlib.sha256((ROOT / 'third_development_questions.json').read_bytes()).hexdigest(),
    'interpretation': 'These are descriptive relation-identification probes on original-model states. They neither generate an answer nor demonstrate that a weight edit can use the information reliably.',
}
(ROOT / f'results/{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
