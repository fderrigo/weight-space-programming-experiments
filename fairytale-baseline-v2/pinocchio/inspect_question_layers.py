"""Read-only map of factual-question information across original layers.

Only previously used development questions are read. Class means and nearest
examples are descriptive probes, not trained models or deployed routing. The
many layer/position comparisons are exploratory, not independent evaluation.
"""
import collections
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from readout_contexts import load_contexts

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'question_layer_diagnostic'
torch.set_grad_enabled(False)
torch.set_num_threads(4)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
states_path = ROOT / f'results/{STEM}_states.npz'
meta_path = ROOT / f'results/{STEM}_states.json'


def fact(name):
    name = name.removesuffix('_answer_start')
    return {'fairy_singular': 'fairy', 'ending_transition': 'ending',
            'wood_finder': 'wood_donor'}.get(name, name)


def normalize(rows):
    return rows / rows.norm(dim=-1, keepdim=True).clamp_min(1e-12)


if not states_path.exists():
    _, _, records, _ = load_contexts(ROOT, t)
    rows = []
    for record in records:
        if record['prefix'] or record['subtoken'] != 0:
            continue
        rows.append({'id': record['question'], 'label': fact(record['fact']),
                     'split': 'probe' if record['question'].startswith('second_') else 'prototype',
                     'ids': record['history']})
    native = json.loads((ROOT / 'results/bounded_native_inputs.json').read_text('utf8'))
    for row in native['sequences']:
        if row['kind'] != 'external':
            continue
        assert row['source'] in ['baseline.json', 'newdevelopment_original.json']
        rows.append({'id': row['id'], 'label': 'outside',
                     'split': 'prototype' if row['source'] == 'baseline.json' else 'probe',
                     'ids': row['ids'][:row['start'] + 1]})
    assert len(rows) == 108
    user_header = t.encode('<|im_start|>user\n', add_special_tokens=False)
    for row in rows:
        ids = row['ids']
        starts = [j + len(user_header) for j in range(len(ids) - len(user_header) + 1)
                  if ids[j:j + len(user_header)] == user_header]
        assert len(starts) == 1
        start = starts[0]
        end = ids.index(t.convert_tokens_to_ids('<|im_end|>'), start)
        assert start < end < len(ids)
        row['user_start'], row['user_end'] = start, end
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
    X, H, captures = [], [], {}
    active, width = None, None

    def pool(value):
        pooled = []
        for j, row in enumerate(active):
            shift = width - len(row['ids'])
            user = value[j, shift + row['user_start']:shift + row['user_end']]
            pooled.append(torch.stack([value[j, -1], user[-1], user.mean(0)]))
        return torch.stack(pooled).clone()

    def xhook(index):
        def capture(module, args):
            captures[index] = pool(args[0])
        return capture

    def raw_last_hook(module, args):
        captures['raw_last'] = pool(args[0])

    hooks = [layer.mlp.register_forward_pre_hook(xhook(i))
             for i, layer in enumerate(m.model.layers)]
    hooks.append(m.model.norm.register_forward_pre_hook(raw_last_hook))
    for begin in range(0, len(rows), 4):
        active = rows[begin:begin + 4]
        width = max(len(row['ids']) for row in active)
        ids = torch.tensor([[t.pad_token_id] * (width - len(row['ids'])) + row['ids']
                            for row in active])
        mask = torch.tensor([[0] * (width - len(row['ids'])) + [1] * len(row['ids'])
                             for row in active])
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        output = m.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                         output_hidden_states=True, use_cache=False)
        # Transformers returns the final normalized state instead of raw block
        # 23. Save both, so all layer labels remain unambiguous.
        hs = [pool(h) for h in output.hidden_states[:-1]]
        hs += [captures['raw_last'], pool(output.hidden_states[-1])]
        H.append(torch.stack(hs, dim=2).numpy())
        X.append(torch.stack([captures[i] for i in range(24)], dim=2).numpy())
        if begin % 16 == 0:
            print('Native questions captured', begin + len(active), '/', len(rows), flush=True)
    for hook in hooks:
        hook.remove()
    arrays = {'H': np.concatenate(H), 'X': np.concatenate(X)}
    metadata = {'sequences': rows, 'positions': ['answer_start', 'user_last', 'user_mean'],
                'H_layers': ['embedding'] + [f'block_{i}_raw' for i in range(24)] + ['final_norm'],
                'X_layers': [f'mlp_{i}_input' for i in range(24)],
                'weights_changed': False, 'training': False, 'counts_as_correction': False,
                'third_set_read': False, 'final_set_read': False}
    np.savez_compressed(states_path, **arrays)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf8')
    del m, output, H, X, arrays

metadata = json.loads(meta_path.read_text('utf8'))
rows = metadata['sequences']
train = [i for i, row in enumerate(rows) if row['split'] == 'prototype']
targets = [i for i, row in enumerate(rows) if row['split'] == 'probe' and row['label'] != 'outside']
outside = [i for i, row in enumerate(rows) if row['split'] == 'probe' and row['label'] == 'outside']
assert len(train) == 64 and len(targets) == 24 and len(outside) == 20
classes = sorted(set(row['label'] for row in rows))
labels = [row['label'] for row in rows]
probes = []
with np.load(states_path) as arrays:
    for kind in ['H', 'X']:
        values = torch.from_numpy(arrays[kind].copy())
        for position_index, position in enumerate(metadata['positions']):
            for layer_index, layer in enumerate(metadata[kind + '_layers']):
                F = values[:, position_index, layer_index]
                mean = F[train].mean(0)
                centers = torch.stack([F[[i for i in train if labels[i] == label]].mean(0)
                                       for label in classes])
                for method, prototypes, prototype_labels in [
                    ('class_mean', centers, classes),
                    ('nearest_example', F[train], [labels[i] for i in train]),
                ]:
                    scores = normalize(F - mean) @ normalize(prototypes - mean).T
                    predictions = [prototype_labels[i] for i in scores.argmax(-1).tolist()]
                    counts = {'target_correct': sum(predictions[i] == labels[i] for i in targets),
                              'outside_correct': sum(predictions[i] == 'outside' for i in outside)}
                    probes.append({'representation': kind, 'position': position, 'layer': layer,
                                   'method': method, **counts,
                                   'balanced_accuracy': .5 * (counts['target_correct'] / 24
                                                              + counts['outside_correct'] / 20),
                                   'predictions': [{'id': rows[i]['id'], 'expected': labels[i],
                                                    'predicted': predictions[i]}
                                                   for i in targets + outside]})
probes.sort(key=lambda r: (-r['balanced_accuracy'], -r['target_correct']))
report = {'weights_changed': False, 'training': False, 'gradients': False, 'optimizer': False,
          'third_set_read': False, 'final_set_read': False, 'counts_as_correction': False,
          'interpretation': 'Exploratory question-identity probes on development only; no answer generation or factual knowledge claim. Layers and positions are compared descriptively, with no learned classifier.',
          'prototype_count': len(train), 'target_count': len(targets), 'outside_count': len(outside),
          'prototype_labels': dict(collections.Counter(labels[i] for i in train)),
          'probe_count': len(probes), 'probes': probes}
(ROOT / f'results/{STEM}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print('Top exploratory probes:', flush=True)
for row in probes[:15]:
    print({k: v for k, v in row.items() if k != 'predictions'}, flush=True)
