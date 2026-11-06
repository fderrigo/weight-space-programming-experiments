"""Read-only inspection of question identity and word-completion information.

Compare native states to descriptive mean-state directions. No network weights,
gradients, optimizers, routing, generated answers or reserved questions are used.
"""
import collections
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from readout_contexts import load_contexts

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
torch.set_grad_enabled(False)
torch.set_num_threads(4)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
K, R, records, basis = load_contexts(ROOT, t)
with safe_open(MODEL / 'model.safetensors', framework='pt') as source:
    post_norm = source.get_tensor('model.layers.23.post_attention_layernorm.weight').float()
    gate = source.get_tensor('model.layers.23.mlp.gate_proj.weight').float()
    up = source.get_tensor('model.layers.23.mlp.up_proj.weight').float()
residual_before_mlp = R - K @ basis['before'].T
X = residual_before_mlp / (residual_before_mlp.square().mean(-1, keepdim=True) + 1e-6).sqrt() * post_norm
reconstructed = torch.nn.functional.silu(X @ gate.T) * (X @ up.T)
relative_reconstruction_error = float((reconstructed - K).norm() / K.norm())
assert relative_reconstruction_error < 1e-5


def fact(name):
    name = name.removesuffix('_answer_start')
    return {'fairy_singular': 'fairy', 'ending_transition': 'ending',
            'wood_finder': 'wood_donor'}.get(name, name)


labels = [fact(r['fact']) for r in records]
external_meta = json.loads((ROOT / 'results/bounded_native_inputs.json').read_text('utf8'))
with np.load(ROOT / 'results/bounded_native_inputs.npz') as z:
    external_states = torch.from_numpy(z['X'].copy()).float()
external, ext_x, offset = [], [], 0
for row in external_meta['sequences']:
    count = len(row['ids']) - row['start']
    if row['kind'] == 'external':
        external.append({'id': row['id'], 'source': row['source']})
        ext_x.append(external_states[offset])
    offset += count
assert offset == len(external_states)
ext_x = torch.stack(ext_x)
ext_a = torch.nn.functional.silu(ext_x @ gate.T) * (ext_x @ up.T)
train = [i for i, row in enumerate(records)
         if not row['question'].startswith('second_') and row['prefix'] == '' and row['subtoken'] == 0]
second_start = [i for i, row in enumerate(records)
                if row['question'].startswith('second_') and row['prefix'] == '' and row['subtoken'] == 0]
outside_train = [i for i, r in enumerate(external) if r['source'] == 'baseline.json']
outside_test = [i for i, r in enumerate(external) if r['source'] == 'newdevelopment_original.json']
assert len(second_start) == 24 and len(outside_test) == 20

native_meta = json.loads((ROOT / 'results/native_second_continuations.json').read_text('utf8'))
native_x, native_a, native_labels = [], [], []
with np.load(ROOT / 'results/native_second_continuations_states.npz') as z:
    for i, row in enumerate(native_meta):
        native_x.append(torch.from_numpy(z[f'X_{i}'].copy()).float())
        native_a.append(torch.from_numpy(z[f'A_{i}'].copy()).float())
        native_labels.extend([fact(row['fact'])] * len(native_x[-1]))
native_x, native_a = torch.cat(native_x), torch.cat(native_a)
report = {
    'weights_changed': False, 'training': False, 'gradients': False, 'optimizer': False,
    'third_set_read': False, 'final_set_read': False, 'counts_as_correction': False,
    'purpose': 'Inspect whether question identity and lexical progress are separately readable from native states before designing another edit. Similarity probes are descriptive analysis, not a deployed classifier.',
    'native_activation_reconstruction_relative_error': relative_reconstruction_error,
    'factual_prototype_questions': len(train), 'outside_prototype_questions': len(outside_train),
    'factual_counts': dict(collections.Counter(labels[i] for i in train)),
    'readouts': {},
}


def normalize(rows):
    return rows / rows.norm(dim=-1, keepdim=True).clamp_min(1e-12)


for name, F, external_F, native_F in [('mlp_input', X, ext_x, native_x),
                                     ('native_mlp_activation', K, ext_a, native_a)]:
    classes = sorted(set(labels[i] for i in train)) + ['outside']
    training = torch.cat([F[train], external_F[outside_train]])
    mean = training.mean(0)
    centers = torch.stack([F[[i for i in train if labels[i] == label]].mean(0)
                            for label in classes[:-1]]
                           + [external_F[outside_train].mean(0)])
    directions = normalize(centers - mean)

    def classify(values, truth, ids):
        scores = normalize(values - mean) @ directions.T
        predictions = scores.argmax(-1).tolist()
        return [{'id': row_id, 'expected': true, 'predicted': classes[pred],
                 'correct': classes[pred] == true,
                 'score_margin': float(scores[i, pred] - scores[i].topk(2).values[1])}
                for i, (row_id, true, pred) in enumerate(zip(ids, truth, predictions))]

    targets = classify(F[second_start], [labels[i] for i in second_start],
                       [records[i]['question'] for i in second_start])
    controls = classify(external_F[outside_test], ['outside'] * len(outside_test),
                        [external[i]['id'] for i in outside_test])
    native = classify(native_F, native_labels, list(range(len(native_F))))
    # A heterogeneous outside category need not have one useful mean. Compare
    # also with individual native examples, using the same prototype sources.
    example_values = torch.cat([F[train], external_F[outside_train]])
    example_labels = [labels[i] for i in train] + ['outside'] * len(outside_train)
    example_directions = normalize(example_values - mean)
    nearest_fact = (normalize(F[second_start] - mean) @ example_directions.T).argmax(-1)
    nearest_outside = (normalize(external_F[outside_test] - mean) @ example_directions.T).argmax(-1)
    nearest_native = (normalize(native_F - mean) @ example_directions.T).argmax(-1)
    # Lexical progress is measured with the true fact supplied only to this
    # read-only analysis. It is not an inference-time routing mechanism.
    old_indices = list(range(694))
    word_groups = {}
    for i in old_indices:
        word_groups.setdefault((labels[i], records[i]['token']), []).append(i)
    field_means = {label: F[[i for i in old_indices if labels[i] == label]].mean(0)
                   for label in set(labels[i] for i in old_indices)}
    word_centers = {key: F[ix].mean(0) - field_means[key[0]] for key, ix in word_groups.items()}
    lexical = []
    for i in range(694, len(F)):
        label, token = labels[i], records[i]['token']
        keys = [key for key in word_groups if key[0] == label]
        direction = normalize(torch.stack([word_centers[key] for key in keys]))
        scores = normalize((F[i] - field_means[label])[None]) @ direction.T
        predicted = keys[int(scores.argmax())][1]
        lexical.append({'question': records[i]['question'], 'prefix': records[i]['prefix'],
                        'expected': records[i]['piece'], 'predicted': t.decode([predicted]),
                        'label_observed_previously': (label, token) in word_groups,
                        'correct': predicted == token})
    report['readouts'][name] = {
        'new_question_fact_identity_correct': sum(r['correct'] for r in targets),
        'new_question_fact_identity_total': len(targets),
        'new_external_rejected': sum(r['correct'] for r in controls),
        'new_external_total': len(controls),
        'native_continuation_fact_identity_correct': sum(r['correct'] for r in native),
        'native_continuation_states': len(native),
        'nearest_example_question_fact_correct': sum(example_labels[int(j)] == labels[i] for i, j in zip(second_start, nearest_fact)),
        'nearest_example_external_rejected': sum(example_labels[int(j)] == 'outside' for j in nearest_outside),
        'nearest_example_native_fact_correct': sum(example_labels[int(j)] == label for j, label in zip(nearest_native, native_labels)),
        'lexical_progress_with_fact_given_correct': sum(r['correct'] for r in lexical),
        'lexical_progress_total': len(lexical),
        'lexical_labels_previously_observed': sum(r['label_observed_previously'] for r in lexical),
        'question_details': targets, 'external_details': controls, 'lexical_details': lexical,
    }
np.savez_compressed(ROOT / 'results/factorized_feature_states.npz', X=X.numpy())
(ROOT / 'results/factorized_feature_diagnostic.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k != 'readouts'}, flush=True)
for name, result in report['readouts'].items():
    print(name, {k: v for k, v in result.items() if not k.endswith('_details')}, flush=True)
