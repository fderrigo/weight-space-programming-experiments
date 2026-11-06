"""Actual weight-only forward tests of unprotected and protected carriers."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from semantic_carrier_weights import apply_carrier, restore_carrier

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'semantic_carrier_forward'
assert not (ROOT / f'results/{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
old_rows = json.loads((ROOT / 'results/question_layer_diagnostic_states.json').read_text('utf8'))['sequences']
new_rows = json.loads((ROOT / 'results/third_question_features_states.json').read_text('utf8'))['rows']
rows = old_rows + new_rows
assert len(rows) == 152
with np.load(ROOT / 'results/semantic_carrier_geometry.npz') as z:
    U = torch.from_numpy(z['carrier'].copy()).double()
    B = torch.from_numpy(z['semantic_basis'].copy()).double()
    Pmu = torch.from_numpy(z['mean_projection'].copy()).double()
    centers = torch.from_numpy(z['semantic_centers'].copy()).double()
    alpha = float(z['alpha'])
with np.load(ROOT / 'results/question_layer_diagnostic_states.npz') as z:
    original_R = torch.from_numpy(z['H'][:, 0, 24].copy())
    original_incoming = torch.from_numpy(z['H'][:, 0, 21].copy())
with np.load(ROOT / 'results/attention_feature_diagnostic_states.npz') as z:
    original_means = torch.from_numpy(z['prefix_values'][:, 21].copy()).flatten(1)
with np.load(ROOT / 'results/third_question_features_states.npz') as z:
    original_R = torch.cat([original_R, torch.from_numpy(z['final_residual'].copy())])
    original_means = torch.cat([original_means, torch.from_numpy(z['prefix_values_21'].copy())])
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'conditions': ['unprotected', 'protected'], 'condition_parameters_fixed_before_forwards': True,
    'geometry_sha256': hashlib.sha256((ROOT / 'results/semantic_carrier_geometry.npz').read_bytes()).hexdigest(),
    'weight_transform_code_sha256': hashlib.sha256((ROOT / 'semantic_carrier_weights.py').read_bytes()).hexdigest(),
    'reason': 'Test whether the declared twelve-dimensional signal survives native layers, and whether explicit projections preserve it with less global disturbance than the earlier amplified attention map.',
    'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
    'diagnostic_only': True, 'free_generation_performed': False,
    'third_set_already_observed': True, 'final_set_read': False,
    'no_full_checkpoint_export': True,
    'readout': 'Cosine against the same thirteen original prototype centers, without refitting; descriptive question-type classification only.',
}
(ROOT / f'results/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
m = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
active, width, captured = None, None, {}


def input_hook(name):
    def capture(module, args):
        captured[name] = args[0][:, -1].clone()
    return capture


def output_hook(name):
    def capture(module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        captured[name] = value[:, -1].clone()
    return capture


def value_hook(module, args, output):
    captured['means'] = torch.stack([output[j, width - len(row['ids']):].mean(0)
                                     for j, row in enumerate(active)])


hooks = [m.model.layers[21].register_forward_pre_hook(input_hook('incoming')),
         m.model.layers[21].self_attn.v_proj.register_forward_hook(value_hook),
         m.model.layers[23].mlp.register_forward_pre_hook(input_hook('X23')),
         m.model.layers[23].mlp.down_proj.register_forward_pre_hook(input_hook('A23')),
         m.model.norm.register_forward_pre_hook(input_hook('R'))]
hooks += [m.model.layers[i].register_forward_hook(output_hook(f'block_{i}')) for i in [21, 22, 23]]
labels = [row['label'] for row in rows]
classes = sorted(set(labels))
earlier = [i for i, row in enumerate(old_rows) if row['split'] == 'probe']
third = list(range(len(old_rows), len(rows)))
reports = []


def normalize(value):
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)


for mode in spec['conditions']:
    originals, weights = apply_carrier(m, mode)
    (ROOT / f'results/{STEM}_{mode}_weights.json').write_text(json.dumps(weights, indent=2), encoding='utf8')
    print('Applied', mode, weights['changed_parameters'], 'parameters in', weights['changed_tensors'], 'tensors', flush=True)
    stored = {name: [] for name in ['incoming', 'means', 'X23', 'A23', 'R', 'block_21', 'block_22', 'block_23']}
    for begin in range(0, len(rows), 4):
        active = rows[begin:begin + 4]
        width = max(len(row['ids']) for row in active)
        ids = torch.tensor([[t.pad_token_id] * (width - len(row['ids'])) + row['ids'] for row in active])
        mask = torch.tensor([[0] * (width - len(row['ids'])) + [1] * len(row['ids']) for row in active])
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        m.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False)
        for name in stored:
            stored[name].append(captured[name].numpy())
        if begin % 24 == 0:
            print(mode, 'forward', begin + len(active), '/', len(rows), flush=True)
    values = {name: torch.from_numpy(np.concatenate(parts)) for name, parts in stored.items()}
    np.savez_compressed(ROOT / f'results/{STEM}_{mode}_states.npz',
                        **{name: value.numpy() for name, value in values.items()})
    means_error = float((values['means'] - original_means).norm() / original_means.norm())
    incoming_error = float((values['incoming'][:108] - original_incoming).norm() / original_incoming.norm())
    assert max(means_error, incoming_error) < 1e-4
    signal = alpha * values['means'].double() @ Pmu @ B.T
    expected = values['incoming'].double() @ U + signal
    actual = values['R'].double() @ U
    errors = (actual - expected).norm(dim=-1) / signal.norm(dim=-1).clamp_min(1e-12)
    if mode == 'protected':
        assert float(errors.max()) < 1e-4, float(errors.max())
    comparisons = []
    for name, coordinates in [('pure_signal', signal / alpha),
                              ('expected_with_inherited', expected / alpha),
                              ('actual_final', actual / alpha)]:
        predictions = [classes[i] for i in (normalize(coordinates) @ normalize(centers).T).argmax(-1).tolist()]
        for split, selected in [('earlier_development', earlier), ('third_already_observed', third)]:
            comparisons.append({
                'representation': name, 'split': split,
                'target_count': sum(labels[i] != 'outside' for i in selected),
                'target_correct': sum(predictions[i] == labels[i] for i in selected if labels[i] != 'outside'),
                'outside_count': sum(labels[i] == 'outside' for i in selected),
                'outside_correct': sum(predictions[i] == 'outside' for i in selected if labels[i] == 'outside'),
                'predictions': [{'id': rows[i]['id'], 'expected': labels[i], 'predicted': predictions[i]}
                                for i in selected],
            })
    block_errors = {}
    for index in [21, 22, 23]:
        distance = (values[f'block_{index}'].double() @ U - expected).norm(dim=-1)
        relative = distance / signal.norm(dim=-1).clamp_min(1e-12)
        block_errors[str(index)] = {'mean_absolute': float(distance.mean()),
                                    'mean_over_signal': float(relative.mean()),
                                    'max_over_signal': float(relative.max())}
    shift = (values['R'] - original_R).norm(dim=-1)
    relative_shift = shift / original_R.norm(dim=-1).clamp_min(1e-12)
    shifts = []
    for split, selected in [('earlier_development', earlier), ('third_already_observed', third)]:
        for group in ['target', 'outside']:
            indices = [i for i in selected if (labels[i] == 'outside') == (group == 'outside')]
            shifts.append({'split': split, 'group': group, 'count': len(indices),
                           'mean_residual_change': float(shift[indices].mean()),
                           'max_residual_change': float(shift[indices].max()),
                           'mean_relative_residual_change': float(relative_shift[indices].mean())})
    restore_carrier(m, originals)
    del originals
    report = {'mode': mode, 'weights': weights,
              'native_prefix_values_relative_error': means_error,
              'native_incoming_residual_relative_error': incoming_error,
              'carrier_drift_by_block': block_errors, 'residual_changes': shifts,
              'comparisons': comparisons, 'weights_restored_in_memory': True,
              'free_generation_performed': False, 'counts_as_factual_correction': False}
    reports.append(report)
    (ROOT / f'results/{STEM}.json').write_text(json.dumps(
        {'spec': spec, 'conditions': reports, 'completed': len(reports) == len(spec['conditions'])},
        ensure_ascii=False, indent=2), encoding='utf8')
    print(mode, 'carrier drift', block_errors, 'residual shifts', shifts, flush=True)
    for comparison in comparisons:
        print({key: value for key, value in comparison.items() if key != 'predictions'}, flush=True)
for hook in hooks:
    hook.remove()
(ROOT / f'results/{STEM}_states.json').write_text(json.dumps(
    {'rows': rows, 'conditions': spec['conditions'], 'final_set_read': False},
    ensure_ascii=False, indent=2), encoding='utf8')
print('Both declared carrier conditions completed; no factual-correction claim.', flush=True)
