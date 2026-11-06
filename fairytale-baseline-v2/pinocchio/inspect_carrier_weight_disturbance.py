"""Separate large coefficient changes from their actual forward effects.

Read saved states and original weights only. No model edit, optimization, or
new question is performed here.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors'
OUT = RESULTS / 'semantic_carrier_weight_disturbance.json'
assert not OUT.exists()
torch.set_grad_enabled(False)
torch.set_num_threads(2)
meta = json.loads((RESULTS / 'semantic_carrier_forward_states.json').read_text('utf8'))
rows = meta['rows']
with np.load(RESULTS / 'semantic_carrier_geometry.npz') as z:
    U = torch.from_numpy(z['carrier'].copy()).double()
with np.load(RESULTS / 'question_layer_diagnostic_states.npz') as z:
    old_X = torch.from_numpy(z['X'][:, 0, 23].copy())
with np.load(RESULTS / 'third_question_features_states.npz') as z:
    old_X = torch.cat([old_X, torch.from_numpy(z['last_mlp_input'].copy())])
with safe_open(MODEL, framework='pt') as f:
    gain = f.get_tensor('model.layers.23.post_attention_layernorm.weight').float()
    gate = f.get_tensor('model.layers.23.mlp.gate_proj.weight').float()
    up = f.get_tensor('model.layers.23.mlp.up_proj.weight').float()
    down = f.get_tensor('model.layers.23.mlp.down_proj.weight').float()


def projected_input(weight):
    scaled = weight.double() * gain.double()[None, :]
    return ((scaled - (scaled @ U) @ U.T) / gain.double()[None, :]).float()


new_gate, new_up = projected_input(gate), projected_input(up)
new_down = (down.double() - U @ (U.T @ down.double())).float()
native_gate, native_up = old_X @ gate.T, old_X @ up.T
native_A = torch.nn.functional.silu(native_gate) * native_up
native_output = native_A @ down.T
weight_notes = []
for name, original, changed in [('gate', gate, new_gate), ('up', up, new_up), ('down', down, new_down)]:
    delta = changed - original
    note = {'projection': name, 'original_norm': float(original.norm()),
            'difference_norm': float(delta.norm()),
            'relative_weight_difference': float(delta.norm() / original.norm())}
    if name != 'down':
        energy = delta.double().square().sum(0)
        indices = energy.topk(8).indices.tolist()
        note['top_input_columns'] = [{'index': i, 'norm_gain': float(gain[i]),
                                     'fraction_of_squared_difference': float(energy[i] / energy.sum())}
                                    for i in indices]
        note['rms_gain_weighted_relative_difference'] = float(
            (delta * gain[None, :]).norm() / (original * gain[None, :]).norm())
    weight_notes.append(note)
reports = []
for mode in ['unprotected', 'protected']:
    with np.load(RESULTS / f'semantic_carrier_forward_{mode}_states.npz') as z:
        X = torch.from_numpy(z['X23'].copy())
        captured_A = torch.from_numpy(z['A23'].copy())
    G, V, D = (new_gate, new_up, new_down) if mode == 'protected' else (gate, up, down)
    gate_values, up_values = X @ G.T, X @ V.T
    A = torch.nn.functional.silu(gate_values) * up_values
    recapture_error = float((A - captured_A).norm() / captured_A.norm())
    assert recapture_error < 1e-4, recapture_error
    changes = {
        'gate_preactivation': ((gate_values - native_gate).norm(dim=-1), native_gate.norm(dim=-1)),
        'up_preactivation': ((up_values - native_up).norm(dim=-1), native_up.norm(dim=-1)),
        'mlp_activation': ((A - native_A).norm(dim=-1), native_A.norm(dim=-1)),
        'mlp_output': ((A @ D.T - native_output).norm(dim=-1), native_output.norm(dim=-1)),
    }
    groups = []
    for split, indices in [('earlier', [i for i, r in enumerate(rows[:108]) if r['split'] == 'probe']),
                           ('third_observed', list(range(108, 152)))]:
        for group in ['target', 'outside']:
            selected = [i for i in indices if (rows[i]['label'] == 'outside') == (group == 'outside')]
            groups.append({'split': split, 'group': group,
                           'mean_relative_changes': {name: float((delta[selected] / norm[selected].clamp_min(1e-12)).mean())
                                                     for name, (delta, norm) in changes.items()},
                           'gate_sign_change_fraction': float(((gate_values[selected] > 0) != (native_gate[selected] > 0)).float().mean())})
    reports.append({'mode': mode, 'activation_recapture_relative_error': recapture_error, 'groups': groups})
result = {'source_forward_sha256': hashlib.sha256((RESULTS / 'semantic_carrier_forward.json').read_bytes()).hexdigest(),
          'weights': weight_notes, 'forward_effects': reports,
          'read_only_diagnostic': True, 'weights_changed': False,
          'training': False, 'gradients': False, 'optimizer': False, 'final_set_read': False,
          'interpretation': 'Coefficient size is not the same as functional disturbance. Very small RMS gains can make the conjugated input projection large; measured activations and generated text are needed.'}
OUT.write_text(json.dumps(result, indent=2), encoding='utf8')
print(json.dumps(result, indent=2), flush=True)
