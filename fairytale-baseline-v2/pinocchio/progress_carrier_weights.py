"""One explicit progress-aware carrier transform; no amplitude search."""
import hashlib
from pathlib import Path

import numpy as np
import torch

from semantic_carrier_weights import tensor_sha, restore_carrier

ROOT = Path(__file__).resolve().parent
AMPLITUDE_MULTIPLIER = 4.0


def apply_progress_carrier(model):
    assert not torch.is_grad_enabled()
    source = ROOT / 'results/progress_carrier_geometry.npz'
    with np.load(source) as z:
        U = torch.from_numpy(z['carrier'].copy()).double()
        B = torch.from_numpy(z['semantic_basis'].copy()).double()
        Pmu = torch.from_numpy(z['mean_projection'].copy()).double()
        alpha = AMPLITUDE_MULTIPLIER * float(z['alpha'])
    indices = torch.arange(6 * 64, 8 * 64)
    addition = (alpha * U @ B @ Pmu).float()
    originals, differences = {}, []

    def replace(name, after):
        assert name not in originals and torch.isfinite(after).all()
        value = model.get_parameter(name)
        before = value.clone()
        originals[name] = before
        value.copy_(after)
        differences.append({'tensor': name, 'shape': list(value.shape),
                            'changed_parameters': int(torch.count_nonzero(value - before)),
                            'difference_norm': float((value - before).norm()),
                            'before_sha256_f32': tensor_sha(before),
                            'after_sha256_f32': tensor_sha(value)})

    def input_projection(value, gain):
        gain = gain.double()
        assert float(gain.abs().min()) > 1e-6
        scaled = value.double() * gain[None, :]
        return ((scaled - (scaled @ U) @ U.T) / gain[None, :]).float()

    def output_projection(value):
        value = value.double()
        return (value - U @ (U.T @ value)).float()

    q = model.model.layers[21].self_attn.q_proj
    new_q, new_bias = q.weight.clone(), q.bias.clone()
    new_q[indices], new_bias[indices] = 0, 0
    replace('model.layers.21.self_attn.q_proj.weight', new_q)
    replace('model.layers.21.self_attn.q_proj.bias', new_bias)
    new_O = output_projection(model.model.layers[21].self_attn.o_proj.weight)
    new_O[:, indices] += addition
    replace('model.layers.21.self_attn.o_proj.weight', new_O)
    for index in [21, 22, 23]:
        layer = model.model.layers[index]
        if index > 21:
            for name in ['q', 'k', 'v']:
                replace(f'model.layers.{index}.self_attn.{name}_proj.weight',
                        input_projection(getattr(layer.self_attn, name + '_proj').weight,
                                         layer.input_layernorm.weight))
            replace(f'model.layers.{index}.self_attn.o_proj.weight',
                    output_projection(layer.self_attn.o_proj.weight))
        for name in ['gate', 'up']:
            replace(f'model.layers.{index}.mlp.{name}_proj.weight',
                    input_projection(getattr(layer.mlp, name + '_proj').weight,
                                     layer.post_attention_layernorm.weight))
        replace(f'model.layers.{index}.mlp.down_proj.weight', output_projection(layer.mlp.down_proj.weight))
    return originals, {
        'kind': 'progress_aware_protected_carrier',
        'source_geometry_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'amplitude_multiplier': AMPLITUDE_MULTIPLIER, 'alpha': alpha,
        'carrier_dimensions': U.shape[1], 'heads': [6, 7],
        'changed_parameters': sum(r['changed_parameters'] for r in differences),
        'changed_tensors': len(differences), 'tensor_differences': differences,
        'formula': 'Same protected-carrier projections as the recorded earlier transform, using the progress-aware twelve-dimensional basis and a single declared fourfold semantic amplitude. All other architecture, parameters and code paths remain native.',
        'training': False, 'gradients': False, 'optimizer': False,
    }
