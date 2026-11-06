"""Declared low-rank weight transforms for two semantic-carrier conditions."""
import hashlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def apply_carrier(model, mode):
    assert mode in ['unprotected', 'protected'] and not torch.is_grad_enabled()
    source = ROOT / 'results/semantic_carrier_geometry.npz'
    with np.load(source) as z:
        U = torch.from_numpy(z['carrier'].copy()).double()
        B = torch.from_numpy(z['semantic_basis'].copy()).double()
        Pmu = torch.from_numpy(z['mean_projection'].copy()).double()
        alpha = float(z['alpha'])
    indices = torch.tensor(list(range(6 * 64, 8 * 64)))
    addition = (alpha * U @ B @ Pmu).float()
    originals, differences = {}, []

    def replace(name, after):
        assert name not in originals
        value = model.get_parameter(name)
        before = value.clone()
        assert torch.isfinite(after).all() and after.shape == before.shape
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
    O = model.model.layers[21].self_attn.o_proj.weight
    new_O = output_projection(O) if mode == 'protected' else O.clone()
    new_O[:, indices] += addition
    replace('model.layers.21.self_attn.o_proj.weight', new_O)
    if mode == 'protected':
        for index in [21, 22, 23]:
            layer = model.model.layers[index]
            if index > 21:
                for name in ['q', 'k', 'v']:
                    projection = getattr(layer.self_attn, name + '_proj')
                    replace(f'model.layers.{index}.self_attn.{name}_proj.weight',
                            input_projection(projection.weight, layer.input_layernorm.weight))
                replace(f'model.layers.{index}.self_attn.o_proj.weight',
                        output_projection(layer.self_attn.o_proj.weight))
            for name in ['gate', 'up']:
                projection = getattr(layer.mlp, name + '_proj')
                replace(f'model.layers.{index}.mlp.{name}_proj.weight',
                        input_projection(projection.weight, layer.post_attention_layernorm.weight))
            replace(f'model.layers.{index}.mlp.down_proj.weight',
                    output_projection(layer.mlp.down_proj.weight))
    report = {
        'mode': mode, 'source_geometry_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'alpha': alpha, 'carrier_dimensions': U.shape[1], 'heads': [6, 7],
        'changed_parameters': sum(row['changed_parameters'] for row in differences),
        'changed_tensors': len(differences), 'tensor_differences': differences,
        'formula': 'Uniform block-21 heads 6/7; add alpha*U*B*Pmu to their original O columns. Protected condition additionally applies P=I-UUt to O21 and all later attention/MLP output projections, and W diag(g) P diag(g)^-1 to MLP21 and later attention/MLP input projections. RMS denominators remain native and may still change.',
        'training': False, 'gradients': False, 'optimizer': False,
    }
    return originals, report


def restore_carrier(model, originals):
    assert not torch.is_grad_enabled()
    for name, before in originals.items():
        value = model.get_parameter(name)
        value.copy_(before)
        assert torch.equal(value, before)
