"""Extend the frozen attention selector with one actual reference coordinate."""
import hashlib
from pathlib import Path

import numpy as np
import torch

from selector_carrier_weights import apply_selector_carrier
from semantic_carrier_weights import restore_carrier, tensor_sha

ROOT = Path(__file__).resolve().parent


def apply_selector_reference(model):
    assert not torch.is_grad_enabled()
    originals, parent_report = apply_selector_carrier(model)
    path = ROOT / 'results/selector_reference_geometry.npz'
    with np.load(path) as z:
        v = torch.from_numpy(z['reference_direction'].copy()).double()[:, None]
        amplitude = float(z['reference_amplitude'])

    def replace(name, after):
        value = model.get_parameter(name)
        assert name in originals
        assert value.shape == after.shape and torch.isfinite(after).all()
        value.copy_(after)

    def input_projection(value, gain):
        gain = gain.double()
        scaled = value.double() * gain[None, :]
        return ((scaled - (scaled @ v) @ v.T) / gain[None, :]).float()

    def output_projection(value):
        value = value.double()
        return (value - v @ (v.T @ value)).float()

    block = model.model.layers[22]
    W, bias = block.self_attn.v_proj.weight.clone(), block.self_attn.v_proj.bias.clone()
    W[12], bias[12] = 0, 1
    replace('model.layers.22.self_attn.v_proj.weight', W)
    replace('model.layers.22.self_attn.v_proj.bias', bias)
    O = output_projection(block.self_attn.o_proj.weight)
    for head in range(6):
        O[:, head * 64 + 12] = 0
    O[:, 6 * 64 + 12] = amplitude * v[:, 0].float()
    replace('model.layers.22.self_attn.o_proj.weight', O)
    for index in [22, 23]:
        block = model.model.layers[index]
        if index == 23:
            for name in ['q', 'k', 'v']:
                replace(f'model.layers.{index}.self_attn.{name}_proj.weight',
                        input_projection(getattr(block.self_attn, name + '_proj').weight, block.input_layernorm.weight))
            replace(f'model.layers.{index}.self_attn.o_proj.weight', output_projection(block.self_attn.o_proj.weight))
        for name in ['gate', 'up']:
            replace(f'model.layers.{index}.mlp.{name}_proj.weight',
                    input_projection(getattr(block.mlp, name + '_proj').weight, block.post_attention_layernorm.weight))
        replace(f'model.layers.{index}.mlp.down_proj.weight', output_projection(block.mlp.down_proj.weight))
    differences = []
    for name, before in originals.items():
        after = model.get_parameter(name)
        differences.append({'tensor': name, 'shape': list(after.shape),
                            'changed_parameters': int(torch.count_nonzero(after - before)),
                            'difference_norm': float((after - before).norm()),
                            'before_sha256_f32': tensor_sha(before), 'after_sha256_f32': tensor_sha(after)})
    report = {'kind': 'native_attention_selector_with_constant_reference',
              'class_coordinates': parent_report['class_coordinates'],
              'reference_amplitude': amplitude, 'reference_dimensions': 1,
              'reference_geometry_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'parent_selector_sha256': parent_report['selector_sha256'],
              'changed_tensors': len(differences), 'changed_parameters': sum(r['changed_parameters'] for r in differences),
              'tensor_differences': differences, 'training': False, 'gradients': False, 'optimizer': False}
    return originals, report
