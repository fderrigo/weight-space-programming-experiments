"""Late constant reference with the declared native RMS compensation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from progress_carrier_weights import apply_progress_carrier
from semantic_carrier_weights import restore_carrier, tensor_sha

ROOT = Path(__file__).resolve().parent
CONDITION = 'reference_with_fixed_native_RMS_compensation'


def apply_late_reference(model):
    assert not torch.is_grad_enabled()
    originals, parent = apply_progress_carrier(model)
    geometry_path = ROOT / 'results/late_reference_geometry.npz'
    effects = json.loads((ROOT / 'results/late_reference_effects.json').read_text('utf8'))
    assert hashlib.sha256(geometry_path.read_bytes()).hexdigest() == effects['spec']['geometry_sha256']
    with np.load(geometry_path) as z:
        coordinate = int(z['value_coordinate'])
        O = torch.from_numpy(z['o_after'].copy())
        amplitude = float(z['reference_amplitude'])
        compensation = float(z['native_RMS_compensation'])
    protected_path = ROOT / 'patches/late_reference_protected_mlp.safetensors'
    assert hashlib.sha256(protected_path.read_bytes()).hexdigest() == effects['spec']['protected_mlp_sha256']
    protected = load_file(protected_path)
    attention = load_file(ROOT / 'patches/late_reference_parent_attention.safetensors')
    block = model.model.layers[23]
    assert torch.equal(block.self_attn.o_proj.weight, attention['o_weight'])
    assert torch.equal(block.self_attn.v_proj.weight, attention['v_weight'])
    assert torch.equal(block.self_attn.v_proj.bias, attention['v_bias'])

    def replace(name, after):
        value = model.get_parameter(name)
        if name not in originals:
            originals[name] = value.clone()
        assert value.shape == after.shape and torch.isfinite(after).all()
        value.copy_(after)

    W, bias = block.self_attn.v_proj.weight.clone(), block.self_attn.v_proj.bias.clone()
    W[coordinate], bias[coordinate] = 0, 1
    replace('model.layers.23.self_attn.v_proj.weight', W)
    replace('model.layers.23.self_attn.v_proj.bias', bias)
    replace('model.layers.23.self_attn.o_proj.weight', O)
    expected = effects['reconstruction_checks'][CONDITION]['last_mlp_sha256']
    for name in ['gate', 'up', 'down']:
        value = protected[name] * compensation if name != 'down' else protected[name]
        assert tensor_sha(value) == expected[name]
        replace(f'model.layers.23.mlp.{name}_proj.weight', value)
    differences = []
    for name, before in originals.items():
        after = model.get_parameter(name)
        differences.append({'tensor': name, 'shape': list(after.shape),
                            'changed_parameters': int(torch.count_nonzero(after - before)),
                            'difference_norm': float((after - before).norm()),
                            'before_sha256_f32': tensor_sha(before), 'after_sha256_f32': tensor_sha(after)})
    return originals, {'kind': 'late_constant_reference_on_progress_carrier', 'condition': CONDITION,
                       'parent_geometry_sha256': parent['source_geometry_sha256'],
                       'geometry_sha256': hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
                       'value_coordinate': coordinate, 'reference_amplitude': amplitude,
                       'native_RMS_compensation': compensation, 'protected_dimensions': 13,
                       'selection_reason': 'The predeclared median RMS factor reduced reconstructed external residual disturbance from20.21 to9.14 and preserved125 rather than92 of153 native next tokens. No gain search was performed.',
                       'changed_tensors': len(differences), 'changed_parameters': sum(r['changed_parameters'] for r in differences),
                       'tensor_differences': differences, 'training': False, 'gradients': False, 'optimizer': False}
