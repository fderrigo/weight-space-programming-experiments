"""Apply the frozen last-block class selector, reference and RMS compensation."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from progress_carrier_weights import apply_progress_carrier
from semantic_carrier_weights import restore_carrier, tensor_sha

ROOT = Path(__file__).resolve().parent


def apply_late_selector(model):
    assert not torch.is_grad_enabled()
    originals, parent = apply_progress_carrier(model)
    geometry = json.loads((ROOT / 'results/late_selector_geometry.json').read_text('utf8'))
    paths = {kind: ROOT / f'patches/late_selector_{kind}.safetensors' for kind in ['attention', 'mlp']}
    for kind, path in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == geometry[kind + '_sha256']
    expected_parent = load_file(ROOT / 'patches/late_selector_parent_attention.safetensors')
    for name, before in expected_parent.items():
        assert torch.equal(model.get_parameter('model.layers.23.self_attn.' + name), before)

    def replace(name, after):
        value = model.get_parameter(name)
        if name not in originals:
            originals[name] = value.clone()
        assert value.shape == after.shape and torch.isfinite(after).all()
        value.copy_(after)

    for name, value in load_file(paths['attention']).items():
        replace('model.layers.23.self_attn.' + name, value)
    for name, value in load_file(paths['mlp']).items():
        replace(f'model.layers.23.mlp.{name}_proj.weight', value)
    differences = []
    for name, before in originals.items():
        after = model.get_parameter(name)
        differences.append({'tensor': name, 'shape': list(after.shape),
                            'changed_parameters': int(torch.count_nonzero(after - before)),
                            'difference_norm': float((after - before).norm()),
                            'before_sha256_f32': tensor_sha(before), 'after_sha256_f32': tensor_sha(after)})
    return originals, {'kind': 'last_block_class_selector_with_reference',
                       'class_coordinates': geometry['class_coordinates'],
                       'class_alpha': geometry['class_alpha'],
                       'reference_amplitude': geometry['reference_amplitude'],
                       'native_RMS_compensation': geometry['native_RMS_compensation'],
                       'query_head': geometry['query_head'], 'key_value_head': geometry['key_value_head'],
                       'geometry_sha256': geometry['geometry_sha256'],
                       'attention_sha256': geometry['attention_sha256'], 'mlp_sha256': geometry['mlp_sha256'],
                       'changed_tensors': len(differences), 'changed_parameters': sum(r['changed_parameters'] for r in differences),
                       'tensor_differences': differences, 'training': False, 'gradients': False, 'optimizer': False}
