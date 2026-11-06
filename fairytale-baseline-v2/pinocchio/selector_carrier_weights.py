"""Real native Q/K/V/O edits for relation selection and protected class values."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from progress_carrier_weights import apply_progress_carrier
from semantic_carrier_weights import tensor_sha, restore_carrier

ROOT = Path(__file__).resolve().parent


def apply_selector_carrier(model):
    assert not torch.is_grad_enabled()
    originals, _ = apply_progress_carrier(model)
    selector_path = ROOT / 'results/template_attention_selector.npz'
    geometry_path = ROOT / 'results/selector_class_geometry.npz'
    with np.load(selector_path) as z:
        selector = {k: torch.from_numpy(z[k].copy()) for k in ['q_weight', 'q_bias', 'k_weight', 'k_bias', 'value_rows']}
    with np.load(geometry_path) as z:
        V = torch.from_numpy(z['class_carrier'].copy()).double()
        alpha = float(z['alpha'])
    labels = list(json.loads((ROOT / 'results/template_attention_selector.json').read_text('utf8'))['slots_by_class'])
    facts = [label for label in labels if label != 'outside']

    def replace(name, after):
        value = model.get_parameter(name)
        if name not in originals:
            originals[name] = value.clone()
        assert value.shape == after.shape and torch.isfinite(after).all()
        value.copy_(after)

    def input_projection(value, gain):
        gain = gain.double()
        assert float(gain.abs().min()) > 1e-6
        scaled = value.double() * gain[None, :]
        return ((scaled - (scaled @ V) @ V.T) / gain[None, :]).float()

    def output_projection(value):
        value = value.double()
        return (value - V @ (V.T @ value)).float()

    layer = model.model.layers[22]
    q, qb = layer.self_attn.q_proj.weight.clone(), layer.self_attn.q_proj.bias.clone()
    axes = torch.arange(50, 64)
    for head in range(6):
        q[head * 64 + axes] = 0
        qb[head * 64 + axes] = 0
    q[6 * 64:7 * 64] = selector['q_weight']
    qb[6 * 64:7 * 64] = selector['q_bias']
    replace('model.layers.22.self_attn.q_proj.weight', q)
    replace('model.layers.22.self_attn.q_proj.bias', qb)
    k, kb = layer.self_attn.k_proj.weight.clone(), layer.self_attn.k_proj.bias.clone()
    k[:64], kb[:64] = selector['k_weight'], selector['k_bias']
    replace('model.layers.22.self_attn.k_proj.weight', k)
    replace('model.layers.22.self_attn.k_proj.bias', kb)
    v, vb = layer.self_attn.v_proj.weight.clone(), layer.self_attn.v_proj.bias.clone()
    v[:12] = selector['value_rows'][[labels.index(c) for c in facts]]
    vb[:12] = 0
    replace('model.layers.22.self_attn.v_proj.weight', v)
    replace('model.layers.22.self_attn.v_proj.bias', vb)
    O = output_projection(layer.self_attn.o_proj.weight)
    for head in range(6):
        O[:, head * 64:head * 64 + 12] = 0
    O[:, 6 * 64:7 * 64] = 0
    O[:, 6 * 64:6 * 64 + 12] = (alpha * V).float()
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
    report = {'kind': 'native_attention_relation_selector', 'class_coordinates': facts,
              'class_alpha': alpha, 'semantic_dimensions': 12, 'class_dimensions': 12,
              'selector_sha256': hashlib.sha256(selector_path.read_bytes()).hexdigest(),
              'class_geometry_sha256': hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
              'changed_tensors': len(differences), 'changed_parameters': sum(r['changed_parameters'] for r in differences),
              'tensor_differences': differences, 'training': False, 'gradients': False, 'optimizer': False,
              'formula': 'Retain the previous semantic carrier. Replace block22 head6 queries and KV-head0 memory coordinates; suppress those coordinates in the other shared heads. Write twelve factual attention values into a new orthogonal quiet subspace, with the outside value zero. Protect that subspace through MLP22 and block23 using gain-compensated input and output projections.'}
    return originals, report
