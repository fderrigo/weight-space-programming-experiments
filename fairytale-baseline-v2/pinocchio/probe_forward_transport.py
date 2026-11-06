"""Forward-routing algebra for transporting a proposed edit through layers.

This is NOT a Jacobian: attention probabilities, RMS denominators, and the
SwiGLU gate values are held fixed from an original forward pass. Their
derivatives are never computed. Compare its predicted intervention with actual
nonlinear forward passes. Hooks are diagnostic only, never a final solution.
"""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from readout_contexts import load_contexts

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'forward_transport'
output_path = ROOT / f'results/{STEM}.json'
assert not output_path.exists()
torch.set_grad_enabled(False)
torch.set_num_threads(6)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
_, cached_R, records, _ = load_contexts(ROOT, t)
bare_indices = [i for i, row in enumerate(records) if row['prefix'] == '']
bare = load_file(ROOT / 'patches/bare_context_basis.safetensors')
selection = {'creator': ' Ge', 'material': ' leg', 'nose': ' all', 'fairy': 'A',
             'deceivers': ' G', 'donkey': ' as', 'ending': ' bamb', 'puppeteer': ' Mang',
             'friend': ' Luc', 'wood_donor': ' Ma', 'rescuer': ' Al', 'fish_father': 'Ge'}
selected = []
for fact, piece in selection.items():
    hits = [(j, i) for j, i in enumerate(bare_indices)
            if records[i]['question'] == f'second_{fact}_0' and records[i]['piece'] == piece]
    assert len(hits) == 1
    selected.append(hits[0])
m = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True, torch_dtype=torch.float32, attn_implementation='eager').eval()
capture = {}
handles = []


def pre_capture(index, name):
    def collect(module, args):
        capture[(index, name)] = args[0][0, -1].clone()
    return collect


def gate_capture(index):
    def collect(module, args, output):
        capture[(index, 'gate')] = torch.nn.functional.silu(output[0, -1]).clone()
    return collect


def attention_capture(index):
    def collect(module, args, output):
        assert output[1] is not None
        capture[(index, 'self_probability')] = output[1][0, :, -1, -1].clone()
    return collect


for index in range(15, 24):
    layer = m.model.layers[index]
    handles += [layer.input_layernorm.register_forward_pre_hook(pre_capture(index, 'pre_attention')),
                layer.post_attention_layernorm.register_forward_pre_hook(pre_capture(index, 'pre_mlp')),
                layer.mlp.gate_proj.register_forward_hook(gate_capture(index)),
                layer.self_attn.register_forward_hook(attention_capture(index))]
handles += [m.model.norm.register_forward_pre_hook(pre_capture(24, 'raw_final')),
            m.model.layers[14].mlp.down_proj.register_forward_pre_hook(pre_capture(14, 'key'))]
E = m.get_output_embeddings().weight.detach() * m.model.norm.weight.detach()
identity = torch.eye(896, dtype=torch.float64)
results, arrays = [], {}


def decode_state(raw, row):
    logits = (raw / (raw.square().mean() + 1e-6).sqrt()) @ E.T
    history = sorted(set(row['history']))
    logits[history] = torch.where(logits[history] < 0, logits[history] * 1.1, logits[history] / 1.1)
    token = row['token']
    return {'target_rank': int((logits > logits[token]).sum()) + 1,
            'winner': t.decode([int(logits.argmax())]), 'target_logit': float(logits[token])}


def intervene(ids, delta):
    def add_delta(module, args, output):
        changed = output.clone()
        changed[:, -1] += delta
        return changed
    handle = m.model.layers[14].register_forward_hook(add_delta)
    try:
        m.model(input_ids=ids, use_cache=False)
        return capture[(24, 'raw_final')].clone()
    finally:
        handle.remove()


for case_index, (bare_index, record_index) in enumerate(selected):
    row = records[record_index]
    ids = torch.tensor([row['history']])
    m.model(input_ids=ids, use_cache=False)
    original_R = capture[(24, 'raw_final')].clone()
    original_key = capture[(14, 'key')].clone()
    desired = bare['values'][bare_index].double()
    T = identity.clone()
    layer_notes = []
    for index in range(15, 24):
        layer = m.model.layers[index]
        raw1 = capture[(index, 'pre_attention')].double()
        raw2 = capture[(index, 'pre_mlp')].double()
        rms1 = (raw1.square().mean() + 1e-6).sqrt()
        rms2 = (raw2.square().mean() + 1e-6).sqrt()
        norm1 = layer.input_layernorm.weight.double() / rms1
        norm2 = layer.post_attention_layernorm.weight.double() / rms2
        self_weight = capture[(index, 'self_probability')].double()
        # The current token changes only its own V contribution when routing
        # probabilities and all earlier-token states are held fixed.
        value_map = layer.self_attn.v_proj.weight.double().reshape(2, 64, 896)
        value_map = value_map.repeat_interleave(7, dim=0).reshape(896, 896)
        value_map = value_map * self_weight.repeat_interleave(64)[:, None]
        attention_map = identity + (layer.self_attn.o_proj.weight.double() @ value_map) * norm1[None]
        # This is an exact forward factorization at the original state with
        # the SwiGLU gate frozen, not a derivative of that gate.
        gate = capture[(index, 'gate')].double()
        mlp_map = identity + ((layer.mlp.down_proj.weight.double() * gate[None])
                              @ layer.mlp.up_proj.weight.double()) * norm2[None]
        T = mlp_map @ attention_map @ T
        layer_notes.append({'layer': index, 'native_attention_self_mean': float(self_weight.mean()),
                            'native_pre_attention_rms': float(rms1), 'native_pre_mlp_rms': float(rms2)})
    u, s, vh = torch.linalg.svd(T, full_matrices=False)
    keep = s > s[0] * .001
    transported = vh[keep].T @ ((u[:, keep].T @ desired) / s[keep])
    predicted = T @ transported
    direct_R = intervene(ids, desired.float())
    transported_R = intervene(ids, transported.float())
    measured = transported_R.double() - original_R.double()
    result = {
        'id': row['question'], 'piece': row['piece'], 'subtoken': row['subtoken'],
        'source_layer_after': 14, 'transport_layers': list(range(15, 24)),
        'native': decode_state(original_R, row),
        'last_layer_reference': decode_state(original_R + desired.float(), row),
        'direct_middle_injection': decode_state(direct_R, row),
        'transported_middle_injection': decode_state(transported_R, row),
        'desired_final_shift_norm': float(desired.norm()),
        'transported_middle_shift_norm': float(transported.norm()),
        'transport_singular_range': [float(s[-1]), float(s[0])],
        'transport_retained_rank': int(keep.sum()),
        'linear_equation_relative_error': float((predicted - desired).norm() / desired.norm()),
        'actual_final_shift_relative_error': float((measured - desired).norm() / desired.norm()),
        'actual_desired_cosine': float(torch.nn.functional.cosine_similarity(measured, desired, dim=0)),
        'native_eager_vs_saved_sdpa_relative_difference': float(
            (original_R - cached_R[record_index]).norm() / original_R.norm()),
        'layers': layer_notes,
        'weights_changed': False, 'temporary_activation_interventions': True,
        'counts_as_correction': False, 'third_set_read': False, 'final_set_read': False,
    }
    results.append(result)
    arrays[f'transport_{case_index}'] = T.float().numpy()
    arrays[f'middle_delta_{case_index}'] = transported.float().numpy()
    arrays[f'final_delta_{case_index}'] = desired.float().numpy()
    arrays[f'middle_native_key_{case_index}'] = original_key.numpy()
    arrays[f'raw_original_{case_index}'] = original_R.numpy()
    arrays[f'raw_transported_{case_index}'] = transported_R.numpy()
    print(row['question'], row['piece'], 'ranks native/direct/transport/reference',
          [result[key]['target_rank'] for key in ['native', 'direct_middle_injection',
                                                  'transported_middle_injection', 'last_layer_reference']],
          'shift norms', result['transported_middle_shift_norm'], result['desired_final_shift_norm'], flush=True)
    output_path.write_text(json.dumps({'method': 'Frozen native routing and SwiGLU gate forward factorization; not a Jacobian or gradient calculation.',
                                      'pinv_rtol': .001, 'training': False, 'gradients': False,
                                      'optimizer': False, 'cases': results}, ensure_ascii=False, indent=2), encoding='utf8')
for handle in handles:
    handle.remove()
np.savez_compressed(ROOT / f'results/{STEM}_states.npz', **arrays)
print('Completed twelve diagnostic transports; no network weights changed.', flush=True)
