"""Read exact existing histories under the declared final-attention algebra.

Original forwards only; compute the changed last-block states algebraically.
The preceding probe verified that algebra against actual edited weights. No
generated responses, training, corrective inference, or reserved questions.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from readout_contexts import load_contexts

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'centered_contexts'
assert not (ROOT / f'results/{STEM}.npz').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(6)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
all_K, all_R, all_records, _ = load_contexts(ROOT, t)
selected = [i for i, row in enumerate(all_records) if row['prefix'] == '']
records = [all_records[i] for i in selected]
assert len(records) == 420
groups = {}
for i, row in enumerate(records):
    groups.setdefault((row['question'], row['fact']), []).append(i)
sequences = []
for (question, fact), members in groups.items():
    longest = max((records[i]['history'] for i in members), key=len)
    assert all(records[i]['history'] == longest[:len(records[i]['history'])] for i in members)
    sequences.append({'id': question, 'kind': 'factual', 'fact': fact, 'ids': longest,
                      'members': members,
                      'positions': [len(records[i]['history']) - 1 for i in members]})
assert len(sequences) == 74
native = json.loads((ROOT / 'results/bounded_native_inputs.json').read_text('utf8'))
for row in native['sequences']:
    if row['kind'] == 'external':
        sequences.append({**row, 'kind': 'external_old',
                          'positions': list(range(row['start'], len(row['ids'])))})
extended = json.loads((ROOT / 'results/extended_native_inputs.json').read_text('utf8'))
for row in extended['sequences']:
    sequences.append({**row, 'kind': 'narrative' if row['kind'] == 'narrative' else 'preserved_existing',
                      'positions': list(range(row['start'], len(row['ids'])))})
review = json.loads((ROOT / 'results/native_second_continuations_review.json').read_text('utf8'))
correct_ids = {row['id'] for row in review['annotations'] if row['correct_with_supplied_unit']}
native_second = json.loads((ROOT / 'results/native_second_continuations_states.json').read_text('utf8'))
for row in native_second['sequences']:
    if row['id'] in correct_ids:
        sequences.append({**row, 'kind': 'native_second_correct',
                          'source': 'native_second_continuations_states.json',
                          'positions': list(range(row['start'], len(row['ids'])))})
control_count = sum(len(row['positions']) for row in sequences if row['kind'] != 'factual')
# The exact stored old external histories contain 641 positions, not the 525
# retained by an older matrix slice. Keep all 641, plus 711 newer external,
# 2642 extended native/narrative and 144 correct second-set continuation states.
assert control_count == 641 + 711 + 2642 + 144 == 4138, control_count
basis_path = ROOT / 'patches/centered_question_readout_basis.safetensors'
basis = load_file(basis_path)
indices = basis['indices']
heads = [int(indices[0] // 64), int(indices[64] // 64)]
m = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
block = m.model.layers[23]
assert torch.equal(block.self_attn.o_proj.weight[:, indices], basis['o_before'])
G, U, D = block.mlp.gate_proj.weight, block.mlp.up_proj.weight, block.mlp.down_proj.weight
g = block.post_attention_layernorm.weight
capture = {}


def keep(name, output=False):
    def hook(module, inputs, result=None):
        capture[name] = (result if output else inputs[0])[0].clone()
    return hook


hooks = [block.mlp.register_forward_pre_hook(keep('X')),
         block.mlp.down_proj.register_forward_pre_hook(keep('A')),
         m.model.norm.register_forward_pre_hook(keep('R')),
         block.self_attn.o_proj.register_forward_pre_hook(keep('heads')),
         block.self_attn.v_proj.register_forward_hook(keep('values', True))]
factual = {name: torch.empty(len(records), width) for name, width in
           [('X', 896), ('K', 4864), ('R', 896), ('original_R', 896), ('original_K', 4864)]}
control = {name: [] for name in ['X', 'K', 'R', 'original_R']}
offset = 0
for number, row in enumerate(sequences):
    m.model(input_ids=torch.tensor([row['ids']]), use_cache=False)
    positions = torch.tensor(row['positions'])
    original_A = capture['A'][positions]
    original_R = capture['R'][positions]
    H = original_R - original_A @ D.T
    values = capture['values'].cumsum(0) / torch.arange(1, len(row['ids']) + 1)[:, None]
    old_head = capture['heads'][positions].reshape(-1, 14, 64)[:, heads].flatten(1)
    changed_H = H - old_head @ basis['o_before'].T + values[positions] @ basis['o_after'].T
    changed_X = changed_H * torch.rsqrt(changed_H.square().mean(-1, keepdim=True) + 1e-6) * g
    changed_A = torch.nn.functional.silu(changed_X @ G.T) * (changed_X @ U.T)
    changed_R = changed_H + changed_A @ D.T
    values_to_store = {'X': changed_X, 'K': changed_A, 'R': changed_R, 'original_R': original_R}
    if row['kind'] == 'factual':
        for name, value in values_to_store.items():
            factual[name][row['members']] = value
        factual['original_K'][row['members']] = original_A
    else:
        row['control_offset'], row['control_states'] = offset, len(positions)
        offset += len(positions)
        for name, value in values_to_store.items():
            control[name].append(value.numpy())
    if number % 16 == 0:
        print('Existing histories', number + 1, '/', len(sequences), flush=True)
for hook in hooks:
    hook.remove()
key_error = float((factual['original_K'] - all_K[selected]).norm() / all_K[selected].norm())
residual_error = float((factual['original_R'] - all_R[selected]).norm() / all_R[selected].norm())
assert key_error < 1e-5 and residual_error < 1e-5
assert offset == control_count
np.savez_compressed(ROOT / f'results/{STEM}.npz',
                    **{f'factual_{key}': value.numpy() for key, value in factual.items()},
                    **{f'control_{key}': np.concatenate(value) for key, value in control.items()})
with basis_path.open('rb') as handle:
    basis_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
report = {
    'basis_sha256': basis_hash, 'factual_states': len(records), 'preservation_states': control_count,
    'original_factual_key_relative_error': key_error, 'original_factual_residual_relative_error': residual_error,
    'records': [{k: v for k, v in row.items() if k != 'history'} for row in records],
    'sequences': sequences, 'actual_weights_changed_during_collection': False,
    'states_after_attention_edit_calculated_algebraically': True,
    'original_narrative_preservation_is_not_canonical_correction': True,
    'training': False, 'gradients': False, 'optimizer': False,
    'free_generation_performed': False, 'third_set_read': False, 'final_set_read': False,
}
(ROOT / f'results/{STEM}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k not in ['records', 'sequences']}, flush=True)
