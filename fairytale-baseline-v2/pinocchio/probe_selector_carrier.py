"""Actual weight-only forward validation of native attention class selection."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from selector_carrier_weights import apply_selector_carrier, restore_carrier

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'selector_carrier_states'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
source = RDIR / 'progress_carrier_states.json'
meta = json.loads(source.read_text('utf8'))
assert meta['completed']
records, sequences = meta['records'], meta['sequences']
selector_meta = json.loads((RDIR / 'template_attention_selector.json').read_text('utf8'))
classes = list(selector_meta['slots_by_class'])
facts = [label for label in classes if label != 'outside']
with np.load(RDIR / 'template_attention_selector.npz') as z:
    expected_values = z['actual_values'][:, [classes.index(c) for c in facts]].copy()
with np.load(RDIR / 'selector_class_geometry.npz') as z:
    V = torch.from_numpy(z['class_carrier'].copy()).double()
    alpha = float(z['alpha'])
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'reason': 'Verify that the specified Q/K/V/O matrices select the same relation values in the actual model and that those values survive to the last MLP. Only then can a fact-writing candidate use them.',
    'actual_weights_changed': True, 'diagnostic_hooks_only': True,
    'source_sequence_and_record_metadata': str(source.relative_to(ROOT)),
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'transform_code_sha256': hashlib.sha256((ROOT / 'selector_carrier_weights.py').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'class_geometry_sha256': hashlib.sha256((RDIR / 'selector_class_geometry.npz').read_bytes()).hexdigest(),
    'selector_sha256': hashlib.sha256((RDIR / 'template_attention_selector.npz').read_bytes()).hexdigest(),
    'repeat_reason': 'The previous full collection restored the model but failed the declared stored-simulation tolerance before saving arrays. Keep this tolerance unchanged, retain actual arrays, and separately reconstruct attention using actual Q/K/V and native rotary outputs.',
    'template_unchanged': True, 'tokenizer_unchanged': True,
    'training': False, 'gradients': False, 'optimizer': False,
    'free_generation_performed': False, 'counts_as_correction': False,
    'third_set_used': False, 'final_set_read': False,
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
model_dir = ROOT.parent.parent / 'little-red-weightspace/models/original'
m = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.float32).eval()
originals, weights = apply_selector_carrier(m)
(RDIR / f'{STEM}_weights.json').write_text(json.dumps(weights, indent=2), encoding='utf8')
captured = {}


def capture_input(name):
    def hook(module, args):
        captured[name] = args[0][0].clone()
    return hook


def capture_head(module, args):
    captured['selected_values'] = args[0][0, :, 6 * 64:6 * 64 + 12].clone()


def capture_projected(name, start, width):
    def hook(module, args, output):
        captured[name] = output[0, :, start:start + width].clone()
    return hook


def capture_rotary(module, args, output):
    captured['cos'], captured['sin'] = output[0][0].clone(), output[1][0].clone()


hooks = [m.model.layers[22].register_forward_pre_hook(capture_input('incoming22')),
         m.model.layers[22].self_attn.o_proj.register_forward_pre_hook(capture_head),
         m.model.layers[23].mlp.register_forward_pre_hook(capture_input('X')),
         m.model.layers[23].mlp.down_proj.register_forward_pre_hook(capture_input('A')),
         m.model.norm.register_forward_pre_hook(capture_input('R'))]
hooks += [m.model.layers[22].self_attn.q_proj.register_forward_hook(capture_projected('actual_q', 6 * 64, 64)),
          m.model.layers[22].self_attn.k_proj.register_forward_hook(capture_projected('actual_k', 0, 64)),
          m.model.layers[22].self_attn.v_proj.register_forward_hook(capture_projected('actual_v', 0, 64)),
          m.model.rotary_emb.register_forward_hook(capture_rotary)]
arrays = {name: np.empty((len(records), width), dtype=np.float32)
          for name, width in [('incoming22', 896), ('selected_values', 12), ('X', 896), ('A', 4864), ('R', 896)]}
native_reference = np.empty((len(records), 12), dtype=np.float32)
for number, sequence in enumerate(sequences):
    m.model(input_ids=torch.tensor([sequence['ids']]), use_cache=False)
    positions = torch.tensor(sequence['positions'])
    begin, end = sequence['offset'], sequence['offset'] + sequence['states']
    for name in arrays:
        arrays[name][begin:end] = captured[name][positions].numpy()
    rotated = {}
    for name in ['actual_q', 'actual_k']:
        value = captured[name]
        half = torch.cat([-value[:, 32:], value[:, :32]], dim=-1)
        rotated[name] = value * captured['cos'] + half * captured['sin']
    reference = torch.nn.functional.scaled_dot_product_attention(
        rotated['actual_q'][None, None], rotated['actual_k'][None, None],
        captured['actual_v'][None, None], is_causal=True, dropout_p=0.0)[0, 0, :, :12]
    native_reference[begin:end] = reference[positions].numpy()
    if number % 16 == 0:
        print('actual selector histories', number + 1, '/', len(sequences), flush=True)
for hook in hooks:
    hook.remove()
restore_carrier(m, originals)
with np.load(RDIR / 'progress_attention_states.npz') as z:
    incoming_error = float(np.linalg.norm(arrays['incoming22'] - z['incoming22']) / np.linalg.norm(z['incoming22']))
assert incoming_error < 1e-5
value_error = arrays['selected_values'] - expected_values
max_value_error = float(np.abs(value_error).max())
relative_value_error = float(np.linalg.norm(value_error) / np.linalg.norm(expected_values))
simulation_passed = max_value_error < .001 and relative_value_error < 1e-4
native_reference_error = arrays['selected_values'] - native_reference
native_reference_max = float(np.abs(native_reference_error).max())
native_reference_relative = float(np.linalg.norm(native_reference_error) / np.linalg.norm(native_reference))
expected_carrier = torch.from_numpy(arrays['incoming22']).double() @ V + alpha * torch.from_numpy(arrays['selected_values']).double()
actual_carrier = torch.from_numpy(arrays['R']).double() @ V
carrier_error = float((actual_carrier - expected_carrier).norm() / expected_carrier.norm())
assert carrier_error < 1e-5
np.savez_compressed(RDIR / f'{STEM}.npz', **arrays, native_attention_reference=native_reference)
report = {'spec': spec, 'weights': weights, 'class_coordinates': facts,
          'states': len(records), 'sequences': len(sequences),
          'incoming_reference_relative_error': incoming_error,
          'selected_value_max_absolute_error': max_value_error,
          'selected_value_relative_error': relative_value_error,
          'stored_simulation_tolerance_passed': simulation_passed,
          'native_attention_reference_max_error': native_reference_max,
          'native_attention_reference_relative_error': native_reference_relative,
          'native_attention_reference_tolerance_passed': native_reference_max < 1e-5 and native_reference_relative < 1e-5,
          'class_carrier_relative_identity_error': carrier_error,
          'arrays_sha256': hashlib.sha256((RDIR / f'{STEM}.npz').read_bytes()).hexdigest(),
          'weights_restored_exactly': True, 'completed': True}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k not in ['spec', 'weights']}, flush=True)
