"""Collect actual states after adding the constant attention reference."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from selector_reference_weights import apply_selector_reference, restore_carrier

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'selector_reference_states'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
source = RDIR / 'progress_carrier_states.json'
meta = json.loads(source.read_text('utf8'))
assert meta['completed']
records, sequences = meta['records'], meta['sequences']
with np.load(RDIR / 'selector_reference_geometry.npz') as z:
    v = torch.from_numpy(z['reference_direction'].copy()).double()
    amplitude = float(z['reference_amplitude'])
with np.load(RDIR / 'selector_class_geometry.npz') as z:
    V = torch.from_numpy(z['class_carrier'].copy()).double()
    class_alpha = float(z['alpha'])
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'reason': 'Verify that the actual value bias supplies a constant, leaves class selection unchanged, and survives as a protected reference in the final residual. Collect the actual native features for any subsequent fact-writing construction.',
    'actual_weights_changed': True, 'diagnostic_hooks_only': True,
    'source_sequence_and_record_metadata': str(source.relative_to(ROOT)),
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'transform_code_sha256': hashlib.sha256((ROOT / 'selector_reference_weights.py').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'geometry_sha256': hashlib.sha256((RDIR / 'selector_reference_geometry.npz').read_bytes()).hexdigest(),
    'relative_identity_tolerance': 1e-5, 'attention_max_error_tolerance': 1e-5,
    'template_unchanged': True, 'tokenizer_unchanged': True,
    'training': False, 'gradients': False, 'optimizer': False,
    'free_generation_performed': False, 'counts_as_correction': False,
    'third_set_used': False, 'final_set_read': False,
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
model_dir = ROOT.parent.parent / 'little-red-weightspace/models/original'
m = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.float32).eval()
originals, weights = apply_selector_reference(m)
(RDIR / f'{STEM}_weights.json').write_text(json.dumps(weights, indent=2), encoding='utf8')
parent_mlp = ROOT / 'patches/selector_reference_parent_last_mlp.safetensors'
assert not parent_mlp.exists()
save_file({name: m.get_parameter(f'model.layers.23.mlp.{name}_proj.weight').clone().contiguous()
           for name in ['gate', 'up', 'down']}, parent_mlp)
captured = {}


def capture_input(name):
    def hook(module, args):
        captured[name] = args[0][0].clone()
    return hook


def capture_head(module, args):
    captured['selected_values'] = args[0][0, :, 6 * 64:6 * 64 + 12].clone()
    captured['constant_value'] = args[0][0, :, 6 * 64 + 12:6 * 64 + 13].clone()


hooks = [m.model.layers[22].register_forward_pre_hook(capture_input('incoming22')),
         m.model.layers[22].self_attn.o_proj.register_forward_pre_hook(capture_head),
         m.model.layers[23].mlp.register_forward_pre_hook(capture_input('X')),
         m.model.layers[23].mlp.down_proj.register_forward_pre_hook(capture_input('A')),
         m.model.norm.register_forward_pre_hook(capture_input('R'))]
arrays = {name: np.empty((len(records), width), dtype=np.float32)
          for name, width in [('incoming22', 896), ('selected_values', 12), ('constant_value', 1), ('X', 896), ('A', 4864), ('R', 896)]}
try:
    for number, sequence in enumerate(sequences):
        m.model(input_ids=torch.tensor([sequence['ids']]), use_cache=False)
        positions = torch.tensor(sequence['positions'])
        begin, end = sequence['offset'], sequence['offset'] + sequence['states']
        for name in arrays:
            arrays[name][begin:end] = captured[name][positions].numpy()
        if number % 16 == 0:
            print('actual reference histories', number + 1, '/', len(sequences), flush=True)
finally:
    for hook in hooks:
        hook.remove()
    restore_carrier(m, originals)
with np.load(RDIR / 'selector_carrier_states.npz') as z:
    incoming_error = float(np.linalg.norm(arrays['incoming22'] - z['incoming22']) / np.linalg.norm(z['incoming22']))
    class_attention_max_error = float(np.abs(arrays['selected_values'] - z['selected_values']).max())
constant_error = float(np.abs(arrays['constant_value'] - 1).max())
incoming = torch.from_numpy(arrays['incoming22']).double()
R = torch.from_numpy(arrays['R']).double()
expected_reference = incoming @ v + amplitude * torch.from_numpy(arrays['constant_value'][:, 0]).double()
reference_error = float((R @ v - expected_reference).norm() / expected_reference.norm())
expected_class = incoming @ V + class_alpha * torch.from_numpy(arrays['selected_values']).double()
class_error = float((R @ V - expected_class).norm() / expected_class.norm())
passed = incoming_error < 1e-5 and class_attention_max_error < 1e-5 and constant_error < 1e-5 and reference_error < 1e-5 and class_error < 1e-5
np.savez_compressed(RDIR / f'{STEM}.npz', **arrays)
report = {'spec': spec, 'weights': weights, 'class_coordinates': weights['class_coordinates'],
          'states': len(records), 'sequences': len(sequences),
          'incoming_reference_relative_error': incoming_error,
          'class_attention_max_error_from_previous_actual': class_attention_max_error,
          'constant_value_max_error': constant_error,
          'reference_carrier_relative_identity_error': reference_error,
          'class_carrier_relative_identity_error': class_error,
          'all_declared_tolerances_passed': passed,
          'arrays_sha256': hashlib.sha256((RDIR / f'{STEM}.npz').read_bytes()).hexdigest(),
          'parent_last_mlp_sha256': hashlib.sha256(parent_mlp.read_bytes()).hexdigest(),
          'weights_restored_exactly': True, 'completed': True}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k not in ['spec', 'weights']}, flush=True)
assert passed, 'Actual reference did not satisfy the declared invariants; arrays retained for diagnosis.'
