"""Collect actual full-model states after the frozen late selector edit."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from late_selector_weights import apply_late_selector, restore_carrier

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'late_selector_states'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
source = RDIR / 'progress_carrier_states.json'
meta = json.loads(source.read_text('utf8'))
sequences = meta['sequences']
selected = list(range(len(sequences)))
indices = [j for i in selected for j in range(sequences[i]['offset'], sequences[i]['offset'] + sequences[i]['states'])]
with np.load(RDIR / 'late_selector_reconstruction.npz') as z:
    expected = {name: z[name][indices].copy() for name in ['incoming23', 'X', 'A', 'R']}
with np.load(RDIR / 'late_template_selector.npz') as z:
    output_indices = int(z['query_head']) * 64 + z['value_coordinates'][:12]
    output_index = int(z['query_head']) * 64 + int(z['value_coordinates'][12])
with np.load(RDIR / 'late_selector_geometry.npz') as z:
    V = z['class_carrier'].copy()
    reference = z['reference_direction'].copy()
    alpha = float(z['class_alpha'])
    reference_amplitude = float(z['reference_amplitude'])
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'reason': 'Collect actual changed-weight full-model states for factual gates and output equations. Check carrier identities, and report every discrepancy from canonical-prefix simulation.',
    'sample_rule': 'All197 already observed histories; no new or final questions.',
    'sequence_indices': selected, 'record_indices': indices,
    'scope': 'All4707 selected positions from full-model forwards with actual weights. No output is substituted and these prefixed histories are not free generation.',
    'physical_identity_relative_tolerance': 1e-5, 'constant_max_absolute_tolerance': 1e-5,
    'reconstruction_comparison': 'X/A/R simulation discrepancies are measured, not a pass criterion. Actual states replace simulations for subsequent fact construction. Incoming unchanged-state replay and the carrier identities must meet1e-5.',
    'actual_weights_changed': True, 'diagnostic_hooks_only': True,
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'transform_code_sha256': hashlib.sha256((ROOT / 'late_selector_weights.py').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'training': False, 'gradients': False, 'optimizer': False,
    'free_generation_performed': False, 'counts_as_correction': False, 'final_set_read': False,
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
m = AutoModelForCausalLM.from_pretrained(ROOT.parent.parent / 'little-red-weightspace/models/original', local_files_only=True, torch_dtype=torch.float32).eval()
originals, weights = apply_late_selector(m)
(RDIR / f'{STEM}_weights.json').write_text(json.dumps(weights, indent=2), encoding='utf8')
parent_path = ROOT / 'patches/late_selector_parent_last_mlp.safetensors'
assert not parent_path.exists()
save_file({name: m.get_parameter(f'model.layers.23.mlp.{name}_proj.weight').clone().contiguous()
           for name in ['gate', 'up', 'down']}, parent_path)
captured = {}


def capture(name):
    def hook(module, args):
        captured[name] = args[0][0].clone()
    return hook


def capture_constant(module, args):
    captured['constant'] = args[0][0, :, output_index].clone()
    captured['selected_values'] = args[0][0, :, output_indices].clone()


hooks = [m.model.layers[23].register_forward_pre_hook(capture('incoming23')),
         m.model.layers[23].self_attn.o_proj.register_forward_pre_hook(capture_constant),
         m.model.layers[23].mlp.register_forward_pre_hook(capture('X')),
         m.model.layers[23].mlp.down_proj.register_forward_pre_hook(capture('A')),
         m.model.norm.register_forward_pre_hook(capture('R'))]
arrays = {name: [] for name in expected}
selected_values = []
constant_error = 0.0
try:
    for i in selected:
        sequence = sequences[i]
        m.model(input_ids=torch.tensor([sequence['ids']]), use_cache=False)
        positions = torch.tensor(sequence['positions'])
        for name in arrays:
            arrays[name].append(captured[name][positions].numpy().copy())
        selected_values.append(captured['selected_values'][positions].numpy().copy())
        constant_error = max(constant_error, float((captured['constant'] - 1).abs().max()))
        if i % 16 == 0:
            print('actual late selector', i + 1, '/', len(sequences), flush=True)
finally:
    for hook in hooks:
        hook.remove()
    restore_carrier(m, originals)
arrays = {name: np.concatenate(values) for name, values in arrays.items()}
errors = {name: {'relative': float(np.linalg.norm(arrays[name] - expected[name]) / np.linalg.norm(expected[name])),
                 'maximum_absolute': float(np.abs(arrays[name] - expected[name]).max())} for name in arrays}
values = np.concatenate(selected_values)
class_identity = arrays['R'].astype(np.float64) @ V - arrays['incoming23'].astype(np.float64) @ V - alpha * values
reference_identity = arrays['R'].astype(np.float64) @ reference - arrays['incoming23'].astype(np.float64) @ reference - reference_amplitude
class_error = float(np.linalg.norm(class_identity) / np.linalg.norm(alpha * values))
reference_error = float(np.linalg.norm(reference_identity) / np.linalg.norm(arrays['incoming23'].astype(np.float64) @ reference + reference_amplitude))
passed = max(constant_error, class_error, reference_error, errors['incoming23']['relative']) < 1e-5
arrays['selected_values'] = values
np.savez_compressed(RDIR / f'{STEM}.npz', record_indices=np.array(indices), **arrays)
report = {'spec': spec, 'weights': weights, 'sequences_checked': len(selected), 'positions_checked': len(indices),
          'class_coordinates': weights['class_coordinates'],
          'errors_against_local_reconstruction': errors, 'constant_value_max_absolute_error': constant_error,
          'class_carrier_identity_relative_error': class_error, 'reference_identity_relative_error': reference_error,
          'all_declared_tolerances_passed': passed, 'weights_restored_exactly': True, 'completed': True,
          'arrays_sha256': hashlib.sha256((RDIR / f'{STEM}.npz').read_bytes()).hexdigest(),
          'parent_last_mlp_sha256': hashlib.sha256(parent_path.read_bytes()).hexdigest()}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({key: value for key, value in report.items() if key not in ['spec', 'weights']}, flush=True)
assert passed
