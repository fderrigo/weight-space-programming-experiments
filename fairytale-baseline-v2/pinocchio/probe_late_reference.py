"""Validate the reconstructed late edit in actual full-model forwards."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from late_reference_weights import CONDITION, apply_late_reference, restore_carrier

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'late_reference_actual_check'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
source = RDIR / 'progress_carrier_states.json'
meta = json.loads(source.read_text('utf8'))
sequences = meta['sequences']
selected = {next(i for i, s in enumerate(sequences) if s['kind'] == kind) for kind in {s['kind'] for s in sequences}}
for kind, identity in [('external', 'second_control_10'), ('minimal_native', 'second_fairy_0'), ('external', 'long_water_cycle')]:
    selected.add(next(i for i, s in enumerate(sequences) if s['kind'] == kind and s['id'] == identity))
selected = sorted(selected)
indices = [j for i in selected for j in range(sequences[i]['offset'], sequences[i]['offset'] + sequences[i]['states'])]
with np.load(RDIR / f'late_{CONDITION}_reconstruction.npz') as z:
    expected = {name: z[name][indices].copy() for name in ['incoming23', 'X', 'A', 'R']}
with np.load(RDIR / 'late_reference_geometry.npz') as z:
    output_index = int(z['sharing_output_indices'][0])
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'reason': 'Check the last-block reconstruction against actual changed-weight forwards before using it to construct fact-writing matrices.',
    'condition': CONDITION,
    'sample_rule': 'First sequence of each recorded kind plus the arithmetic deviation, completed Azzurri continuation, and long water-cycle control.',
    'sequence_indices': selected, 'record_indices': indices,
    'scope': 'Full-model verification on this declared representative sample, not all197 histories. Other4707-position arrays remain local last-block reconstructions.',
    'relative_tolerance': 1e-5, 'constant_max_absolute_tolerance': 1e-5,
    'actual_weights_changed': True, 'diagnostic_hooks_only': True,
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'transform_code_sha256': hashlib.sha256((ROOT / 'late_reference_weights.py').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'training': False, 'gradients': False, 'optimizer': False,
    'free_generation_performed': False, 'counts_as_correction': False, 'final_set_read': False,
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
m = AutoModelForCausalLM.from_pretrained(ROOT.parent.parent / 'little-red-weightspace/models/original', local_files_only=True, torch_dtype=torch.float32).eval()
originals, weights = apply_late_reference(m)
(RDIR / f'{STEM}_weights.json').write_text(json.dumps(weights, indent=2), encoding='utf8')
parent_path = ROOT / 'patches/late_reference_parent_last_mlp.safetensors'
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


hooks = [m.model.layers[23].register_forward_pre_hook(capture('incoming23')),
         m.model.layers[23].self_attn.o_proj.register_forward_pre_hook(capture_constant),
         m.model.layers[23].mlp.register_forward_pre_hook(capture('X')),
         m.model.layers[23].mlp.down_proj.register_forward_pre_hook(capture('A')),
         m.model.norm.register_forward_pre_hook(capture('R'))]
arrays = {name: [] for name in expected}
constant_error = 0.0
try:
    for i in selected:
        sequence = sequences[i]
        m.model(input_ids=torch.tensor([sequence['ids']]), use_cache=False)
        positions = torch.tensor(sequence['positions'])
        for name in arrays:
            arrays[name].append(captured[name][positions].numpy().copy())
        constant_error = max(constant_error, float((captured['constant'] - 1).abs().max()))
        print('actual late reference', sequence['kind'], sequence['id'], sequence['states'], flush=True)
finally:
    for hook in hooks:
        hook.remove()
    restore_carrier(m, originals)
arrays = {name: np.concatenate(values) for name, values in arrays.items()}
errors = {name: {'relative': float(np.linalg.norm(arrays[name] - expected[name]) / np.linalg.norm(expected[name])),
                 'maximum_absolute': float(np.abs(arrays[name] - expected[name]).max())} for name in arrays}
passed = constant_error < 1e-5 and all(row['relative'] < 1e-5 for row in errors.values())
np.savez_compressed(RDIR / f'{STEM}.npz', record_indices=np.array(indices), **arrays)
report = {'spec': spec, 'weights': weights, 'sequences_checked': len(selected), 'positions_checked': len(indices),
          'errors_against_local_reconstruction': errors, 'constant_value_max_absolute_error': constant_error,
          'all_declared_tolerances_passed': passed, 'weights_restored_exactly': True, 'completed': True,
          'arrays_sha256': hashlib.sha256((RDIR / f'{STEM}.npz').read_bytes()).hexdigest(),
          'parent_last_mlp_sha256': hashlib.sha256(parent_path.read_bytes()).hexdigest()}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({key: value for key, value in report.items() if key not in ['spec', 'weights']}, flush=True)
assert passed
