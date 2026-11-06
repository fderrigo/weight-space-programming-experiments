"""Actual weight forwards for a single declared quieter, stronger carrier."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from progress_carrier_weights import apply_progress_carrier, restore_carrier

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'progress_carrier_states'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


source = RDIR / 'carrier_concept_states.json'
previous = json.loads(source.read_text('utf8'))
assert previous['completed']
sequences, records = previous['sequences'].copy(), previous['records'].copy()
qsource = RDIR / 'question_layer_diagnostic_states.json'
qmeta = json.loads(qsource.read_text('utf8'))['sequences']
for row in qmeta:
    if row['split'] == 'prototype' and row['label'] == 'outside':
        sequences.append({'id': row['id'], 'kind': 'outside_prototype', 'ids': row['ids'],
                          'positions': [len(row['ids']) - 1], 'offset': len(records), 'states': 1})
        records.append({'id': row['id'], 'kind': 'outside_prototype',
                        'position': len(row['ids']) - 1, 'last_token': row['ids'][-1]})
assert len(records) == 4707 and len(sequences) == 197
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'reason': 'Old carrier interference during partial concepts exceeded the semantic signal. The already fixed progress-aware basis reduces that interference. One fourfold amplitude makes its previously measured maximum partial interference/signal below 0.36; measure actual downstream changes rather than assuming preservation.',
    'geometry': 'results/progress_carrier_geometry.npz', 'amplitude_multiplier': 4.0,
    'conditions': 1, 'parameter_search': False, 'training': False, 'gradients': False,
    'optimizer': False, 'actual_weight_transform': True, 'diagnostic_only': True,
    'incomplete_concepts_are_not_corrected_answers': True,
    'free_generation_performed': False, 'third_set_used': False, 'final_set_read': False,
    'source_sha256': sha(source), 'outside_prototype_source_sha256': sha(qsource),
    'geometry_sha256': sha(RDIR / 'progress_carrier_geometry.npz'),
    'transform_code_sha256': sha(ROOT / 'progress_carrier_weights.py'),
    'collector_code_sha256': sha(Path(__file__)),
    'original_checkpoint_sha256': sha(MODEL / 'model.safetensors'),
}
assert spec['original_checkpoint_sha256'] == json.loads((ROOT / 'freeze.json').read_text('utf8'))['checkpoint_sha256']
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
originals, weights = apply_progress_carrier(m)
(RDIR / f'{STEM}_weights.json').write_text(json.dumps(weights, indent=2), encoding='utf8')
captured = {}


def capture_input(name):
    def hook(module, args):
        captured[name] = args[0][0].clone()
    return hook


def capture_values(module, args, output):
    value = output[0]
    captured['prefix_values'] = value.cumsum(0) / torch.arange(1, len(value) + 1)[:, None]


hooks = [m.model.layers[21].register_forward_pre_hook(capture_input('incoming21')),
         m.model.layers[21].self_attn.v_proj.register_forward_hook(capture_values),
         m.model.layers[23].mlp.register_forward_pre_hook(capture_input('X')),
         m.model.layers[23].mlp.down_proj.register_forward_pre_hook(capture_input('A')),
         m.model.norm.register_forward_pre_hook(capture_input('R'))]
arrays = {name: np.empty((len(records), width), dtype=np.float32)
          for name, width in [('incoming21', 896), ('prefix_values', 128), ('X', 896), ('A', 4864), ('R', 896)]}
for number, sequence in enumerate(sequences):
    m.model(input_ids=torch.tensor([sequence['ids']]), use_cache=False)
    positions = torch.tensor(sequence['positions'])
    begin, end = sequence['offset'], sequence['offset'] + sequence['states']
    for name in arrays:
        arrays[name][begin:end] = captured[name][positions].numpy()
    if number % 16 == 0:
        print('progress carrier histories', number + 1, '/', len(sequences), flush=True)
for hook in hooks:
    hook.remove()
restore_carrier(m, originals)
report = {'spec': spec, 'records': records, 'sequences': sequences, 'weights': weights,
          'weights_restored_exactly': True, 'completed': True}
with np.load(RDIR / 'carrier_concept_states_original.npz') as z:
    for name in ['incoming21', 'prefix_values']:
        error = float(np.linalg.norm(arrays[name][:4693] - z[name]) / np.linalg.norm(z[name]))
        assert error < 1e-5, (name, error)
        report[name + '_reference_relative_error'] = error
np.savez_compressed(RDIR / f'{STEM}.npz', **arrays)
report['arrays_sha256'] = sha(RDIR / f'{STEM}.npz')
(RDIR / f'{STEM}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print('Saved', len(records), 'states; original weights restored.', flush=True)
