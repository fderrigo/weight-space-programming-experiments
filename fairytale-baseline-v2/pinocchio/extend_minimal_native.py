"""Finish truncated native diagnostics at the ordinary 128-token answer limit.

No editing or corrective information is added. The original model continues its
own exact token history. These supplied-concept diagnostics prove no correction.
"""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'minimal_fact_units_complete'
source = ROOT / 'results/minimal_fact_units.json'
output = ROOT / f'results/{STEM}.json'
assert not output.exists()
rows = json.loads(source.read_text('utf8'))
selected = [i for i, row in enumerate(rows) if row['done_reason'] == 'length']
assert [rows[i]['id'] for i in selected] == ['second_material_0', 'second_rescuer_1']
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'decision_by': 'Codex', 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'reason': 'Align the two length-truncated native continuations with the ordinary 128-token full-answer limit before preserving them in a weight edit.',
    'max_full_answer_tokens': 128, 'repetition_penalty': 1.1, 'do_sample': False,
    'extended_ids': [rows[i]['id'] for i in selected],
    'weights_changed': False, 'training': False, 'gradients': False,
    'supplied_prefix_diagnostic_only': True, 'counts_as_correction': False,
    'EOS_programmed': False, 'third_set_read': False, 'final_set_read': False,
}
(ROOT / f'results/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
torch.set_grad_enabled(False)
torch.set_num_threads(4)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
m = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
for i in selected:
    row = rows[i]
    unit_length = len(t.encode(row['provided_unit'], add_special_tokens=False))
    remaining = 128 - unit_length - len(row['generated_ids'])
    assert remaining > 0
    history = row['base_ids'] + row['generated_ids']
    ids = torch.tensor([history])
    generated = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                           max_new_tokens=remaining, do_sample=False,
                           repetition_penalty=1.1, pad_token_id=t.pad_token_id)
    tokens = generated[0, len(history):].tolist()
    stop = next((j for j, token in enumerate(tokens) if token in [151643, 151645]), None)
    new_tokens = tokens if stop is None else tokens[:stop]
    row['previous_generated_count'] = len(row['generated_ids'])
    row['generated_ids'] += new_tokens
    row['continuation'] = t.decode(row['generated_ids'], skip_special_tokens=True)
    row['response_with_supplied_unit'] = row['provided_unit'] + row['continuation']
    row['stop_token'] = None if stop is None else tokens[stop]
    row['done_reason'] = 'length' if stop is None else 'stop'
    row['max_full_answer_tokens'] = 128
    print(row['id'], row['done_reason'], repr(row['response_with_supplied_unit']), flush=True)
output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf8')
with np.load(ROOT / 'results/minimal_fact_units_states.npz') as z:
    arrays = {name: z[name].copy() for name in z.files}
capture = {}


def keep(name):
    def hook(module, inputs):
        capture[name] = inputs[0][0].clone()
    return hook


hooks = [m.model.layers[23].mlp.register_forward_pre_hook(keep('X')),
         m.model.layers[23].mlp.down_proj.register_forward_pre_hook(keep('A')),
         m.model.norm.register_forward_pre_hook(keep('R'))]
prefix_errors = {}
for i in selected:
    row = rows[i]
    full = row['base_ids'] + row['generated_ids']
    start = len(row['base_ids']) - 1
    m.model(input_ids=torch.tensor([full]), use_cache=False)
    for name in ['X', 'A', 'R']:
        old = arrays[f'{name}_{i}']
        new = capture[name][start:].numpy()
        error = float(np.linalg.norm(new[:len(old)] - old) / np.linalg.norm(old))
        assert error < 5e-5, (i, name, error)
        prefix_errors[f'{name}_{i}'] = error
        arrays[f'{name}_{i}'] = new
for hook in hooks:
    hook.remove()
np.savez_compressed(ROOT / f'results/{STEM}_states.npz', **arrays)
sequences = [{'id': row['id'], 'ids': row['base_ids'] + row['generated_ids'],
              'start': len(row['base_ids']) - 1, 'states': len(arrays[f'A_{i}'])}
             for i, row in enumerate(rows)]
metadata = {**spec, 'completed_source_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'sequences': sequences, 'prefix_state_relative_errors': prefix_errors,
            'total_states': sum(row['states'] for row in sequences)}
(ROOT / f'results/{STEM}_states.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
print('Completed native preservation states:', metadata['total_states'], flush=True)
