"""Diagnostic native continuations after the already declared short facts.

These are supplied assistant prefixes, never scored as corrected responses.
The original weights remain unchanged. Store exact generated IDs and internal
states to distinguish native false continuations from editing-induced damage.
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'native_second_continuations'
output = ROOT / f'results/{STEM}.json'
assert not output.exists(), 'Do not overwrite previous response evidence'
torch.set_grad_enabled(False)
torch.set_num_threads(4)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
m = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
rows = [r for r in json.loads((ROOT / 'second_development_questions.json').read_text('utf8'))
        if r['group'] == 'target']
units = json.loads((ROOT / 'patches/multipoint_protected_spec.json').read_text('utf8'))['short_units']
out, arrays, collected = [], {}, []
for begin in range(0, len(rows), 4):
    batch = rows[begin:begin + 4]
    bases = []
    for row in batch:
        prompt = t.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                                      tokenize=False, add_generation_prompt=True)
        base = t.encode(prompt, add_special_tokens=False)
        unit = t.encode(units[row['fact']], add_special_tokens=False)
        bases.append(base + unit)
    width = max(map(len, bases))
    ids = torch.tensor([[t.pad_token_id] * (width - len(b)) + b for b in bases])
    mask = torch.tensor([[0] * (width - len(b)) + [1] * len(b) for b in bases])
    generated = m.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                           repetition_penalty=1.1, max_new_tokens=64,
                           pad_token_id=t.pad_token_id)
    for row, base, tokens in zip(batch, bases, generated[:, width:].tolist()):
        stop = next((i for i, tok in enumerate(tokens) if tok in [151643, 151645]), None)
        answer = tokens if stop is None else tokens[:stop]
        record = {**row, 'provided_unit': units[row['fact']],
                  'continuation': t.decode(answer, skip_special_tokens=True),
                  'generated_ids': answer, 'base_ids': base,
                  'stop_token': None if stop is None else tokens[stop],
                  'done_reason': 'length' if stop is None else 'stop',
                  'weights_changed': False, 'supplied_prefix_diagnostic_only': True,
                  'counts_as_correction': False}
        out.append(record)
        print(row['id'], repr(record['provided_unit'] + record['continuation']), flush=True)
    output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf8')

capture = {}

def xhook(module, inputs):
    capture['X'] = inputs[0][0].clone()

def ahook(module, inputs):
    capture['A'] = inputs[0][0].clone()

def rhook(module, inputs):
    capture['R'] = inputs[0][0].clone()

hooks = [m.model.layers[23].mlp.register_forward_pre_hook(xhook),
         m.model.layers[23].mlp.down_proj.register_forward_pre_hook(ahook),
         m.model.norm.register_forward_pre_hook(rhook)]
for i, row in enumerate(out):
    full = row['base_ids'] + row['generated_ids']
    m.model(input_ids=torch.tensor([full]))
    start = len(row['base_ids']) - 1
    for name in ['X', 'A', 'R']:
        arrays[f'{name}_{i}'] = capture[name][start:].numpy()
    collected.append({'id': row['id'], 'start': start, 'states': len(arrays[f'A_{i}']),
                      'ids': full, 'supplied_prefix_diagnostic_only': True})
for hook in hooks:
    hook.remove()
np.savez_compressed(ROOT / f'results/{STEM}_states.npz', **arrays)
(ROOT / f'results/{STEM}_states.json').write_text(
    json.dumps({'sequences': collected, 'weights_changed': False,
                'third_set_read': False, 'final_set_read': False,
                'counts_as_correction': False}, ensure_ascii=False, indent=2), encoding='utf8')
print('Saved native completion states for', len(out), 'diagnostic prefixes', flush=True)
