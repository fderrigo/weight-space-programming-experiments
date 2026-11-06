"""Diagnostic native continuations after minimal factual concepts.

The previous grammatical role phrases sometimes introduce the wrong subject.
Use bare names or the smallest predicate/concept for all twelve relations.
Supplied prefixes are diagnostic only; no weight edit or correction is scored.
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
STEM = 'minimal_fact_units'
output = ROOT / f'results/{STEM}.json'
assert not output.exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
rows = [row for row in json.loads((ROOT / 'second_development_questions.json').read_text('utf8'))
        if row['group'] == 'target']
units = {
    'creator': 'Geppetto', 'material': 'Legno', 'nose': 'Si allunga', 'fairy': 'Azzurri',
    'deceivers': 'Il Gatto e la Volpe', 'donkey': 'Un asino', 'ending': 'Un bambino vero',
    'puppeteer': 'Mangiafoco', 'friend': 'Lucignolo', 'wood_donor': 'Maestro Ciliegia',
    'rescuer': 'Alidoro', 'fish_father': 'Geppetto',
}
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'short_units': units,
    'reason': 'The original role phrases sometimes become a false grammatical subject, and block-14 transfer from correct native continuations did not repair those roles. Test minimal names, nouns, and predicates for every factual relation, without composing a complete response or specifying a stop token.',
    'sampling': {'do_sample': False, 'repetition_penalty': 1.1, 'max_new_tokens': 64,
                 'batch_size': 4, 'torch_threads': 4},
    'previous_native_diagnostic': 'results/native_second_continuations.json',
    'weights_changed': False, 'training': False, 'gradients': False, 'optimizer': False,
    'supplied_prefix_diagnostic_only': True, 'counts_as_correction': False,
    'EOS_programmed': False, 'full_responses_prescribed': False,
    'third_set_read': False, 'final_set_read': False,
}
(ROOT / f'results/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
m = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
out = []
for begin in range(0, len(rows), 4):
    batch = rows[begin:begin + 4]
    bases = []
    for row in batch:
        prompt = t.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                                      tokenize=False, add_generation_prompt=True)
        bases.append(t.encode(prompt, add_special_tokens=False)
                     + t.encode(units[row['fact']], add_special_tokens=False))
    width = max(map(len, bases))
    ids = torch.tensor([[t.pad_token_id] * (width - len(base)) + base for base in bases])
    mask = torch.tensor([[0] * (width - len(base)) + [1] * len(base) for base in bases])
    generated = m.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                           repetition_penalty=1.1, max_new_tokens=64,
                           pad_token_id=t.pad_token_id)
    for row, base, tokens in zip(batch, bases, generated[:, width:].tolist()):
        stop = next((i for i, token in enumerate(tokens) if token in [151643, 151645]), None)
        answer = tokens if stop is None else tokens[:stop]
        continuation = t.decode(answer, skip_special_tokens=True)
        out.append({**row, 'provided_unit': units[row['fact']], 'continuation': continuation,
                    'base_ids': base, 'generated_ids': answer,
                    'response_with_supplied_unit': units[row['fact']] + continuation,
                    'stop_token': None if stop is None else tokens[stop],
                    'done_reason': 'length' if stop is None else 'stop',
                    'weights_changed': False, 'runtime_hooks_during_generation': False,
                    'counts_as_correction': False, 'supplied_prefix_diagnostic_only': True})
        print(row['id'], repr(units[row['fact']] + continuation), flush=True)
    output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf8')
capture = {}


def keep(name):
    def hook(module, inputs):
        capture[name] = inputs[0][0].clone()
    return hook


hooks = [m.model.layers[23].mlp.register_forward_pre_hook(keep('X')),
         m.model.layers[23].mlp.down_proj.register_forward_pre_hook(keep('A')),
         m.model.norm.register_forward_pre_hook(keep('R'))]
arrays, sequences = {}, []
for i, row in enumerate(out):
    full = row['base_ids'] + row['generated_ids']
    start = len(row['base_ids']) - 1
    m.model(input_ids=torch.tensor([full]), use_cache=False)
    for name in ['X', 'A', 'R']:
        arrays[f'{name}_{i}'] = capture[name][start:].numpy()
    sequences.append({'id': row['id'], 'ids': full, 'start': start,
                      'states': len(arrays[f'A_{i}']), 'supplied_prefix_diagnostic_only': True})
for hook in hooks:
    hook.remove()
np.savez_compressed(ROOT / f'results/{STEM}_states.npz', **arrays)
metadata = {'sequences': sequences, 'weights_changed': False, 'training': False,
            'source_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'third_set_read': False, 'final_set_read': False, 'counts_as_correction': False}
(ROOT / f'results/{STEM}_states.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
print('Completed', len(out), 'supplied-concept diagnostics; no correction claimed.', flush=True)
