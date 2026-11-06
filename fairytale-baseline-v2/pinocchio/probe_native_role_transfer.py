"""Causal role-binding diagnostic from known-correct native continuations.

Both sides use the same already-declared short factual unit. Transfer only
native block-14 states, once during the prefix forward. No complete response,
stopping token, trained editor, weight changes, or scored correction.
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
STEM = 'native_role_transfer'
output_path = ROOT / f'results/{STEM}.json'
assert not output_path.exists()
torch.set_grad_enabled(False)
torch.set_num_threads(6)
torch.manual_seed(42)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
native_path = ROOT / 'results/native_second_continuations.json'
native = {row['id']: row for row in json.loads(native_path.read_text('utf8'))}
review = json.loads((ROOT / 'results/native_second_continuations_review.json').read_text('utf8'))
correct = {row['id'] for row in review['annotations'] if row['correct_with_supplied_unit']}
pairs = [('second_creator_0', 'second_creator_1'),
         ('second_puppeteer_1', 'second_puppeteer_0'),
         ('second_rescuer_0', 'second_rescuer_1')]
assert all(source in correct and target not in correct for source, target in pairs)
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'pairs': [{'source': source, 'target': target} for source, target in pairs],
    'layer': 14, 'conditions': ['original', 'last_unit_token', 'entire_unit'],
    'reason': 'These pairs share a supplied short unit but differ in whether the native continuation preserves the role. Use donors whose complete native continuations were already reviewed as correct, instead of reference-context donors that themselves produced false answers. Block 14 was previously informative about question relations; fix that block before seeing outcomes.',
    'intervention': 'Copy the donor block-14 output at the matching last unit token, or at every matching unit token, during the initial prefix forward only.',
    'sampling': {'do_sample': False, 'repetition_penalty': 1.1, 'max_new_tokens': 96,
                 'seed': 42, 'threads': 6, 'batch_size': 3},
    'weights_changed': False, 'training': False, 'gradients': False, 'optimizer': False,
    'supplied_prefix_diagnostic_only': True, 'counts_as_correction': False,
    'full_responses_copied': False, 'EOS_targeted': False,
    'third_set_read': False, 'final_set_read': False,
    'native_source_sha256': hashlib.sha256(native_path.read_bytes()).hexdigest(),
}
(ROOT / f'results/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
m = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
capture = {}


def capture_h(module, inputs, output):
    hidden = output[0] if isinstance(output, tuple) else output
    capture['H'] = hidden[0].clone()


def capture_a(module, inputs):
    capture['A'] = inputs[0][0].clone()


hooks = [m.model.layers[14].register_forward_hook(capture_h),
         m.model.layers[14].mlp.down_proj.register_forward_pre_hook(capture_a)]
donors, arrays, notes = [], {}, []
for i, (source, target) in enumerate(pairs):
    assert native[source]['provided_unit'] == native[target]['provided_unit']
    length = len(t.encode(native[source]['provided_unit'], add_special_tokens=False))
    for side, key in [('donor', source), ('target', target)]:
        ids = native[key]['base_ids']
        m.model(input_ids=torch.tensor([ids]), use_cache=False)
        arrays[f'{side}_H_{i}'] = capture['H'][-length:].numpy()
        arrays[f'{side}_A_{i}'] = capture['A'][-length:].numpy()
    donors.append(torch.from_numpy(arrays[f'donor_H_{i}']))
    notes.append({'source': source, 'target': target, 'unit_tokens': length,
                  'last_token_transfer_norm': float(torch.from_numpy(arrays[f'donor_H_{i}'][-1] - arrays[f'target_H_{i}'][-1]).norm()),
                  'all_unit_transfer_norm': float(torch.from_numpy(arrays[f'donor_H_{i}'] - arrays[f'target_H_{i}']).norm())})
for hook in hooks:
    hook.remove()
np.savez_compressed(ROOT / f'results/{STEM}_states.npz', **arrays)
target_rows = [native[target] for _, target in pairs]
bases = [row['base_ids'] for row in target_rows]
width = max(map(len, bases))
ids = torch.tensor([[t.pad_token_id] * (width - len(base)) + base for base in bases])
mask = torch.tensor([[0] * (width - len(base)) + [1] * len(base) for base in bases])
out = []
for condition in spec['conditions']:
    handle = None
    if condition != 'original':
        def transfer(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[1] != width:
                return output
            hidden = hidden.clone()
            for i, donor in enumerate(donors):
                if condition == 'last_unit_token':
                    hidden[i, -1] = donor[-1]
                else:
                    hidden[i, -len(donor):] = donor
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
        handle = m.model.layers[14].register_forward_hook(transfer)
    try:
        generated = m.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                               repetition_penalty=1.1, max_new_tokens=96,
                               pad_token_id=t.pad_token_id)
    finally:
        if handle is not None:
            handle.remove()
    for row, tokens, note in zip(target_rows, generated[:, width:].tolist(), notes):
        stop = next((i for i, token in enumerate(tokens) if token in [151643, 151645]), None)
        answer_ids = tokens if stop is None else tokens[:stop]
        continuation = t.decode(answer_ids, skip_special_tokens=True)
        out.append({'id': row['id'], 'prompt': row['prompt'], 'fact': row['fact'],
                    'condition': condition, 'source_id': note['source'],
                    'provided_unit': row['provided_unit'], 'continuation': continuation,
                    'response_with_supplied_unit': row['provided_unit'] + continuation,
                    'generated_ids': answer_ids, 'stop_token': None if stop is None else tokens[stop],
                    'done_reason': 'length' if stop is None else 'stop',
                    'weights_changed': False, 'temporary_activation_intervention': condition != 'original',
                    'counts_as_correction': False, 'supplied_prefix_diagnostic_only': True})
        print(condition, row['id'], repr(row['provided_unit'] + continuation), flush=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf8')
(ROOT / f'results/{STEM}_geometry.json').write_text(json.dumps(notes, indent=2), encoding='utf8')
print('Completed', len(out), 'diagnostic continuations; no weight correction claimed.', flush=True)
