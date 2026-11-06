"""Original-model narrative continuations after one supplied opening fact.

Diagnostic only: supplied assistant prefixes are forbidden as a final solution
and none of these responses counts as a weight-edit improvement.
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'native_narrative_opening'
output = ROOT / f'results/{STEM}.json'
assert not output.exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
torch.manual_seed(42)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
m = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
narratives = [row for row in json.loads((ROOT / 'atlas.json').read_text('utf8'))
              if row['group'] == 'narration']
openings = {
    'maker_first': 'Geppetto costruisce Pinocchio, un burattino di legno.',
    'puppet_first': 'Pinocchio nasce come burattino di legno, costruito dal falegname Geppetto.',
}
cases = []
for opening_id, opening in openings.items():
    for row in narratives:
        prefix = ('1. ' if row['id'] == 'narration_1' else '') + opening
        prompt = t.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                                       tokenize=False, add_generation_prompt=True)
        base = t.encode(prompt, add_special_tokens=False) + t.encode(prefix, add_special_tokens=False)
        cases.append({**row, 'opening_id': opening_id, 'provided_prefix': prefix, 'base_ids': base})
width = max(len(row['base_ids']) for row in cases)
ids = torch.tensor([[t.pad_token_id] * (width - len(row['base_ids'])) + row['base_ids'] for row in cases])
mask = torch.tensor([[0] * (width - len(row['base_ids'])) + [1] * len(row['base_ids']) for row in cases])
generated = m.generate(input_ids=ids, attention_mask=mask, max_new_tokens=256, do_sample=False,
                       repetition_penalty=1.1, pad_token_id=t.pad_token_id)
responses = []
for row, tokens in zip(cases, generated[:, width:].tolist()):
    stop = next((j for j, token in enumerate(tokens) if token in [151643, 151645]), None)
    answer_ids = tokens if stop is None else tokens[:stop]
    record = {**row, 'continuation': t.decode(answer_ids, skip_special_tokens=True),
              'generated_ids': answer_ids, 'stop_token': None if stop is None else tokens[stop],
              'done_reason': 'length' if stop is None else 'stop',
              'weights_changed': False, 'counts_as_correction': False,
              'supplied_prefix_diagnostic_only': True, 'runtime_hooks_during_generation': False,
              'third_set_read': False, 'final_set_read': False}
    responses.append(record)
    print(row['opening_id'], row['id'], repr(record['continuation']), flush=True)
output.write_text(json.dumps(responses, ensure_ascii=False, indent=2), encoding='utf8')
capture, arrays = {}, {}


def hook(name):
    def capture_state(module, args):
        capture[name] = args[0][0].clone()
    return capture_state


handles = [m.model.layers[23].mlp.register_forward_pre_hook(hook('X')),
           m.model.layers[23].mlp.down_proj.register_forward_pre_hook(hook('A')),
           m.model.norm.register_forward_pre_hook(hook('R'))]
for index, row in enumerate(responses):
    m.model(input_ids=torch.tensor([row['base_ids'] + row['generated_ids']]), use_cache=False)
    start = len(row['base_ids']) - 1
    for name in ['X', 'A', 'R']:
        arrays[f'{name}_{index}'] = capture[name][start:].numpy()
for handle in handles:
    handle.remove()
np.savez_compressed(ROOT / f'results/{STEM}_states.npz', **arrays)
print('Saved four supplied-opening diagnostics; zero claimed corrections.', flush=True)
