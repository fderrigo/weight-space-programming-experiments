"""Capture original states on exact existing control/native/narrative histories.

No new supplied answers or generated responses. Narrative histories come from
the original side of the paired attention-gain run, not its edited responses.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'extended_native_inputs'
assert not (ROOT / f'results/{STEM}.npz').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
expanded = json.loads((ROOT / 'results/expanded_context_states.json').read_text('utf8'))
sequences = []
for kind, key in [('external', 'external_sequences'), ('native_unit', 'native_sequences')]:
    for row in expanded[key]:
        sequences.append({'id': row['id'], 'kind': kind, 'ids': row['ids'],
                          'start': row['start'], 'source': 'expanded_context_states.json'})
attention_source = ROOT / 'results/attention_gain_screen.json'
original_narratives = [row for row in json.loads(attention_source.read_text('utf8'))
                      if row['stage'] == 'original' and row['group'] == 'narration']
assert len(original_narratives) == 2
for row in original_narratives:
    prompt = t.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                                   tokenize=False, add_generation_prompt=True)
    base = t.encode(prompt, add_special_tokens=False)
    sequences.append({'id': row['id'], 'kind': 'narrative', 'ids': base + row['generated_ids'],
                      'start': len(base) - 1, 'source': 'attention_gain_screen.json original stage',
                      'stop_token': row['stop_token']})
m = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
capture = {}


def xhook(module, args):
    capture['X'] = args[0][0].clone()


def rhook(module, args):
    capture['R'] = args[0][0].clone()


hooks = [m.model.layers[23].mlp.register_forward_pre_hook(xhook),
         m.model.norm.register_forward_pre_hook(rhook)]
xs, rs, offset = [], [], 0
for index, row in enumerate(sequences):
    m.model(input_ids=torch.tensor([row['ids']]), use_cache=False)
    X, R = capture['X'][row['start']:], capture['R'][row['start']:]
    row['offset'], row['states'] = offset, len(X)
    offset += len(X)
    xs.append(X.numpy())
    rs.append(R.numpy())
    if index % 10 == 0:
        print('Exact original sequences', index + 1, '/', len(sequences), flush=True)
for hook in hooks:
    hook.remove()
np.savez_compressed(ROOT / f'results/{STEM}.npz', X=np.concatenate(xs), R=np.concatenate(rs))
with attention_source.open('rb') as f:
    source_hash = hashlib.file_digest(f, 'sha256').hexdigest()
metadata = {'sequences': sequences, 'states': offset, 'weights_changed': False,
            'training': False, 'gradients': False, 'counts_as_correction': False,
            'third_set_read': False, 'final_set_read': False,
            'narrative_source_sha256': source_hash,
            'purpose': 'Describe and preserve original language trajectories; preserving a false narrative is not correcting it.'}
(ROOT / f'results/{STEM}.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf8')
print('Saved', offset, 'native states from', len(sequences), 'existing sequences.', flush=True)
