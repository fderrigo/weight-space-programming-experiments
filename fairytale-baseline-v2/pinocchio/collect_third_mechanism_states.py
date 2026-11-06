"""Trace supplied concepts and actual failed-transfer response histories."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from late_selector_weights import apply_late_selector, restore_carrier

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'third_mechanism_states'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
source = RDIR / 'late_margin_preservation_third_observed.json'
responses = json.loads(source.read_text('utf8'))
manifest = json.loads((RDIR / 'late_margin_preservation_third_observed_manifest.json').read_text('utf8'))
assert manifest['completed'] and hashlib.sha256(source.read_bytes()).hexdigest() == manifest['responses_sha256']
units = json.loads((RDIR / 'minimal_fact_units_spec.json').read_text('utf8'))['short_units']
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
sequences, records = [], []
for row in responses:
    base = t.apply_chat_template([{'role': 'user', 'content': row['prompt']}], add_generation_prompt=True)
    conditions = [('actual_response', row['generated_ids'])]
    if row['group'] == 'target':
        question = next(q for q in json.loads((ROOT / 'third_development_questions.json').read_text('utf8')) if q['id'] == row['id'])
        fact = question['fact']
        conditions.insert(0, ('canonical_fact', t.encode(units[fact], add_special_tokens=False)))
    else:
        fact = 'outside'
    for kind, tokens in conditions:
        ids = base + tokens
        positions = list(range(len(base) - 1, len(ids)))
        seq = {'id': row['id'], 'group': row['group'], 'fact': fact, 'kind': kind,
               'ids': ids, 'base_length': len(base), 'positions': positions,
               'offset': len(records), 'states': len(positions),
               'provided_concept': units[fact] if kind == 'canonical_fact' else None}
        sequences.append(seq)
        for j, position in enumerate(positions):
            next_token = tokens[j] if j < len(tokens) else (row['stop_token'] if kind == 'actual_response' else None)
            records.append({'id': row['id'], 'group': row['group'], 'fact': fact, 'kind': kind,
                            'position': position, 'subtoken': j, 'last_token': ids[position],
                            'next_token': next_token,
                            'unit_complete': kind == 'canonical_fact' and j == len(tokens)})
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'hypothesis': 'Original first-position semantic recognition exceeds whole-response repair. Separate relation selection, token progression, output writing and native continuation on the failed observed transfer set before changing weights.',
    'data': 'All44 actual frozen edited response histories plus24 canonical minimal-concept paths, including the completed-unit position. These supplied histories are diagnostic and never scored as repairs.',
    'conditions': ['original', 'late_selector_parent'],
    'collection': ['head14_13', 'prefix_values21', 'incoming23', 'X', 'A', 'R', 'selector_values'],
    'later_full_patch_reconstruction': 'The frozen margin patch changes only MLP23 relative to the collected parent. Reconstruct it on these states and compare predicted tokens with recorded actual generation at shared prefixes.',
    'source_sha256': manifest['responses_sha256'],
    'basis_sha256': manifest['basis_sha256'],
    'collection_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'parent_transform_sha256': hashlib.sha256((ROOT / 'late_selector_weights.py').read_bytes()).hexdigest(),
    'training': False, 'gradients': False, 'optimizer': False,
    'new_candidate_constructed': False, 'free_generation_performed': False,
    'counts_as_correction': False, 'third_set_already_observed': True, 'final_set_read': False,
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
with np.load(RDIR / 'late_template_selector.npz') as z:
    selected_indices = int(z['query_head']) * 64 + z['value_coordinates'][:12]
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
capture = {}


def keep(name):
    def hook(module, args):
        capture[name] = args[0][0].clone()
    return hook


def head_hook(module, args):
    capture['head14_13'] = args[0][0, :, 13 * 64:14 * 64].clone()


def value_hook(module, args, output):
    capture['prefix_values21'] = output[0].cumsum(0) / torch.arange(1, output.shape[1] + 1)[:, None]


def selector_hook(module, args):
    capture['selector_values'] = args[0][0, :, selected_indices].clone()


hooks = [m.model.layers[14].self_attn.o_proj.register_forward_pre_hook(head_hook),
         m.model.layers[21].self_attn.v_proj.register_forward_hook(value_hook),
         m.model.layers[23].register_forward_pre_hook(keep('incoming23')),
         m.model.layers[23].self_attn.o_proj.register_forward_pre_hook(selector_hook),
         m.model.layers[23].mlp.register_forward_pre_hook(keep('X')),
         m.model.layers[23].mlp.down_proj.register_forward_pre_hook(keep('A')),
         m.model.norm.register_forward_pre_hook(keep('R'))]
reports = []
for mode in spec['conditions']:
    originals = None
    weights = None
    if mode == 'late_selector_parent':
        originals, weights = apply_late_selector(m)
        assert weights == json.loads((RDIR / 'late_selector_states_weights.json').read_text('utf8'))
    arrays = {name: np.empty((len(records), width), dtype=np.float32) for name, width in
              [('head14_13', 64), ('prefix_values21', 128), ('incoming23', 896),
               ('X', 896), ('A', 4864), ('R', 896), ('selector_values', 12)]}
    try:
        for number, seq in enumerate(sequences):
            m.model(input_ids=torch.tensor([seq['ids']]), use_cache=False)
            positions = torch.tensor(seq['positions'])
            begin, end = seq['offset'], seq['offset'] + seq['states']
            for name in arrays:
                arrays[name][begin:end] = capture[name][positions].numpy()
            if number % 12 == 0:
                print(mode, number + 1, '/', len(sequences), flush=True)
    finally:
        if originals is not None:
            restore_carrier(m, originals)
    path = RDIR / f'{STEM}_{mode}.npz'
    np.savez_compressed(path, **arrays)
    reports.append({'condition': mode, 'weights': weights, 'arrays_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'weights_restored_exactly': originals is not None})
    (RDIR / f'{STEM}.json').write_text(json.dumps({'spec': spec, 'sequences': sequences, 'records': records,
                                                 'reports': reports, 'completed': len(reports) == 2}, indent=2), encoding='utf8')
for hook in hooks:
    hook.remove()
print('Completed mechanism collection', len(sequences), 'histories', len(records), 'positions', flush=True)
