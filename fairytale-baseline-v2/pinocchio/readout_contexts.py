"""Read saved native states in the exact order of the expanded factual basis."""
import json

import numpy as np
import torch
from safetensors.torch import load_file


def load_contexts(root, tokenizer):
    oldmeta = root / 'results/orthogonal_story_roles_2048_controls_native_states.json'
    oldcases = json.loads(oldmeta.read_text('utf8'))['case_records']
    units = json.loads((root / 'patches/multipoint_protected_spec.json').read_text('utf8'))['short_units']
    first = json.loads((root / 'new_development_questions.json').read_text('utf8'))
    newcases = []
    for row in first:
        if row['group'] != 'target':
            continue
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                                                tokenize=False, add_generation_prompt=True)
        newcases.append({'fact': row['fact'] + '_answer_start', 'question': row['id'],
                         'prefix': '', 'base': tokenizer.encode(prompt, add_special_tokens=False),
                         'concept': tokenizer.encode(units[row['fact']], add_special_tokens=False)})
    expanded = json.loads((root / 'results/expanded_context_states.json').read_text('utf8'))
    groups = [(oldcases, oldmeta.with_suffix('.npz')),
              (newcases, root / 'results/multipoint_protected_new_states.npz'),
              (expanded['cases'], root / 'results/expanded_context_states.npz')]
    keys, residuals, records, seen = [], [], [], {}
    for cases, path in groups:
        with np.load(path) as arrays:
            for i, case in enumerate(cases):
                for j, token in enumerate(case['concept']):
                    sequence = tuple(case['base'] + case['concept'][:j])
                    if sequence in seen:
                        assert seen[sequence] == token
                        continue
                    seen[sequence] = token
                    keys.append(torch.from_numpy(arrays[f'A_{i}'][j].copy()))
                    residuals.append(torch.from_numpy(arrays[f'R_{i}'][j].copy()))
                    records.append({'fact': case['fact'], 'question': case['question'],
                                    'prefix': case['prefix'], 'token': token,
                                    'piece': tokenizer.decode([token]), 'subtoken': j,
                                    'history': list(sequence)})
    K, R = torch.stack(keys), torch.stack(residuals)
    basis = load_file(root / 'patches/expanded_context_basis.safetensors')
    assert torch.equal(K, basis['keys'])
    return K, R, records, basis
