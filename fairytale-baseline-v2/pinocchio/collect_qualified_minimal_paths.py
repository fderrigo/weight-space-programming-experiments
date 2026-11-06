"""Protect observed native continuations of the two qualified minimal concepts."""
import datetime
import json
from pathlib import Path
import numpy as np
import torch
from transformers import RepetitionPenaltyLogitsProcessor
from best_reference_runtime import load_reference, restore_original
from native_name_memory_helpers import sha, remove

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'qualified_minimal_paths'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
responses = json.loads((RDIR / 'best_minimal_repairs_responses.json').read_text('utf8'))
source_manifest = json.loads((RDIR / 'best_minimal_repairs_manifest.json').read_text('utf8'))
assert source_manifest['completed'] and source_manifest['responses_sha256'] == sha(RDIR / 'best_minimal_repairs_responses.json')
judgments = {
    'second_creator_0': (True, 'Correct Geppetto and explicit distinction from the book author.'),
    'second_material_0': (False, 'Wood is correct but the maker is falsely called Giuseppe Cilicchi.'),
    'second_nose_0': (False, 'Head, lips and unrelated shadow claims instead of faithful nose elongation.'),
    'second_deceivers_0': (True, 'Correct complete Cat and Fox answer with native punctuation and stop.'),
    'second_ending_0': (False, 'Correct initial real-boy words followed by the invented title Il Collido.'),
    'second_puppeteer_0': (False, 'Correct name followed by invented commercial claims and publication date1904; truncated.'),
}
selected = [row for row in responses if judgments[row['id']][0]]
assert len(selected) == 2
spec = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'selection': [{'id': row['id'], 'whole_correct_with_supplied_concept': judgments[row['id']][0],
                       'reason': judgments[row['id']][1]} for row in responses],
        'purpose': 'Collect invariance constraints for actual native continuations, including existing stop preferences. These are not prescribed new answers or EOS targets.',
        'source_sha256': sha(RDIR / 'best_minimal_repairs_responses.json'),
        'code_sha256': sha(Path(__file__)), 'helper_sha256': sha(ROOT / 'best_reference_runtime.py'),
        'provided_answer_prefix': True, 'counts_as_correction': False,
        'training': False, 'gradients': False, 'optimizer': False}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf8')
m, t, originals, original_hashes = load_reference()
penalty = RepetitionPenaltyLogitsProcessor(1.1)
cache, handles = {}, []


def capture(key):
    def hook(module, args):
        cache[key] = args[0][0].detach().clone()
    return hook


arrays, records, sequences = {'A': [], 'R': []}, [], []
try:
    handles.append(m.model.layers[23].mlp.down_proj.register_forward_pre_hook(capture('A')))
    handles.append(m.model.norm.register_forward_pre_hook(capture('R')))
    for row in selected:
        text = t.apply_chat_template([{'role': 'user', 'content': row['prompt']}], tokenize=False, add_generation_prompt=True)
        prompt = t(text)['input_ids']
        ids = prompt + row['provided_concept_ids'] + row['continuation_ids']
        start = len(prompt) + len(row['provided_concept_ids']) - 1
        positions = list(range(start, len(ids)))
        desired = row['continuation_ids'] + [row['stop_token']]
        assert row['stop_token'] in [151643, 151645] and len(positions) == len(desired)
        output = m.model(input_ids=torch.tensor([ids]), use_cache=False)
        for position, expected in zip(positions, desired):
            score = penalty(torch.tensor([ids[:position + 1]]), m.lm_head(output.last_hidden_state[:, position]).clone())[0]
            assert int(score.argmax()) == expected, (row['id'], position)
            records.append({'id': row['id'], 'position': position, 'next_token': expected,
                            'history_ids': ids[:position + 1], 'index': len(records)})
            for key in arrays:
                arrays[key].append(cache[key][position].numpy())
        sequences.append({'id': row['id'], 'ids': ids, 'positions': positions, 'states': len(positions)})
        print('Verified native continuation', row['id'], len(positions), flush=True)
    np.savez_compressed(RDIR / f'{STEM}.npz', **{key: np.stack(values) for key, values in arrays.items()})
    report = {'spec': spec, 'completed': True, 'records': records, 'sequences': sequences,
              'arrays_sha256': sha(RDIR / f'{STEM}.npz'), 'all_native_decisions_replayed': True}
finally:
    remove(handles)
    restore_original(m, originals, original_hashes)
report['original_weights_restored'] = True
(RDIR / f'{STEM}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
