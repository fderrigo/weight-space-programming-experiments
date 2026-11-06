"""Trace known control regressions and every currently correct observed answer."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from late_selector_weights import apply_late_selector, restore_carrier

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'native_recovery_states'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


sources = []
selected = []
for response_name, review_name, schema in [
    ('late_tail_relations_screen.json', 'late_tail_relations_review.json', 'screen'),
    ('late_tail_relations_third_observed.json', 'late_tail_relations_third_observed_review.json', 'third'),
    ('late_tail_relations_fourth_edited_hf.json', 'late_tail_relations_fourth_review.json', 'fourth'),
]:
    rows, review = read(response_name), read(review_name)
    expected = review['edited_source_sha256'] if schema == 'fourth' else review['source_sha256']
    assert sha(RDIR / response_name) == expected
    annotations = review['annotations'] if schema == 'screen' else review['targets'] + review['controls']
    if schema == 'screen':
        identities = {r['id'] for r in annotations if r['target_whole_correct'] or r['control_main_correct']}
    else:
        identities = {r['id'] for r in annotations if r['edited_whole_correct']}
    for row in rows:
        if row['id'] in identities:
            selected.append({**row, 'kind': 'preserve_current', 'source': response_name})
    sources.append({'responses': response_name, 'responses_sha256': expected,
                    'review': review_name, 'review_sha256': sha(RDIR / review_name),
                    'guard_response_ids': sorted(identities)})
for response_name, identities in [
    ('minimal_fact_completion_third_original_hf.json', ['third_control_04', 'third_control_17']),
    ('late_tail_relations_fourth_original_hf.json', ['fourth_control_07', 'fourth_control_15']),
]:
    rows = read(response_name)
    for row in rows:
        if row['id'] in identities:
            selected.append({**row, 'kind': 'restore_original', 'source': response_name})
    sources.append({'responses': response_name, 'responses_sha256': sha(RDIR / response_name),
                    'restore_response_ids': identities})
assert len({r['id'] for r in selected}) == len(selected)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
sequences, records = [], []
for row in selected:
    base = t.apply_chat_template([{'role': 'user', 'content': row['prompt']}], add_generation_prompt=True)
    ids = base + row['generated_ids']
    positions = list(range(len(base) - 1, len(ids) if row['stop_token'] is not None else len(ids) - 1))
    sequences.append({'id': row['id'], 'kind': row['kind'], 'source': row['source'],
                      'prompt': row['prompt'], 'ids': ids, 'base_length': len(base),
                      'offset': len(records), 'positions': positions, 'states': len(positions)})
    for j, pos in enumerate(positions):
        records.append({'id': row['id'], 'kind': row['kind'], 'subtoken': j, 'position': pos,
                        'next_token': row['generated_ids'][j] if j < len(row['generated_ids']) else row['stop_token']})
basis_path = ROOT / 'patches/late_tail_relations_basis.safetensors'
assert sha(basis_path) == '26cf2c34d06ea631e63a4066969eb1b66712782567a29e853d4ed941f67e0523'
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'hypothesis': 'The latest patch changes native decisions on four observed outside-control paths. Diagnose the first divergence and collect exact original residuals to restore, while retaining every currently correct observed answer from all three screens.',
    'restore_targets': 'The two whole-correct third controls lost (shark and blue), plus two lost correct main claims in flawed fourth controls (fish habitat and London). Restoring flawed original paths is preservation only, not new factual correctness.',
    'guard_rule': 'All strict whole-correct edited target/control responses in the three existing reviews, plus every correct main control in the eighteen-question screen. Preserve all their actual next-token states including native stops.',
    'candidate_basis_sha256': sha(basis_path), 'code_sha256': sha(Path(__file__)),
    'sources': sources, 'sequence_count': len(sequences), 'states': len(records),
    'weights_changed_for_diagnosis': True, 'temporary_read_only_capture_hooks': True,
    'free_generation_performed': False, 'counts_as_correction': False,
    'training': False, 'gradients': False, 'optimizer': False, 'final_set_read': False,
    'third_and_fourth_sets_already_observed': True,
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
assert sha(MODEL / 'model.safetensors') == 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
E = m.model.embed_tokens.weight * m.model.norm.weight
cache = {}
positions = None


def capture(name):
    def hook(module, args):
        cache[name] = args[0][0, positions].clone()
    return hook


handles = [m.model.layers[23].mlp.register_forward_pre_hook(capture('X')),
           m.model.layers[23].mlp.down_proj.register_forward_pre_hook(capture('A')),
           m.model.norm.register_forward_pre_hook(capture('R'))]
basis = load_file(basis_path)
histories = {}
for sequence in sequences:
    for j, pos in enumerate(sequence['positions']):
        histories[sequence['offset'] + j] = sorted(set(sequence['ids'][:pos + 1]))
summaries = {}
for condition in ['original', 'late_tail_relations']:
    originals = None
    if condition != 'original':
        originals, parent = apply_late_selector(m)
        assert parent == read('late_selector_states_weights.json')
        for name in ['gate', 'up', 'down']:
            p = m.get_parameter(f'model.layers.23.mlp.{name}_proj.weight')
            assert torch.equal(p, basis[name + '_before'])
            p.copy_(basis[name + '_after'])
    arrays = {key: np.zeros((len(records), width), dtype=np.float32)
              for key, width in [('X', 896), ('A', 4864), ('R', 896)]}
    try:
        for sequence in sequences:
            if condition == 'original' and sequence['kind'] != 'restore_original':
                continue
            positions = torch.tensor(sequence['positions'])
            m.model(input_ids=torch.tensor([sequence['ids']]), use_cache=False)
            start, end = sequence['offset'], sequence['offset'] + sequence['states']
            for key in arrays:
                arrays[key][start:end] = cache[key].numpy()
            print(condition, sequence['kind'], sequence['id'], sequence['states'], flush=True)
        indices = [i for i, row in enumerate(records) if condition != 'original' or row['kind'] == 'restore_original']
        decisions = []
        R = torch.from_numpy(arrays['R'])
        for start in range(0, len(indices), 16):
            block = indices[start:start + 16]
            residual = R[block]
            scores = (residual / (residual.square().mean(-1) + 1e-6).sqrt()[:, None]) @ E.T
            for index, score in zip(block, scores):
                seen = histories[index]
                score[seen] = torch.where(score[seen] < 0, score[seen] * 1.1, score[seen] / 1.1)
                token = int(score.argmax())
                decisions.append({'record_index': index, **records[index], 'winner': token,
                                  'winner_text': t.decode([token]),
                                  'recorded_next_rank': int((score > score[records[index]['next_token']]).sum()) + 1})
                if condition == 'original' or records[index]['kind'] == 'preserve_current':
                    assert token == records[index]['next_token'], decisions[-1]
        path = RDIR / f'{STEM}_{condition}.npz'
        np.savez_compressed(path, **arrays)
        summaries[condition] = {'arrays_sha256': sha(path), 'decisions': decisions,
                                'uncollected_guard_rows_are_zero': condition == 'original'}
    finally:
        if originals is not None:
            restore_carrier(m, originals)
for handle in handles:
    handle.remove()
divergences = []
for sequence in sequences:
    if sequence['kind'] != 'restore_original':
        continue
    first = next((r for r in summaries['late_tail_relations']['decisions']
                  if r['id'] == sequence['id'] and r['winner'] != r['next_token']), None)
    assert first is not None
    divergence = {**first, 'original_token_text': t.decode([first['next_token']]),
                  'shared_response_prefix': t.decode(sequence['ids'][sequence['base_length']:first['position'] + 1])}
    divergences.append(divergence)
    print('FIRST DIVERGENCE', divergence, flush=True)
report = {'spec': spec, 'sequences': sequences, 'records': records, 'conditions': summaries,
          'first_divergences': divergences, 'completed': True, 'weights_restored_in_memory': True,
          'original_paths_replayed_exactly': True, 'current_guard_paths_replayed_exactly': True}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print('Control recovery state collection complete.', flush=True)
