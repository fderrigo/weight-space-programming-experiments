"""Ordinary free generation for the fixed native-recovery patch on observed sets."""
import datetime
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from late_selector_weights import apply_late_selector, restore_carrier
from semantic_carrier_weights import tensor_sha

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'late_native_recovery'


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


sets = [
    ('fourth_observed', 'late_tail_relations_fourth_original_hf.json',
     'late_tail_relations_fourth_edited_hf.json', 'late_tail_relations_fourth_review.json'),
    ('third_observed', 'minimal_fact_completion_third_original_hf.json',
     'late_tail_relations_third_observed.json', 'late_tail_relations_third_observed_review.json'),
    ('screen', 'attention_gain_screen.json', 'late_tail_relations_screen.json', 'late_tail_relations_review.json'),
]
assert all(not (RDIR / f'{STEM}_{name}.json').exists() for name, *_ in sets)
basis_path = ROOT / f'patches/{STEM}_basis.safetensors'
spec_path = ROOT / f'patches/{STEM}_spec.json'
spec = json.loads(spec_path.read_text('utf8'))
diagnostic = read(f'{STEM}_diagnostic.json')
assert diagnostic['all_new_recorded_decisions_match_requested_native_or_preserved_path']
assert sha(ROOT / 'late_selector_weights.py') == spec['parent_transform_code_sha256']
assert sha(RDIR / 'late_selector_geometry.npz') == spec['geometry_sha256']
trace = read('native_recovery_states.json')
protected = {r['id'] for r in trace['sequences'] if r['kind'] == 'preserve_current'}
restore_ids = {r['id'] for r in trace['sequences'] if r['kind'] == 'restore_original'}
torch.set_grad_enabled(False)
torch.set_num_threads(4)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
t.padding_side = 'left'
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
originals, parent = apply_late_selector(m)
assert parent == read('late_selector_states_weights.json')
basis = load_file(basis_path)
for name in ['gate', 'up', 'down']:
    weight = m.get_parameter(f'model.layers.23.mlp.{name}_proj.weight')
    assert torch.equal(weight, basis[name + '_before'])
    weight.copy_(basis[name + '_after'])
    expected = next(r for r in diagnostic['last_mlp_differences'] if r['tensor'] == f'model.layers.23.mlp.{name}_proj.weight')
    assert tensor_sha(weight) == expected['after_sha256_f32']
original_hash = sha(MODEL / 'model.safetensors')
assert original_hash == 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
torch.set_num_threads(6)
manifest_paths = []
try:
    for label, original_name, previous_name, review_name in sets:
        original_rows, previous_rows, review = read(original_name), read(previous_name), read(review_name)
        if label == 'screen':
            tests = [r for r in original_rows if r['stage'] == 'original']
            assert sha(RDIR / original_name) == review['original_source_sha256']
        else:
            tests = original_rows
            assert sha(RDIR / original_name) == review['original_source_sha256']
        previous_hash = review['edited_source_sha256'] if label == 'fourth_observed' else review['source_sha256']
        assert sha(RDIR / previous_name) == previous_hash
        previous_by_id = {r['id']: r for r in previous_rows}
        assert len(tests) == (18 if label == 'screen' else 44)
        manifest = {
            'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'candidate': STEM,
            'set': label, 'basis_sha256': sha(basis_path), 'spec_sha256': sha(spec_path),
            'original_checkpoint_sha256': original_hash,
            'original_responses_source': original_name, 'original_responses_source_sha256': sha(RDIR / original_name),
            'previous_responses_source': previous_name, 'previous_responses_source_sha256': previous_hash,
            'evaluator_sha256': sha(Path(__file__)),
            'sampling': {'runtime': 'Transformers', 'dtype': 'float32', 'do_sample': False,
                         'repetition_penalty': 1.1, 'seed': 42, 'batch_size': 4, 'threads': 6,
                         'max_new_tokens': {'facts_controls': 128, 'narratives': 256}},
            'same_runtime_template_and_batch_grouping': True, 'actual_weights_changed': True,
            'runtime_hooks': False, 'corrective_context': False,
            'training': False, 'gradients': False, 'optimizer': False,
            'calibration_only': True, 'fresh_validation': False,
            'question_set_already_observed': True, 'final_set_read': False,
            'changed_parameters': diagnostic['changed_parameters'], 'changed_tensors': diagnostic['changed_tensors'],
            'responses_expected': len(tests), 'completed': False, 'preservation_checks': [],
        }
        output = RDIR / f'{STEM}_{label}.json'
        manifest_path = RDIR / f'{STEM}_{label}_manifest.json'
        manifest_paths.append(manifest_path)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
        torch.manual_seed(42)
        responses = []
        for group in ['target', 'control', 'narration']:
            selected = [r for r in tests if r['group'] == group]
            limit = 256 if group == 'narration' else 128
            for start in range(0, len(selected), 4):
                batch = selected[start:start + 4]
                prompts = [t.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False,
                                                add_generation_prompt=True) for r in batch]
                enc = t(prompts, padding=True, return_tensors='pt')
                generated = m.generate(**enc, do_sample=False, repetition_penalty=1.1,
                                       max_new_tokens=limit, pad_token_id=t.pad_token_id)
                for row, tokens in zip(batch, generated[:, enc['input_ids'].shape[1]:].tolist()):
                    assert row['max_new_tokens'] == limit
                    stop = next((i for i, token in enumerate(tokens) if token in [151643, 151645]), None)
                    ids = tokens if stop is None else tokens[:stop]
                    response = {k: row[k] for k in ['id', 'group', 'prompt']}
                    response.update({'stage': 'edited_' + STEM, 'response': t.decode(ids, skip_special_tokens=True),
                                     'generated_ids': ids, 'stop_token': None if stop is None else tokens[stop],
                                     'done_reason': 'length' if stop is None else 'stop', 'max_new_tokens': limit,
                                     'actual_weights_changed': True, 'runtime_hooks': False,
                                     'fresh_validation': False, 'calibration_only': True, 'final_set_read': False})
                    responses.append(response)
                    if row['id'] in protected | restore_ids:
                        reference = row if row['id'] in restore_ids else previous_by_id[row['id']]
                        exact = ids == reference['generated_ids'] and response['stop_token'] == reference['stop_token']
                        manifest['preservation_checks'].append({'id': row['id'], 'restore_original': row['id'] in restore_ids,
                                                                'exact_response_ids_and_stop': exact})
                    print(label, row['id'], repr(response['response']), flush=True)
                output.write_text(json.dumps(responses, ensure_ascii=False, indent=2), encoding='utf8')
                manifest['responses_completed'] = len(responses)
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
        manifest.update({'completed': True, 'responses_sha256': sha(output),
                         'all_declared_preservation_checks_passed': all(r['exact_response_ids_and_stop'] for r in manifest['preservation_checks'])})
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
        print('COMPLETED', label, len(responses), 'preservation', manifest['all_declared_preservation_checks_passed'], flush=True)
finally:
    restore_carrier(m, originals)
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text('utf8'))
        manifest['weights_restored_in_memory'] = True
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
