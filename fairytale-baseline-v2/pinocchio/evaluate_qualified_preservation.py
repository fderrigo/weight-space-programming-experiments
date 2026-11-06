"""Ordinary observed checks plus frozen new three-condition comparison."""
import datetime
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from best_reference_runtime import load_reference, restore_original
from native_name_memory_helpers import sha, tensor_sha

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'qualified_native_preservation'
DEADLINE = datetime.datetime(2026, 11, 5, 19, 0, tzinfo=datetime.timezone.utc)
torch.set_grad_enabled(False)
torch.manual_seed(42)


def save(name, value):
    (RDIR / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


freeze_path = ROOT / 'sixth_development_freeze.json'
freeze = json.loads(freeze_path.read_text('utf8'))
assert sha(Path(__file__)) == freeze['evaluator_sha256']
assert sha(ROOT / 'sixth_development_questions.json') == freeze['questions_sha256']
candidate_path = ROOT / f'patches/{STEM}_basis.safetensors'
assert sha(candidate_path) == freeze['candidate_sha256']
queue = [
    ('screen', 'candidate', RDIR / 'qualified_concept_repairs_screen.json', False),
    ('fifth_observed', 'candidate', RDIR / 'fifth_development_qualified_concept_repairs.json', False),
    ('sixth_original', 'original', ROOT / 'sixth_development_questions.json', True),
    ('sixth_reference', 'reference', ROOT / 'sixth_development_questions.json', True),
    ('sixth_candidate', 'candidate', ROOT / 'sixth_development_questions.json', True),
    ('third_observed', 'candidate', RDIR / 'qualified_concept_repairs_third_observed.json', False),
    ('fourth_observed', 'candidate', RDIR / 'qualified_concept_repairs_fourth_observed.json', False),
]
assert all(not (RDIR / f'{STEM}_{label}.json').exists() for label, _, _, _ in queue)
model, tokenizer, originals, original_hashes = load_reference()
reference = {name: model.get_parameter(name).clone() for name in originals}
reference_hashes = {name: tensor_sha(parameter) for name, parameter in model.named_parameters()}
candidate = load_file(candidate_path)
for name in ['gate', 'up']:
    assert torch.equal(candidate[name + '_after'], model.get_parameter(f'model.layers.23.mlp.{name}_proj.weight'))
diagnostic = json.loads((RDIR / f'{STEM}_diagnostic.json').read_text('utf8'))
manifests = {}
try:
    for label, condition, source, fresh in queue:
        if condition == 'original':
            restore_original(model, originals, original_hashes)
        else:
            for name, value in reference.items():
                model.get_parameter(name).copy_(value)
            assert all(tensor_sha(p) == reference_hashes[name] for name, p in model.named_parameters())
            if condition == 'candidate':
                model.model.layers[23].mlp.down_proj.weight.copy_(candidate['down_after'])
                assert tensor_sha(model.model.layers[23].mlp.down_proj.weight) == diagnostic['last_mlp_differences'][2]['after_sha256_f32']
        assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
        tests = json.loads(source.read_text('utf8'))
        manifest = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'condition': condition,
                    'label': label, 'source': str(source.relative_to(ROOT)), 'source_sha256': sha(source),
                    'basis_sha256': sha(candidate_path), 'evaluator_sha256': sha(Path(__file__)),
                    'freeze_sha256': sha(freeze_path), 'responses_expected': len(tests), 'completed': False,
                    'runtime_hooks': False, 'provided_answer_prefix': False, 'corrective_context': False,
                    'actual_weights_changed': condition != 'original', 'fresh_development': fresh,
                    'do_sample': False, 'repetition_penalty': 1.1, 'batch_size': 4, 'threads': 6,
                    'seed': 42, 'training': False, 'gradients': False, 'optimizer': False, 'reserved_test_used': False}
        manifests[label] = manifest
        save(f'{STEM}_{label}_manifest.json', manifest)
        responses = []
        for group in ['target', 'control', 'narration']:
            selected = [r for r in tests if r['group'] == group]
            for start in range(0, len(selected), 4):
                assert datetime.datetime.now(datetime.timezone.utc) < DEADLINE, '20:00 Rome deadline'
                batch = selected[start:start + 4]
                maximum = batch[0].get('max_new_tokens', 128)
                assert all(r.get('max_new_tokens', 128) == maximum for r in batch)
                prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False, add_generation_prompt=True) for r in batch]
                encoded = tokenizer(prompts, padding=True, return_tensors='pt')
                generated = model.generate(**encoded, do_sample=False, repetition_penalty=1.1, max_new_tokens=maximum, pad_token_id=tokenizer.pad_token_id)
                for row, ids in zip(batch, generated[:, encoded['input_ids'].shape[1]:].tolist()):
                    stop = next((i for i, token in enumerate(ids) if token in [151643, 151645]), None)
                    tokens = ids if stop is None else ids[:stop]
                    response = {k: row[k] for k in ['id', 'prompt', 'group']}
                    response.update({'stage': label, 'response': tokenizer.decode(tokens, skip_special_tokens=True),
                                     'generated_ids': tokens, 'stop_token': None if stop is None else ids[stop],
                                     'done_reason': 'length' if stop is None else 'stop', 'max_new_tokens': maximum,
                                     'actual_weights_changed': condition != 'original', 'runtime_hooks': False,
                                     'reference_in_input': False, 'fresh_development': fresh})
                    responses.append(response)
                    print(label, row['id'], repr(response['response']), flush=True)
                save(f'{STEM}_{label}.json', responses)
                manifest['responses_completed'] = len(responses)
                save(f'{STEM}_{label}_manifest.json', manifest)
        manifest.update({'completed': True, 'responses_sha256': sha(RDIR / f'{STEM}_{label}.json')})
        save(f'{STEM}_{label}_manifest.json', manifest)
finally:
    restore_original(model, originals, original_hashes)
    for label, manifest in manifests.items():
        manifest['original_parameters_restored'] = True
        save(f'{STEM}_{label}_manifest.json', manifest)
    save(f'{STEM}_evaluation_manifest.json', {'completed_labels': [k for k, v in manifests.items() if v['completed']],
         'all_completed': len(manifests) == len(queue) and all(v['completed'] for v in manifests.values()), 'original_parameters_restored': True})
    print('Original parameters restored.', flush=True)
