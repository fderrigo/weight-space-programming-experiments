"""Paired ordinary evaluation of original, best reference and frozen new edit."""
import datetime
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from best_reference_runtime import load_reference, restore_original
from native_name_memory_helpers import sha, tensor_sha

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'fifth_development'
torch.set_grad_enabled(False)
torch.manual_seed(42)


def save(name, value):
    (RDIR / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def log(message):
    with (RDIR / f'{STEM}.log').open('a', encoding='utf8') as stream:
        stream.write(datetime.datetime.now(datetime.timezone.utc).isoformat() + ' ' + message + '\n')
    print(message, flush=True)


question_path = ROOT / f'{STEM}_questions.json'
freeze_path = ROOT / f'{STEM}_freeze.json'
freeze = json.loads(freeze_path.read_text('utf8'))
tests = json.loads(question_path.read_text('utf8'))
assert len(tests) == 44 and len({row['id'] for row in tests}) == 44
assert sha(question_path) == freeze['questions_sha256']
assert sha(Path(__file__)) == freeze['evaluator_sha256']
candidate_path = ROOT / 'patches/qualified_concept_repairs_basis.safetensors'
assert sha(candidate_path) == freeze['candidate_sha256']
assert sha(ROOT / 'patches/late_native_recovery_basis.safetensors') == freeze['reference_sha256']
assert freeze['candidate_fixed_before_question_creation']
labels = ['original', 'native_recovery', 'qualified_concept_repairs']
assert all(not (RDIR / f'{STEM}_{label}.json').exists() for label in labels)
m, t, originals, original_hashes = load_reference()
best = {name: m.get_parameter(name).clone() for name in originals}
best_hashes = {name: tensor_sha(p) for name, p in m.named_parameters()}
candidate = load_file(candidate_path)
for name in ['gate', 'up']:
    assert torch.equal(candidate[name + '_after'], m.get_parameter(f'model.layers.23.mlp.{name}_proj.weight'))
base_down = m.model.layers[23].mlp.down_proj.weight.clone()
manifests = {}
completed = False
try:
    for label in labels:
        if label == 'original':
            restore_original(m, originals, original_hashes)
        else:
            for name, value in best.items():
                m.get_parameter(name).copy_(value)
            assert all(tensor_sha(p) == best_hashes[name] for name, p in m.named_parameters())
            if label == 'qualified_concept_repairs':
                m.model.layers[23].mlp.down_proj.weight.copy_(candidate['down_after'])
        assert all(not module._forward_hooks and not module._forward_pre_hooks for module in m.modules())
        manifest = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'condition': label, 'freeze_sha256': sha(freeze_path),
                    'questions_sha256': freeze['questions_sha256'],
                    'candidate_sha256': freeze['candidate_sha256'],
                    'reference_sha256': freeze['reference_sha256'],
                    'evaluator_sha256': sha(Path(__file__)), 'runtime_hooks': False,
                    'corrective_context': False, 'provided_answer_prefix': False,
                    'do_sample': False, 'repetition_penalty': 1.1, 'seed': 42, 'threads': 6,
                    'fresh_development': True, 'reserved_test_used': False,
                    'training': False, 'gradients': False, 'optimizer': False,
                    'actual_weights_changed': label != 'original',
                    'responses_expected': 44, 'completed': False}
        manifests[label] = manifest
        save(f'{STEM}_{label}_manifest.json', manifest)
        responses = []
        for group in ['target', 'control']:
            selected = [row for row in tests if row['group'] == group]
            for start in range(0, len(selected), 4):
                assert datetime.datetime.now(datetime.timezone.utc) < datetime.datetime(2026, 11, 5, 19, 0, tzinfo=datetime.timezone.utc), '20:00 Rome deadline'
                batch = selected[start:start + 4]
                text = [t.apply_chat_template([{'role': 'user', 'content': row['prompt']}], tokenize=False, add_generation_prompt=True) for row in batch]
                encoded = t(text, padding=True, return_tensors='pt')
                generated = m.generate(**encoded, do_sample=False, repetition_penalty=1.1, max_new_tokens=128, pad_token_id=t.pad_token_id)
                for row, ids in zip(batch, generated[:, encoded['input_ids'].shape[1]:].tolist()):
                    stop = next((i for i, token in enumerate(ids) if token in [151643, 151645]), None)
                    tokens = ids if stop is None else ids[:stop]
                    record = {**row, 'stage': f'{STEM}_{label}', 'response': t.decode(tokens, skip_special_tokens=True),
                              'generated_ids': tokens, 'stop_token': None if stop is None else ids[stop],
                              'done_reason': 'length' if stop is None else 'stop', 'max_new_tokens': 128,
                              'actual_weights_changed': label != 'original', 'runtime_hooks': False,
                              'reference_in_input': False, 'fresh_development': True}
                    responses.append(record)
                    log(label + ' ' + row['id'] + ' ' + repr(record['response']))
                save(f'{STEM}_{label}.json', responses)
                manifest['responses_completed'] = len(responses)
                save(f'{STEM}_{label}_manifest.json', manifest)
        manifest.update({'completed': True, 'responses_sha256': sha(RDIR / f'{STEM}_{label}.json')})
        save(f'{STEM}_{label}_manifest.json', manifest)
    completed = True
finally:
    restore_original(m, originals, original_hashes)
    for label, manifest in manifests.items():
        manifest['original_weights_restored_after_evaluation'] = True
        save(f'{STEM}_{label}_manifest.json', manifest)
    save(f'{STEM}_comparison_manifest.json', {'completed': completed,
         'freeze_sha256': sha(freeze_path), 'conditions_completed': [label for label, manifest in manifests.items() if manifest['completed']],
         'original_parameters_restored': True, 'runtime_hooks': False})
    log('Comparison completed=' + str(completed) + '; original weights restored.')
