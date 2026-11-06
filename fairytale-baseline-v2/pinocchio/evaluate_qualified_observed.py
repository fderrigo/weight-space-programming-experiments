"""Check the frozen edit on two previously observed sets; never fresh evidence."""
import datetime
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from best_reference_runtime import load_reference, restore_original
from native_name_memory_helpers import sha, tensor_sha

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'qualified_concept_repairs'
DEADLINE = datetime.datetime(2026, 11, 5, 19, 0, tzinfo=datetime.timezone.utc)
SOURCES = {
    'third': 'minimal_fact_completion_third_original_hf.json',
    'fourth': 'late_tail_relations_fourth_original_hf.json',
}
torch.set_grad_enabled(False)
torch.manual_seed(42)


def save(name, value):
    (RDIR / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


candidate_path = ROOT / f'patches/{STEM}_basis.safetensors'
assert sha(candidate_path) == '40ddba2f0d5fc147ad6a20f0cf4b92ec5c1aa73898c88bf145f566b419614b7e'
assert all(not (RDIR / f'{STEM}_{label}_observed.json').exists() for label in SOURCES)
source_manifests = {
    'third': json.loads((RDIR / 'minimal_fact_completion_third_evaluation_manifest.json').read_text('utf8')),
    'fourth': json.loads((RDIR / 'late_tail_relations_fourth_evaluation_manifest.json').read_text('utf8')),
}
for label, filename in SOURCES.items():
    assert source_manifests[label]['completed']
    assert sha(RDIR / filename) == source_manifests[label]['original_responses_sha256']
model, tokenizer, originals, original_hashes = load_reference()
candidate = load_file(candidate_path)
for name in ['gate', 'up']:
    assert torch.equal(candidate[name + '_after'], model.get_parameter(f'model.layers.23.mlp.{name}_proj.weight'))
model.model.layers[23].mlp.down_proj.weight.copy_(candidate['down_after'])
candidate_hashes = {name: tensor_sha(parameter) for name, parameter in model.named_parameters()}
assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
manifests = {}
try:
    for label, filename in SOURCES.items():
        rows = json.loads((RDIR / filename).read_text('utf8'))
        assert len(rows) == 44 and sum(row['group'] == 'target' for row in rows) == 24
        manifest = {
            'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'candidate': STEM, 'set': label, 'basis_sha256': sha(candidate_path),
            'source_sha256': sha(RDIR / filename),
            'evaluator_sha256': sha(Path(__file__)),
            'source_manifest_sha256': sha(RDIR / ('minimal_fact_completion_third_evaluation_manifest.json' if label == 'third' else 'late_tail_relations_fourth_evaluation_manifest.json')),
            'sampling': source_manifests[label]['sampling'],
            'actual_weights_changed': True, 'runtime_hooks': False,
            'corrective_context': False, 'provided_answer_prefix': False,
            'training': False, 'gradients': False, 'optimizer': False,
            'fresh_validation': False, 'previously_observed_questions': True,
            'reserved_test_used': False, 'responses_expected': 44, 'completed': False,
        }
        manifests[label] = manifest
        save(f'{STEM}_{label}_observed_manifest.json', manifest)
        responses = []
        for group in ['target', 'control']:
            selected = [row for row in rows if row['group'] == group]
            for start in range(0, len(selected), 4):
                assert datetime.datetime.now(datetime.timezone.utc) < DEADLINE, '20:00 Rome deadline'
                batch = selected[start:start + 4]
                prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}], tokenize=False, add_generation_prompt=True) for row in batch]
                encoded = tokenizer(prompts, padding=True, return_tensors='pt')
                generated = model.generate(**encoded, do_sample=False, repetition_penalty=1.1, max_new_tokens=128, pad_token_id=tokenizer.pad_token_id)
                for row, ids in zip(batch, generated[:, encoded['input_ids'].shape[1]:].tolist()):
                    assert row['max_new_tokens'] == 128
                    stop = next((i for i, token in enumerate(ids) if token in [151643, 151645]), None)
                    tokens = ids if stop is None else ids[:stop]
                    response = {key: row[key] for key in ['id', 'prompt', 'group']}
                    response.update({'stage': f'{STEM}_{label}_observed',
                                     'response': tokenizer.decode(tokens, skip_special_tokens=True),
                                     'generated_ids': tokens, 'stop_token': None if stop is None else ids[stop],
                                     'done_reason': 'length' if stop is None else 'stop', 'max_new_tokens': 128,
                                     'actual_weights_changed': True, 'runtime_hooks': False,
                                     'fresh_validation': False, 'previously_observed_questions': True})
                    responses.append(response)
                    print(label, row['id'], repr(response['response']), flush=True)
                save(f'{STEM}_{label}_observed.json', responses)
                manifest['responses_completed'] = len(responses)
                save(f'{STEM}_{label}_observed_manifest.json', manifest)
        assert all(tensor_sha(parameter) == candidate_hashes[name] for name, parameter in model.named_parameters())
        manifest.update({'completed': True, 'responses_sha256': sha(RDIR / f'{STEM}_{label}_observed.json'), 'candidate_unchanged_during_evaluation': True})
        save(f'{STEM}_{label}_observed_manifest.json', manifest)
finally:
    restore_original(model, originals, original_hashes)
    for label, manifest in manifests.items():
        manifest['original_parameters_restored'] = True
        save(f'{STEM}_{label}_observed_manifest.json', manifest)
    print('Original parameters restored.', flush=True)
