"""Paired original/frozen-candidate free generation on new frozen questions."""
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
STEM = 'late_tail_relations_fourth'
manifest_path = RDIR / f'{STEM}_evaluation_manifest.json'
assert not manifest_path.exists()


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


freeze_path = ROOT / 'fourth_development_freeze.json'
freeze = json.loads(freeze_path.read_text('utf8'))
question_path = ROOT / 'fourth_development_questions.json'
assert sha(question_path) == freeze['question_sha256']
assert freeze['candidate_fixed_before_question_creation'] and not freeze['outputs_generated']
basis_path = ROOT / 'patches/late_tail_relations_basis.safetensors'
spec_path = ROOT / 'patches/late_tail_relations_spec.json'
assert sha(basis_path) == freeze['candidate_basis_sha256']
assert sha(spec_path) == freeze['candidate_spec_sha256']
spec = json.loads(spec_path.read_text('utf8'))
assert sha(ROOT / 'late_selector_weights.py') == spec['parent_transform_code_sha256']
assert sha(RDIR / 'late_selector_geometry.npz') == spec['geometry_sha256']
rows = json.loads(question_path.read_text('utf8'))
assert len(rows) == 44 and sum(r['group'] == 'target' for r in rows) == 24
outputs = {mode: RDIR / f'{STEM}_{mode}_hf.json' for mode in ['original', 'edited']}
assert not any(p.exists() for p in outputs.values())
torch.set_grad_enabled(False)
torch.set_num_threads(4)
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
basis = load_file(basis_path)
diagnostic = json.loads((RDIR / 'late_tail_relations_diagnostic.json').read_text('utf8'))
manifest = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'candidate': 'late_tail_relations', 'basis_sha256': sha(basis_path), 'spec_sha256': sha(spec_path),
    'freeze_sha256': sha(freeze_path), 'question_sha256': sha(question_path),
    'original_checkpoint_sha256': sha(MODEL / 'model.safetensors'),
    'evaluator_sha256': sha(Path(__file__)),
    'sampling': {'runtime': 'Transformers', 'dtype': 'float32', 'do_sample': False,
                 'repetition_penalty': 1.1, 'seed': 42, 'batch_size': 4, 'threads': 6,
                 'max_new_tokens': 128, 'template': 'unchanged original tokenizer chat template'},
    'same_runtime_template_and_batch_grouping': True,
    'candidate_fixed_before_question_creation': True, 'fresh_development_validation': True,
    'final_set_read': False, 'runtime_hooks': False, 'corrective_context': False,
    'training': False, 'gradients': False, 'optimizer': False,
    'changed_parameters': diagnostic['changed_parameters'], 'changed_tensors': diagnostic['changed_tensors'],
    'current_exported_checkpoint_attempt': 6, 'responses_expected_per_condition': 44,
    'completed_conditions': [], 'completed': False}
assert manifest['original_checkpoint_sha256'] == json.loads((ROOT / 'freeze.json').read_text('utf8'))['checkpoint_sha256']
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
for mode in ['original', 'edited']:
    originals = None
    if mode == 'edited':
        torch.set_num_threads(4)
        originals, parent = apply_late_selector(model)
        assert parent == json.loads((RDIR / 'late_selector_states_weights.json').read_text('utf8'))
        for name in ['gate', 'up', 'down']:
            weight = model.get_parameter(f'model.layers.23.mlp.{name}_proj.weight')
            assert torch.equal(weight, basis[name + '_before'])
            weight.copy_(basis[name + '_after'])
            expected = next(r for r in diagnostic['last_mlp_differences'] if r['tensor'] == f'model.layers.23.mlp.{name}_proj.weight')
            assert tensor_sha(weight) == expected['after_sha256_f32']
    torch.manual_seed(42)
    torch.set_num_threads(6)
    responses = []
    try:
        for group in ['target', 'control']:
            selected = [r for r in rows if r['group'] == group]
            for start in range(0, len(selected), 4):
                batch = selected[start:start + 4]
                prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False, add_generation_prompt=True) for r in batch]
                enc = tokenizer(prompts, padding=True, return_tensors='pt')
                generated = model.generate(**enc, do_sample=False, repetition_penalty=1.1,
                                           max_new_tokens=128, pad_token_id=tokenizer.pad_token_id)
                for row, tokens in zip(batch, generated[:, enc['input_ids'].shape[1]:].tolist()):
                    stop = next((i for i, token in enumerate(tokens) if token in [151643, 151645]), None)
                    ids = tokens if stop is None else tokens[:stop]
                    response = {k: row[k] for k in ['id', 'group', 'fact', 'prompt']}
                    response.update({'stage': mode + '_late_tail_relations_fourth',
                                     'response': tokenizer.decode(ids, skip_special_tokens=True),
                                     'generated_ids': ids, 'stop_token': None if stop is None else tokens[stop],
                                     'done_reason': 'length' if stop is None else 'stop', 'max_new_tokens': 128,
                                     'actual_weights_changed': mode == 'edited', 'runtime_hooks': False,
                                     'fresh_development_validation': True, 'final_set_read': False})
                    responses.append(response)
                    print(mode, row['id'], repr(response['response']), flush=True)
                outputs[mode].write_text(json.dumps(responses, ensure_ascii=False, indent=2), encoding='utf8')
    finally:
        if originals is not None:
            restore_carrier(model, originals)
            manifest['weights_restored_in_memory'] = True
    assert len(responses) == 44
    manifest['completed_conditions'].append(mode)
    manifest[mode + '_responses_sha256'] = sha(outputs[mode])
    manifest['completed'] = len(manifest['completed_conditions']) == 2
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
print('Completed paired fresh development', 44, 'answers per condition', flush=True)
