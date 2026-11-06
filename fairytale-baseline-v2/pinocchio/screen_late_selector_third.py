"""Evaluate a fixed late-selector patch on the already observed third set.

This is development generalization evidence, never a fresh or final test.
The original answers are reused from their verified, identical HF evaluation.
"""
import datetime
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from late_selector_weights import apply_late_selector, restore_carrier
from semantic_carrier_weights import tensor_sha

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = sys.argv[1]
assert STEM.isidentifier()
output = ROOT / f'results/{STEM}_third_observed.json'
assert not output.exists()


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


source = ROOT / 'results/minimal_fact_completion_third_original_hf.json'
old_manifest = json.loads((ROOT / 'results/minimal_fact_completion_third_evaluation_manifest.json').read_text('utf8'))
assert old_manifest['completed'] and sha(source) == old_manifest['original_responses_sha256']
tests = json.loads(source.read_text('utf8'))
assert len(tests) == 44 and sum(r['group'] == 'target' for r in tests) == 24
spec_path = ROOT / f'patches/{STEM}_spec.json'
basis_path = ROOT / f'patches/{STEM}_basis.safetensors'
spec = json.loads(spec_path.read_text('utf8'))
diagnostic = json.loads((ROOT / f'results/{STEM}_diagnostic.json').read_text('utf8'))
assert sha(ROOT / 'late_selector_weights.py') == spec['parent_transform_code_sha256']
assert sha(ROOT / 'results/late_selector_geometry.npz') == spec['geometry_sha256']
basis = load_file(basis_path)
torch.set_grad_enabled(False)
torch.set_num_threads(4)
torch.manual_seed(42)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
t.padding_side = 'left'
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
originals, parent = apply_late_selector(m)
assert parent == json.loads((ROOT / 'results/late_selector_states_weights.json').read_text('utf8'))
for name in ['gate', 'up', 'down']:
    weight = m.get_parameter(f'model.layers.23.mlp.{name}_proj.weight')
    assert torch.equal(weight, basis[name + '_before'])
    weight.copy_(basis[name + '_after'])
    expected = next(r for r in diagnostic['last_mlp_differences'] if r['tensor'] == f'model.layers.23.mlp.{name}_proj.weight')
    assert tensor_sha(weight) == expected['after_sha256_f32']
torch.set_num_threads(6)
manifest = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'basis_sha256': sha(basis_path), 'spec_sha256': sha(spec_path),
    'original_checkpoint_sha256': sha(MODEL / 'model.safetensors'),
    'original_responses_source_sha256': sha(source), 'evaluator_sha256': sha(Path(__file__)),
    'question_sha256': old_manifest['question_sha256'],
    'sampling': old_manifest['sampling'], 'same_runtime_template_and_batch_grouping': True,
    'actual_weights_changed': True, 'runtime_hooks': False, 'corrective_context': False,
    'training': False, 'gradients': False, 'optimizer': False,
    'calibration_only': True, 'fresh_validation': False, 'third_set_already_observed': True,
    'candidate_fixed_before_questions_ever_observed': False, 'final_set_read': False,
    'changed_parameters': diagnostic['changed_parameters'],
    'changed_tensors': diagnostic['changed_tensors'], 'current_exported_checkpoint_attempt': 6,
    'responses_expected': len(tests), 'completed': False,
}
assert manifest['original_checkpoint_sha256'] == old_manifest['original_checkpoint_sha256']
manifest_path = ROOT / f'results/{STEM}_third_observed_manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
responses = []
for group in ['target', 'control']:
    selected = [r for r in tests if r['group'] == group]
    for begin in range(0, len(selected), 4):
        batch = selected[begin:begin + 4]
        prompts = [t.apply_chat_template([{'role': 'user', 'content': r['prompt']}],
                                         tokenize=False, add_generation_prompt=True) for r in batch]
        enc = t(prompts, padding=True, return_tensors='pt')
        generated = m.generate(**enc, do_sample=False, repetition_penalty=1.1,
                               max_new_tokens=128, pad_token_id=t.pad_token_id)
        for row, tokens in zip(batch, generated[:, enc['input_ids'].shape[1]:].tolist()):
            assert row['max_new_tokens'] == 128
            stop = next((i for i, token in enumerate(tokens) if token in [151643, 151645]), None)
            ids = tokens if stop is None else tokens[:stop]
            response = {key: row[key] for key in ['id', 'prompt', 'group']}
            response.update({'stage': 'edited_' + STEM,
                             'response': t.decode(ids, skip_special_tokens=True),
                             'generated_ids': ids, 'stop_token': None if stop is None else tokens[stop],
                             'done_reason': 'length' if stop is None else 'stop', 'max_new_tokens': 128,
                             'actual_weights_changed': True, 'runtime_hooks': False,
                             'calibration_only': True, 'fresh_validation': False,
                             'third_set_already_observed': True})
            responses.append(response)
            print(row['id'], repr(response['response']), flush=True)
        output.write_text(json.dumps(responses, ensure_ascii=False, indent=2), encoding='utf8')
restore_carrier(m, originals)
manifest.update({'responses_completed': len(responses), 'responses_sha256': sha(output),
                 'completed': True, 'weights_restored_in_memory': True})
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
