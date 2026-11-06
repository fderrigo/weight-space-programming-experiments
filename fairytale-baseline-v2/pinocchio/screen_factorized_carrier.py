"""Unassisted ordinary generation with the saved factorized carrier patch."""
import datetime
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from semantic_carrier_weights import apply_carrier, restore_carrier, tensor_sha

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'factorized_carrier'
output = ROOT / f'results/{STEM}_screen.json'
assert not output.exists()


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


source = ROOT / 'results/attention_gain_screen.json'
source_manifest = json.loads((ROOT / 'results/attention_gain_manifest.json').read_text('utf8'))
assert sha(source) == source_manifest['responses_sha256']
tests = [r for r in json.loads(source.read_text('utf8')) if r['stage'] == 'original']
assert len(tests) == 18
spec_path = ROOT / f'patches/{STEM}_spec.json'
basis_path = ROOT / f'patches/{STEM}_basis.safetensors'
spec = json.loads(spec_path.read_text('utf8'))
diagnostic = json.loads((ROOT / f'results/{STEM}_diagnostic.json').read_text('utf8'))
assert sha(ROOT / 'semantic_carrier_weights.py') == spec['parent_transform_code_sha256']
assert sha(ROOT / 'results/semantic_carrier_geometry.npz') == spec['geometry_sha256']
basis = load_file(basis_path)
torch.set_grad_enabled(False)
torch.set_num_threads(4)
torch.manual_seed(42)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
t.padding_side = 'left'
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
originals, parent = apply_carrier(m, 'protected')
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
    'sampling': source_manifest['settings'], 'same_runtime_template_and_batch_grouping': True,
    'actual_weights_changed': True, 'runtime_hooks': False, 'corrective_context': False,
    'training': False, 'gradients': False, 'optimizer': False,
    'calibration_only': True, 'fresh_validation': False, 'final_set_read': False,
    'changed_parameters': diagnostic['changed_parameters'],
    'changed_tensors': diagnostic['changed_tensors'], 'current_exported_checkpoint_attempt': 6,
    'responses_expected': len(tests), 'completed': False,
}
assert manifest['original_checkpoint_sha256'] == json.loads((ROOT / 'freeze.json').read_text('utf8'))['checkpoint_sha256']
manifest_path = ROOT / f'results/{STEM}_screen_manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
responses = []
for group in ['target', 'control', 'narration']:
    selected = [r for r in tests if r['group'] == group]
    limit = 256 if group == 'narration' else 128
    for begin in range(0, len(selected), 4):
        batch = selected[begin:begin + 4]
        prompts = [t.apply_chat_template([{'role': 'user', 'content': r['prompt']}],
                                         tokenize=False, add_generation_prompt=True) for r in batch]
        enc = t(prompts, padding=True, return_tensors='pt')
        generated = m.generate(**enc, do_sample=False, repetition_penalty=1.1,
                               max_new_tokens=limit, pad_token_id=t.pad_token_id)
        for row, tokens in zip(batch, generated[:, enc['input_ids'].shape[1]:].tolist()):
            assert row['max_new_tokens'] == limit
            stop = next((i for i, token in enumerate(tokens) if token in [151643, 151645]), None)
            ids = tokens if stop is None else tokens[:stop]
            response = {**row, 'stage': 'edited_factorized_carrier',
                        'response': t.decode(ids, skip_special_tokens=True),
                        'generated_ids': ids, 'stop_token': None if stop is None else tokens[stop],
                        'done_reason': 'length' if stop is None else 'stop',
                        'actual_weights_changed': True, 'runtime_hooks': False,
                        'calibration_only': True, 'fresh_validation': False}
            responses.append(response)
            print(row['id'], repr(response['response']), flush=True)
        output.write_text(json.dumps(responses, ensure_ascii=False, indent=2), encoding='utf8')
restore_carrier(m, originals)
manifest.update({'responses_completed': len(responses), 'responses_sha256': sha(output),
                 'completed': True, 'weights_restored_in_memory': True})
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
